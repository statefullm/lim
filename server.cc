#include "server.h"
#include "output.h"
#include "filesystem.h"
#include "taskset.h"
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/inotify.h>
#include <cstdlib>

// --- LLLM Server Process Management ---
pid_t g_lllm_server_pid = -1;

extern const string HOME;

static const char* INOTIFY_DIR = "/tmp";
static const char* SERVER_READY_PATH = "/tmp/lllm.server_ready";
static const char* BROWSER_READY_PATH = "/tmp/lllm.browser_ready";
static const char* SERVER_PID_PATH = "/tmp/lllm.server.pid";
static bool marker_file_valid(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) return false;
    int ch = fgetc(fp);
    fclose(fp);
    return ch != EOF;
}

static bool wait_for_file(const char* dir, const char* filename, bool check_existing) {
    if (check_existing) {
        string path = string(dir) + "/" + filename;
        if (marker_file_valid(path.c_str())) return true;
    }

    int fd = inotify_init();
    if (fd < 0) return false;

    int wd = inotify_add_watch(fd, dir, IN_CREATE | IN_MODIFY);
    if (wd < 0) {
        close(fd);
        return false;
    }

    string path = string(dir) + "/" + filename;

    while (true) {
        char buf[4096];
        ssize_t len = read(fd, buf, sizeof(buf));
        if (len < 0) {
            if (errno == EINTR) {
                inotify_rm_watch(fd, wd);
                close(fd);
                return false;
            }
            inotify_rm_watch(fd, wd);
            close(fd);
            return false;
        }

        for (char* p = buf; p < buf + len; ) {
            struct inotify_event* event = (struct inotify_event*)p;
            if (event->len) {
                string name(event->name);
                if (name == filename && marker_file_valid(path.c_str())) {
                    inotify_rm_watch(fd, wd);
                    close(fd);
                    return true;
                }
            }
            p += sizeof(struct inotify_event) + event->len;
        }
    }
}

static void kill_stale_server() {
    FILE* fp = fopen(SERVER_PID_PATH, "r");
    if (!fp) return;
    char buf[32];
    if (fgets(buf, sizeof(buf), fp)) {
        pid_t pid = strtol(buf, nullptr, 10);
        if (pid > 1 && kill(pid, 0) == 0) {
            kill(pid, SIGKILL);
            usleep(100000);
            waitpid(pid, NULL, WNOHANG);
        }
    }
    fclose(fp);
}

bool is_lllm_server_running() {
    const char* commands[] = {
        "ss -tlnp 2>/dev/null | grep -q ':8765 '",
        "netstat -tlnp 2>/dev/null | grep -q ':8765 '",
        "lsof -i :8765 2>/dev/null | grep -q LISTEN",
        "python3 -c \"import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', 8765)); s.close(); exit(1)\" 2>/dev/null || exit 0"
    };

    for (const char* cmd : commands) {
        FILE* fp = popen(cmd, "r");
        if (fp != nullptr) {
            int status = pclose(fp);
            if (WEXITSTATUS(status) == 1) {
                continue;
            } else if (WEXITSTATUS(status) != 0) {
                return true;
            }
        }
    }

    FILE* fp = popen("python3 -c \"import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                     "s.bind(('0.0.0.0', 8765)); s.close();\" 2>/dev/null", "r");
    if (fp != nullptr) {
        int status = pclose(fp);
        return status != 0;
    }

    return false;
}

void start_lllm_server_if_needed() {
    if (g_lllm_server_pid != -1) return;

    // If the port is free but a PID file exists, the server crashed -- kill any
    // zombie process and clean up. If the port is in use, leave it alone.
    if (!is_lllm_server_running()) {
        kill_stale_server();
    } else {
        log_diagnostic("lllmServer is already running on port 8765. Skipping startup.");
        g_lllm_server_pid = -2;
        return;
    }

    log_diagnostic("Spinning up local lllmServer.py...");

    const char* home_env = getenv("HOME");
    if (home_env == nullptr) {
        log_diagnostic("ERROR: HOME is not set. Cannot start lllmServer.", true);
        g_lllm_server_pid = -2;
        return;
    }

    unlink(SERVER_READY_PATH);
    unlink(BROWSER_READY_PATH);
    unlink(SERVER_PID_PATH);

    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        string cmd = "exec "+Taskset::e_core_taskset()+"python3 "+string(home_env)+"/lllm/lllmServer.py";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        g_lllm_server_pid = pid;
    }
}

bool wait_for_server_ready() {
    return wait_for_file(INOTIFY_DIR, "lllm.server_ready", true);
}

void cleanup_lllm_server() {
    if (g_lllm_server_pid > 0) {
        kill(-g_lllm_server_pid, SIGKILL);
        waitpid(g_lllm_server_pid, NULL, 0);
        g_lllm_server_pid = -1;
    }
    unlink(FIFO_PATH);
    unlink(SERVER_READY_PATH);
    unlink(BROWSER_READY_PATH);
    unlink(SERVER_PID_PATH);
}

// --- Browser Connection Status Checking ---

bool check_browser_connected() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(get_server_port());
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return false;
    }

    const char* request = "GET /status HTTP/1.0\r\nHost: localhost\r\n\r\n";
    write(sock, request, strlen(request));

    char buffer[512];
    bool connected = false;
    ssize_t bytes_read = read(sock, buffer, sizeof(buffer) - 1);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        if (strstr(buffer, "\"connected\": true") != nullptr ||
            strstr(buffer, "\"connected\":true") != nullptr) {
            connected = true;
        }
    } else {
        close(sock);
        return false;
    }

    close(sock);
    return connected;
}

int get_server_port() {
    const char* env = getenv("LLLM_PORT");
    if (env != nullptr && strlen(env) > 0) {
        char* endp = nullptr;
        long val = strtol(env, &endp, 10);
        if (*endp == '\0' && val > 0 && val < 65536) return static_cast<int>(val);
    }
    return 8765;
}

string get_hostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        string h(hostname);
        size_t dot_pos = h.find('.');
        if (dot_pos != string::npos) {
            h = h.substr(0, dot_pos);
        }
        return h;
    }
    return "localhost";
}

string get_viewer_url() {
    const char* env = getenv("LLLM_VIEWER_URL");
    if (env != nullptr && strlen(env) > 0) {
        return string(env);
    }
    return "http://" + get_hostname() + ":" + std::to_string(get_server_port()) + "/viewer.html";
}

void disable_browser_output() {
    if (!g_browser_warning_suppressed) {
        message("\n\033[1;35m[Browser output disabled]\033[0m\n");
    }
    g_browser_warning_suppressed = true;
}

bool prompt_for_browser_connection() {
    message("\n\033[1;35m[Waiting for browser connection...]\033[0m\n");
    message("Load this URL in your browser:\n");
    message("  \033[1;35m" + get_viewer_url() + "\033[0m\n");

    if (wait_for_file(INOTIFY_DIR, "lllm.browser_ready", false)) {
        message("\033[1;32m[Browser connected! Ready to proceed.]\033[0m\n");
        return true;
    }

    if (get_output_mode() == 3) {
        message("\n\033[1;35m[Interrupted, but stdout output is enabled. Proceeding.]\033[0m\n");
        return true;
    }

    message("\n\033[1;35m[Browser wait cancelled. Output may be lost.]\033[0m\n");
    g_browser_warning_suppressed = true;
    return false;
}
