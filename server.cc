#include "server.h"
#include "output.h"
#include "filesystem.h"
#include "taskset.h"
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/inotify.h>
#include <cstdlib>

// --- LIM Server Process Management ---
pid_t g_lim_server_pid = -1;

extern const string HOME;

static const char* INOTIFY_DIR = "/tmp";
static const char* SERVER_READY_PATH = "/tmp/lim.server_ready";
static const char* BROWSER_READY_PATH = "/tmp/lim.browser_ready";
static const char* SERVER_PID_PATH = "/tmp/lim.server.pid";
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
            struct timespec ts{};
            ts.tv_sec = 3;
            while (true) {
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, &ts);
                int status = 0;
                pid_t result = waitpid(pid, &status, WNOHANG);
                if (result == pid || result == -1) break;
            }
        }
    }
    fclose(fp);
}

bool is_lim_server_running() {
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

void start_lim_server_if_needed() {
    if (g_lim_server_pid != -1) return;

    // If the port is free but a PID file exists, the server crashed -- kill any
    // zombie process and clean up. If the port is in use, try to kill it first
    // (e.g., stale server from a previous session that was cleaned up but the
    // socket hasn't fully released yet), then wait for the port to free up.
    if (!is_lim_server_running()) {
        kill_stale_server();
    } else {
        log_diagnostic("limServer appears to be running on port 8765. Checking for stale process...");

        // Collect PIDs to kill: first via PID file, then via fuser fallback.
        vector<pid_t> pids_to_kill;

        // Attempt 1: read the PID file (may have been cleaned up by previous session).
        FILE* fp = fopen(SERVER_PID_PATH, "r");
        if (fp) {
            char buf[32];
            if (fgets(buf, sizeof(buf), fp)) {
                pid_t pid = strtol(buf, nullptr, 10);
                if (pid > 1 && kill(pid, 0) == 0) {
                    log_diagnostic("Killing stale server from PID file (pid " + std::to_string(pid) + ")");
                    pids_to_kill.push_back(pid);
                }
            }
            fclose(fp);
        }

        // Attempt 2: use fuser to find any process on the port.
        if (pids_to_kill.empty()) {
            const char* port_str = "8765";
            const char* env_port = getenv("LIM_PORT");
            if (env_port && strlen(env_port) > 0) port_str = env_port;

            string fuser_cmd = "fuser " + string(port_str) + "/tcp 2>/dev/null";
            FILE* ffp = popen(fuser_cmd.c_str(), "r");
            if (ffp) {
                char line[256];
                while (fgets(line, sizeof(line), ffp)) {
                    char* tok = strtok(line, " \t\n");
                    while (tok) {
                        pid_t pid = strtol(tok, nullptr, 10);
                        if (pid > 1 && kill(pid, 0) == 0) {
                            log_diagnostic("Killing stale server from fuser (pid " + std::to_string(pid) + ")");
                            pids_to_kill.push_back(pid);
                        }
                        tok = strtok(nullptr, " \t\n");
                    }
                }
                pclose(ffp);
            }
        }

        // Send SIGKILL to all found processes.
        for (pid_t pid : pids_to_kill) {
            kill(pid, SIGKILL);
        }

        // Block until each process exits, with a 3-second timeout per PID.
        for (pid_t pid : pids_to_kill) {
            struct timespec ts{};
            ts.tv_sec = 3;
            while (true) {
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, &ts);
                int status = 0;
                pid_t result = waitpid(pid, &status, WNOHANG);
                if (result == pid || result == -1) break;
            }
        }

        // Single final check: if the port is still occupied, something else
        // is genuinely listening. Fall back gracefully.
        if (is_lim_server_running()) {
            log_diagnostic("Port 8765 still in use. Attempting to reuse existing server...");
            g_lim_server_pid = -2;
            return;
        }

        log_diagnostic("Stale server cleared, proceeding with startup.");
    }

    const char* home_env = getenv("HOME");
    if (home_env == nullptr) {
        log_diagnostic("ERROR: HOME is not set. Cannot start limServer.", true);
        g_lim_server_pid = -2;
        return;
    }

    unlink(SERVER_READY_PATH);
    unlink(BROWSER_READY_PATH);
    unlink(SERVER_PID_PATH);

    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        string cmd = "exec "+Taskset::e_core_taskset()+"python3 "+string(home_env)+"/lim/limServer.py";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        g_lim_server_pid = pid;
    }
}

bool wait_for_server_ready() {
    return wait_for_file(INOTIFY_DIR, "lim.server_ready", true);
}

void cleanup_lim_server() {
    if (g_lim_server_pid > 0) {
        kill(-g_lim_server_pid, SIGKILL);
        waitpid(g_lim_server_pid, NULL, 0);
        g_lim_server_pid = -1;
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
    const char* env = getenv("LIM_PORT");
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
    const char* env = getenv("LIM_VIEWER_URL");
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
    message("Load this URL in your browser, or reload with Ctrl+Shift+R or Ctrl+2:\n");
    message("  \033[1;35m" + get_viewer_url() + "\033[0m\n");

    if (wait_for_file(INOTIFY_DIR, "lim.browser_ready", false)) {
        return true;
    }

    if (get_output_mode() == 3) {
        message("\n\033[1;35m[Interrupted, but stdout output is enabled. Proceeding.]\033[0m\n");
        return true;
    }

    disable_browser_output();
    return false;
}