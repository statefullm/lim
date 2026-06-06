#include "server.h"
#include "output.h"
#include "filesystem.h"
#include <sys/wait.h>
#include <cstdlib>

// --- LLLM Server Process Management ---
pid_t g_lllm_server_pid = -1;

// Forward declaration for HOME from filesystem.h
extern const string HOME;  // Actually FileSystemTools::HOME, but we need it

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

    if (is_lllm_server_running()) {
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

    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        string cmd = "exec taskset -c 16-23 /usr/bin/python "+string(home_env)+"/lllm/lllmServer.py";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        g_lllm_server_pid = pid;
    }
}

void cleanup_lllm_server() {
    if (g_lllm_server_pid > 0) {
        kill(-g_lllm_server_pid, SIGKILL);
        waitpid(g_lllm_server_pid, NULL, 0);
        g_lllm_server_pid = -1;
    }
    unlink(FIFO_PATH);
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
    message("\n\033[1;35m[WARNING: No browser connected!]\033[0m\n");
    message("Output will be lost if you don't view it in the browser.\n");
    message("Please load or reload:\n");
    message("  \033[1;35m[" + get_viewer_url() + "\033[0m\n");
    message("Press Enter when ready... ");
    cout.flush();

    char input[256];
    if (!fgets(input, sizeof(input), stdin)) {
        disable_browser_output();
        stop_generation = 0;
        return false;
    }

    int retries = 5;
    while (retries > 0) {
        if (check_browser_connected()) {
            message("\033[1;32m[Browser connected! Ready to proceed.]\033[0m");
            return true;
        }

        if (get_output_mode() == 3) break;

        message("\033[1;35mStill disconnected. Press Enter to check again...\033[0m ");
        cout.flush();

        if (!fgets(input, sizeof(input), stdin)) {
            disable_browser_output();
            stop_generation = 0;
            return false;
        }
        retries--;
    }

    message("\n\033[1;35m[No browser detected. Output may be lost.]\033[0m\n");
    g_browser_warning_suppressed = true;
    return false;
}
