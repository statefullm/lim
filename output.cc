#include "output.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

// --- Segment Constants ---
const char SEG_LLM_TEXT   = '\x02';  // LLM-generated text (rendered through marked)
const char SEG_HTML       = '\x04';  // Any other raw HTML (tool results, user input, dividers)
const char SEG_SPEED      = '\x05';  // Speed/context diagnostic for status bar

// --- FIFO / Pipe Management ---
int pipe_fd = -1;
const char* FIFO_PATH = "/tmp/lllm.fifo";

// --- Output Mode Control ---
bool g_browser_warning_suppressed = false;

static int get_server_port() {
    const char* env = getenv("LLLM_PORT");
    if (env != nullptr && strlen(env) > 0) {
        char* endp = nullptr;
        long val = strtol(env, &endp, 10);
        if (*endp == '\0' && val > 0 && val < 65536) return static_cast<int>(val);
    }
    return 8765;
}

int get_output_mode() {
    if (g_browser_warning_suppressed) return 1;
    const char* env = getenv("LLLM_OUTPUT");
    if (env == nullptr) return 2;  // Default: browser
    char* endp = nullptr;
    long val = strtol(env, &endp, 10);
    if (*endp != '\0') return 3;
    int mode = static_cast<int>(val);
    if (mode < 0 || mode > 3) return 3;
    return mode;
}

bool should_output_to_stdout() {
    int mode = get_output_mode();
    return mode == 1 || mode == 3;
}

bool should_output_to_browser() {
    int mode = get_output_mode();
    return mode == 2 || mode == 3;
}

bool should_output_think_blocks() {
    int mode = get_output_mode();
    return mode == 1 || mode == 2 || mode == 3;
}

bool should_show_tools() {
    const char* env = getenv("LLLM_SHOW_TOOLS");
    if (env == nullptr) return true;
    char* endp = nullptr;
    long val = strtol(env, &endp, 10);
    if (*endp != '\0') return true;
    return val != 0;
}

void init_output_stream() {
    mkfifo(FIFO_PATH, 0666);
    pipe_fd = open(FIFO_PATH, O_RDWR | O_NONBLOCK);
}

static void ensure_pipe_open() {
    if (pipe_fd < 0) {
        pipe_fd = open(FIFO_PATH, O_RDWR | O_NONBLOCK);
    }
}

void pipe_write(const char* data, size_t len) {
    ensure_pipe_open();
    if (pipe_fd < 0) return;

    const size_t max_retries = 50;
    size_t offset = 0;
    unsigned int retry_count = 0;

    while (offset < len) {
        ssize_t res = write(pipe_fd, data + offset, len - offset);
        if (res > 0) {
            offset += static_cast<size_t>(res);
            retry_count = 0;  // Reset retry counter on successful write
        } else if (res < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // FIFO buffer is full - wait for the reader to drain it.
                // Use poll() with a 5-second timeout instead of sleeping blindly.
                retry_count++;
                if (retry_count > max_retries) {
                    // Give up after too many retries to avoid hanging forever.
                    return;
                }

                struct pollfd pfd;
                pfd.fd = pipe_fd;
                pfd.events = POLLOUT;
                int poll_res = poll(&pfd, 1, 5000);  // 5-second timeout
                if (poll_res <= 0) {
                    // Timeout or error - pipe reader may have disconnected.
                    close(pipe_fd);
                    pipe_fd = -1;
                    return;
                }
            } else {
                // Real error (e.g., EPIPE, EBADF) - give up.
                close(pipe_fd);
                pipe_fd = -1;
                return;
            }
        }
    }
}

void stream(const string& raw_token) {
    if (!should_output_to_browser()) return;

    string filtered = raw_token;
    size_t pos = 0;
    while ((pos = filtered.find("</div>", pos)) != string::npos) {
        filtered.erase(pos, 6);
    }

    pipe_write(&SEG_LLM_TEXT, 1);
    pipe_write(filtered.c_str(), filtered.length());
}

void stream_tool_result(const string& html) {
    if (!should_output_to_browser()) return;
    pipe_write(&SEG_HTML, 1);
    pipe_write(html.c_str(), html.length());
}

void stream_html(const string& html) {
    if (!should_output_to_browser()) return;
    pipe_write(&SEG_HTML, 1);
    pipe_write(html.c_str(), html.length());
}

void stream_speed(const string& speed_text) {
    if (!should_output_to_browser()) return;
    pipe_write(&SEG_SPEED, 1);
    pipe_write(speed_text.c_str(), speed_text.length());
}

void clear_viewer() {
    if (!should_output_to_browser()) return;
    const char marker = 0x01; // SOH control character
    pipe_write(&marker, 1);
}
