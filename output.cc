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
const char SEG_THINK      = '\x06';  // Think/reasoning block content
const char SEG_TURN_END   = '\x07';  // Generation complete signal for viewer

// --- FIFO / Pipe Management ---
int pipe_fd = -1;
const char* FIFO_PATH = "/tmp/lim.fifo";

// --- Output Mode Control ---
bool g_browser_warning_suppressed = false;
bool g_stdout_ended_with_newline = true;  // Start assuming clean state (at prompt)

int get_output_mode() {
    if (g_browser_warning_suppressed) return 1;
    const char* env = getenv("LIM_OUTPUT");
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

// --- HTML Escape Contract (mirrors parsers.cc escape_one_token) ---
static void escape_one_token_out(string& str, const string& token) {
    if (token.size() < 2) return;
    char first = token[0];
    const string suffix = token.substr(1);
    size_t start_pos = 0;
    while (start_pos < str.length()) {
        size_t pos = str.find(first, start_pos);
        if (pos == string::npos) break;
        size_t scan = pos + 1;
        while (scan < str.length() && str[scan] == '\\') scan++;
        bool match = true;
        for (size_t k = 0; k < suffix.length(); k++) {
            if (scan + k >= str.length() || str[scan + k] != suffix[k]) { match = false; break; }
        }
        if (match) {
            str.insert(pos + 1, 1, '\\');
            start_pos = pos + 2 + suffix.length();
        } else {
            start_pos = pos + 1;
        }
    }
}

static string html_escape(const string& input) {
    string result = input;
    // Step 1: Recursively escape pre-existing sentinel tokens.
    escape_one_token_out(result, "@lt@");
    escape_one_token_out(result, "@gt@");
    // Step 2: Replace raw characters with sentinel tokens.
    // Note: backtick is NOT escaped -- only special in markdown (handled by marked), not HTML.
    { size_t pos = 0; while ((pos = result.find('&', pos)) != string::npos) { result.replace(pos, 1, "&amp;"); pos += 5; } }
    { size_t pos = 0; while ((pos = result.find('<', pos)) != string::npos) { result.replace(pos, 1, "@lt@"); pos += 4; } }
    { size_t pos = 0; while ((pos = result.find('>', pos)) != string::npos) { result.replace(pos, 1, "@gt@"); pos += 4; } }
    return result;
}

void stream(const string& raw_token) {
    if (!should_output_to_browser()) return;

    string filtered = raw_token;
    size_t pos = 0;
    while ((pos = filtered.find("</div>", pos)) != string::npos) {
        filtered.erase(pos, 6);
    }

    // HTML-escape <, >, & so they render as text in the browser.
    // Uses sentinel-based recursive escape contract (mirrors parsers.cc).
    string escaped = html_escape(filtered);

    pipe_write(&SEG_LLM_TEXT, 1);
    pipe_write(escaped.c_str(), escaped.length());
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

void stream_think(const string& text) {
    if (!should_output_to_browser()) return;
    string escaped = html_escape(text);
    pipe_write(&SEG_THINK, 1);
    pipe_write(escaped.c_str(), escaped.length());
}
