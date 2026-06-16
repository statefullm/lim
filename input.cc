#include "input.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <poll.h>
#include <cerrno>
#include <signal.h>

// Forward declaration for stop_generation and g_was_interrupted from signals.h
extern volatile sig_atomic_t stop_generation;
extern volatile sig_atomic_t g_was_interrupted;

const char* INPUT_FIFO_PATH = "/tmp/lllm.input.fifo";
int input_fd = -1;

bool use_browser_input() {
    const char* env = getenv("LLLM_INPUT");
    if (env == nullptr) return false;
    return atoi(env) == 1;
}

void init_input_stream() {
    // Remove stale FIFO from a previous run to ensure we create it fresh
    // with the correct permissions (umask may differ between runs).
    unlink(INPUT_FIFO_PATH);
    mkfifo(INPUT_FIFO_PATH, 0666);
    // O_RDWR prevents open() from blocking if no writer is present yet.
    // O_NONBLOCK so reads return EAGAIN instead of blocking indefinitely.
    input_fd = open(INPUT_FIFO_PATH, O_RDWR | O_NONBLOCK);
    if (input_fd < 0) {
        fprintf(stderr, "Warning: Failed to open %s: %s\n", INPUT_FIFO_PATH, strerror(errno));
        // Not fatal -- browser input simply won't work.
    }
}

std::string get_browser_input() {
    std::string buffer;

    while (true) {
        if (stop_generation || g_was_interrupted) {
            return "";  // Interrupted -- return empty like EOF in stdin mode.
        }

        struct pollfd pfd;
        pfd.fd = input_fd;
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 100);  // 100ms timeout for signal responsiveness
        if (ret < 0) {
            if (errno == EINTR) continue;
            return "";  // Real error -- treat as disconnect.
        }
        if (ret == 0) continue;  // Timeout, loop again.

        if (pfd.revents & (POLLIN | POLLHUP)) {
            char chunk[4096];
            ssize_t n = read(input_fd, chunk, sizeof(chunk));
            if (n > 0) {
                buffer.append(chunk, n);
                // Check for complete line.
                size_t nl = buffer.find('\n');
                if (nl != std::string::npos) {
                    std::string line = buffer.substr(0, nl);
                    buffer.erase();  // Discard any leftover partial data.
                    return line;
                }
            } else if (n == 0) {
                // EOF -- writer disconnected.
                return "";
            } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;  // No data yet, loop.
            } else {
                return "";  // Error reading.
            }
        }
    }

    return "";
}

void cleanup_input_stream() {
    if (input_fd >= 0) {
        close(input_fd);
        input_fd = -1;
    }
}

