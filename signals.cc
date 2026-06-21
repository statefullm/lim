#include "signals.h"
#include "output.h"
#include <iostream>

using namespace std;

volatile sig_atomic_t stop_generation = 0;
volatile sig_atomic_t g_was_interrupted = 0;  // Track interrupt across loop iterations

// --- Signal Handler for Task Interruption ---
void sigint_handler(int sig) {
  stop_generation = 1;
  g_was_interrupted = 1;  // Track that we were interrupted (persists across loop iterations)
  // Reset terminal color state
  message("\033[0m\n");
  cout.flush();
}

// --- Signal Setup ---
void setup_signals() {
    // Ignore SIGPIPE to prevent crashes when writing to a FIFO whose reader has disconnected.
    // Writes will fail with EPIPE/EAGAIN instead of terminating the process.
    signal(SIGPIPE, SIG_IGN);

    // Set up SIGINT handler without SA_RESTART so it interrupts blocking syscalls (e.g., fgets)
    struct sigaction sa{};
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // No SA_RESTART - allows signals to interrupt syscalls
    sigaction(SIGINT, &sa, nullptr);
}
