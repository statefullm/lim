#include "signals.h"
#include "output.h"
#include <iostream>

using namespace std;

// --- Global Interrupt Flags ---
volatile sig_atomic_t stop_generation = 0;
volatile sig_atomic_t g_was_interrupted = 0;  // Track interrupt across loop iterations
volatile sig_atomic_t g_was_resumed = 0;      // Set by SIGCONT handler after Ctrl+Z / fg

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

    // Detect fg after Ctrl+Z so we can restore readline's terminal state.
    struct sigaction sa_cont{};
    sa_cont.sa_handler = [](int) { g_was_resumed = 1; };
    sigemptyset(&sa_cont.sa_mask);
    sa_cont.sa_flags = 0;
    sigaction(SIGCONT, &sa_cont, nullptr);
}
