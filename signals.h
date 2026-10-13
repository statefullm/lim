#ifndef SIGNALS_H
#define SIGNALS_H

#include <signal.h>

// --- Global Interrupt Flags ---
extern volatile sig_atomic_t stop_generation;
extern volatile sig_atomic_t g_was_interrupted;
extern volatile sig_atomic_t g_was_resumed;  // Set by SIGCONT handler after Ctrl+Z / fg
extern volatile sig_atomic_t g_quit_requested;  // Set by SIGTERM handler: graceful quit with auto-save

// --- Signal Handler for Task Interruption ---
void sigint_handler(int sig);

// No-op SIGUSR1 handler: exists only to interrupt blocking waits (no
// SA_RESTART).  A manual `kill -USR1` poke wakes the detached terminal wait.
void sigusr1_handler(int sig);

// Set up SIGPIPE (ignore), SIGHUP (ignore), SIGTERM (graceful quit), SIGINT
// (interrupt) and SIGUSR1 (poke).
void setup_signals();

#endif // SIGNALS_H
