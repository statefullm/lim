#ifndef SIGNALS_H
#define SIGNALS_H

#include <signal.h>

// --- Global Interrupt Flags ---
extern volatile sig_atomic_t stop_generation;
extern volatile sig_atomic_t g_was_interrupted;
extern volatile sig_atomic_t g_was_resumed;  // Set by SIGCONT handler after Ctrl+Z / fg

// --- Signal Handler for Task Interruption ---
void sigint_handler(int sig);

// Set up SIGPIPE (ignore) and SIGINT (handler without SA_RESTART).
void setup_signals();

#endif // SIGNALS_H
