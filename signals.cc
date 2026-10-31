#include "signals.h"
#include "output.h"
#include <iostream>
#include <pthread.h>

using namespace std;

// --- Global Interrupt Flags ---
volatile sig_atomic_t stop_generation = 0;
volatile sig_atomic_t g_was_interrupted = 0;  // Track interrupt across loop iterations
volatile sig_atomic_t g_was_resumed = 0;      // Set by SIGCONT handler after Ctrl+Z / fg
volatile sig_atomic_t g_quit_requested = 0;   // SIGTERM: graceful quit with auto-save

// --- Signal Handler for Task Interruption ---
void sigint_handler(int sig) {
  stop_generation = 1;
  g_was_interrupted = 1;  // Track that we were interrupted (persists across loop iterations)
  // Reset terminal color state
  message("\033[0m\n");
  cout.flush();
}

// SIGTERM: graceful quit -- stop any in-flight generation, then exit through
// the normal /quit path (history flush + auto-save).  This is how a
// detached (terminal-less) lim is stopped.
void sigterm_handler(int) {
  stop_generation = 1;
  g_quit_requested = 1;
}

// SIGUSR1 ("poke"): deliberately empty.  Its only effect is to interrupt a
// blocking wait (installed without SA_RESTART), which then re-checks its
// exit condition -- e.g., the detached terminal wait re-runs isatty() after
// a manual `kill -USR1` poke.
void sigusr1_handler(int) {}

// --- Signal Setup ---
void setup_signals() {
  // Ignore SIGPIPE to prevent crashes when writing to a FIFO whose reader has disconnected.
  // Writes will fail with EPIPE/EAGAIN instead of terminating the process.
  signal(SIGPIPE, SIG_IGN);

  // Ignore SIGHUP so lim survives terminal disconnection (the pty sends
  // hangup to the session when the SSH connection drops).
  signal(SIGHUP, SIG_IGN);

  // Set up SIGTERM handler (graceful quit with auto-save) without SA_RESTART
  // so it interrupts blocking waits (e.g., the detached terminal wait).
  struct sigaction sa_term{};
  sa_term.sa_handler = sigterm_handler;
  sigemptyset(&sa_term.sa_mask);
  sa_term.sa_flags = 0;
  sigaction(SIGTERM, &sa_term, nullptr);

  // Set up the SIGUSR1 poke handler without SA_RESTART so it interrupts
  // blocking waits.
  struct sigaction sa_usr1{};
  sa_usr1.sa_handler = sigusr1_handler;
  sigemptyset(&sa_usr1.sa_mask);
  sa_usr1.sa_flags = 0;
  sigaction(SIGUSR1, &sa_usr1, nullptr);

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


  // USR1 (manual poke) and SIGTERM (graceful quit) must be handled by the
  // MAIN thread: the blocking waits they interrupt live there, and llama.cpp
  // runs inference on worker threads -- a process-directed signal can be
  // dequeued by ANY thread that doesn't block it, in which case the main
  // thread's wait is never interrupted (a lost poke = no reattach; a lost
  // SIGTERM = `kill lim` on a detached lim silently fails).  Block both here,
  // before any worker thread
  // exists (workers inherit the mask); the main thread re-unblocks them at
  // the top of get_user_input().
  sigset_t main_thread_only{};
  sigemptyset(&main_thread_only);
  sigaddset(&main_thread_only, SIGUSR1);
  sigaddset(&main_thread_only, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &main_thread_only, nullptr);
}
