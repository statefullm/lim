#ifndef SERVER_H
#define SERVER_H

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pwd.h>
#include <signal.h>

// --- LIM Server Process Management ---
// The browser server is a persistent service: the first lim session that needs it
// starts it detached (double-forked + setsid, reparented to init), and it
// survives lim's exit and crashes so the browser connection never has to be
// re-established between sessions.  Before reuse, lim verifies the running
// server is reading the same FIFO inode lim writes to (via /proc/<pid>/fd).
// lim only ever attaches (reuses) or, for a stale server (previous LIM_PORT,
// or one left on a deleted FIFO node), performs a bounded teardown.  It never
// kills a healthy server, and never kills a process it cannot positively
// identify as limServer.
enum class ServerStart { Reused, Started, Error };
// True when start_lim_server_if_needed() attached to an already-running server
// (the viewer may still be showing a previous session).  Read by session.cc to
// decide whether to stream the "-- New Session --" divider.
extern bool g_lim_server_reused;

bool is_lim_server_running();
// True when a server is listening AND verifiably reading the FIFO inode lim
// writes to.  /reset uses this: a server that fails it (dead, or alive on a
// stale FIFO node) is replaced by start_lim_server_if_needed().
bool is_lim_server_healthy();
ServerStart start_lim_server_if_needed();
bool wait_for_server_ready(long timeout_ms = 15000);

// --- Browser Connection ---
bool check_browser_connected();
std::string get_hostname();
std::string get_viewer_url();
int get_server_port();

void disable_browser_output();
bool prompt_for_browser_connection();

// Forward declaration
extern volatile sig_atomic_t stop_generation;

#endif // SERVER_H
