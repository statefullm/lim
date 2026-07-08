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
extern pid_t g_lim_server_pid;

bool is_lim_server_running();
void start_lim_server_if_needed();
bool wait_for_server_ready();
void cleanup_lim_server();

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
