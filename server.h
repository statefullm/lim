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

using namespace std;

// --- LLLM Server Process Management ---
extern pid_t g_lllm_server_pid;

bool is_lllm_server_running();
void start_lllm_server_if_needed();
void cleanup_lllm_server();

// --- Browser Connection ---
bool check_browser_connected();
string get_hostname();
string get_viewer_url();
int get_server_port();

void disable_browser_output();
bool prompt_for_browser_connection();

// Forward declaration
extern volatile sig_atomic_t stop_generation;

#endif // SERVER_H
