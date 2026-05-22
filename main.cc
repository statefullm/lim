#include "llama.h"
#include "common.h"
#include "parsers.h"
#include "filesystem.h"
#include "network.h"
#include "tokens.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <chrono>
#include <signal.h>
#include <cctype>
#include <set>
#include <algorithm>
#include <deque>
#include <functional>
#include <iomanip>
#include <ctime>
#include <clocale>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


// --- Web Viewer Output Support ---
#include <errno.h>

int pipe_fd = -1;
const char* FIFO_PATH = "/tmp/lllm.fifo";

// --- LLLM_OUTPUT Environment Variable Control ---
// Modes: 3=both stdout+browser, 2=browser only (no stdout), 1=stdout only (no browser), 0=no output (system messages still go to stdout)
static bool g_browser_warning_suppressed = false;  // Don't ask again this session

static int get_output_mode() {
    // If browser disconnected (warning suppressed), always use stdout mode 1
    if (g_browser_warning_suppressed) return 1;

    const char* env = getenv("LLLM_OUTPUT");
    if (env == nullptr) return 2;  // Default: browser
    int mode = atoi(env);
    if (mode < 0 || mode > 3) return 3;
    return mode;
}

// --- LLLM Server Process Management ---
static pid_t g_lllm_server_pid = -1;

// Check if lllmServer is already running on port 8765
static bool is_lllm_server_running() {
    // Try multiple methods to check if port 8765 is in use
    const char* commands[] = {
        "ss -tlnp 2>/dev/null | grep -q ':8765 '",
        "netstat -tlnp 2>/dev/null | grep -q ':8765 '",
        "lsof -i :8765 2>/dev/null | grep -q LISTEN",
        "python3 -c \"import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', 8765)); s.close(); exit(1)\" 2>/dev/null || exit 0"
    };

    for (const char* cmd : commands) {
        FILE* fp = popen(cmd, "r");
        if (fp != nullptr) {
            int status = pclose(fp);
            // If command succeeded (returned 0), port is NOT in use - server not running
            // If command failed (returned non-zero), port IS in use - server likely running
            if (WEXITSTATUS(status) == 1) {
                // The Python test returned 1, meaning bind succeeded = port free
                continue;
            } else if (WEXITSTATUS(status) != 0) {
                // Commands like ss/netstat/lsof returned non-zero when grep found a match
                return true;
            }
        }
    }

    // Try the Python socket test directly
    FILE* fp = popen("python3 -c \"import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                     "s.bind(('0.0.0.0', 8765)); s.close();\" 2>/dev/null", "r");
    if (fp != nullptr) {
        int status = pclose(fp);
        // If bind succeeded (status == 0), port is free - server not running
        // If bind failed (status != 0), port is in use - server running
        return status != 0;
    }

    return false;
}

static void start_lllm_server_if_needed() {
    if (g_lllm_server_pid != -1) return;  // Already started by this instance

    // Check if lllmServer is already running on port 8765
    if (is_lllm_server_running()) {
        log_diagnostic("lllmServer is already running on port 8765. Skipping startup.");
        g_lllm_server_pid = -2;  // Mark as externally managed
        return;
    }

    log_diagnostic("Spinning up local lllmServer.py...");

    if (HOME.empty()) {
        log_diagnostic("ERROR: HOME is not set. Cannot start lllmServer.", true);
        g_lllm_server_pid = -2;
        return;
    }

    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        string cmd = "exec taskset -c 16-23 /usr/bin/python "+HOME+"/lllm/lllmServer.py";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        g_lllm_server_pid = pid;
    }
}

static void cleanup_lllm_server() {
    // Only clean up if we started the server ourselves (not externally managed)
    if (g_lllm_server_pid > 0) {
        kill(-g_lllm_server_pid, SIGKILL);
        waitpid(g_lllm_server_pid, NULL, 0);
        g_lllm_server_pid = -1;
    }
    // Remove the FIFO file to clean up resources
    unlink(FIFO_PATH);
    // If g_lllm_server_pid == -2, the server was already running externally - do not kill it
}

// Forward declaration for log_diagnostic from filesystem.h
void log_diagnostic(const string& msg, bool logOnly = false);

bool should_output_to_stdout() {
    int mode = get_output_mode();
    return mode == 1 || mode == 3;
}

bool should_output_to_browser() {
    int mode = get_output_mode();
    return mode == 2 || mode == 3;
}

template<typename... Args>
void format_and_print(Args&&... args) {
    ostringstream oss;
    ((oss << forward<Args>(args)), ...);
    cout << oss.str();
}

#define message(...) format_and_print(__VA_ARGS__)

// --- Server Port Configuration ---
static int get_server_port() {
    // Allow custom port via environment variable (default: 8765)
    const char* env = getenv("LLLM_PORT");
    if (env != nullptr && strlen(env) > 0) {
        return atoi(env);
    }
    return 8765;
}

// --- Browser Connection Status Checking ---

static bool check_browser_connected() {
    // Use raw socket to query HTTP status endpoint - no external process needed
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(get_server_port());
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return false;
    }

    const char* request = "GET /status HTTP/1.0\r\nHost: localhost\r\n\r\n";
    write(sock, request, strlen(request));

    char buffer[512];
    bool connected = false;
    ssize_t bytes_read = read(sock, buffer, sizeof(buffer) - 1);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        if (strstr(buffer, "\"connected\": true") != nullptr ||
            strstr(buffer, "\"connected\":true") != nullptr) {
            connected = true;
        }
    } else {
        close(sock);
        return false;
    }

    close(sock);
    return connected;
}

static string get_hostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        string h(hostname);
        // Get short hostname by removing domain suffix after first dot
        size_t dot_pos = h.find('.');
        if (dot_pos != string::npos) {
            h = h.substr(0, dot_pos);
        }
        return h;
    }
    return "localhost";
}

static string get_viewer_url() {
    // Allow custom viewer URL via environment variable
    const char* env = getenv("LLLM_VIEWER_URL");
    if (env != nullptr && strlen(env) > 0) {
        return string(env);
    }
    // Default to system hostname and configurable port
    return "http://" + get_hostname() + ":" + std::to_string(get_server_port()) + "/viewer.html";
}

// Forward declaration for interrupt flag used in browser connection check
extern volatile sig_atomic_t stop_generation;

static void disable_browser_output() {
    if (!g_browser_warning_suppressed) {
        message("\n\033[1;35m[Browser output disabled]\033[0m\n");
    }
    g_browser_warning_suppressed = true;  // Prevent this message from appearing again this session, and switch to stdout mode
}

static bool prompt_for_browser_connection() {
    message("\n\033[1;35m[WARNING: No browser connected!]\033[0m\n");
    message("Output will be lost if you don't view it in the browser.\n");
    message("Please load or reload:\n");
    message("  \033[1;35m[" + get_viewer_url() + "\033[0m\n");
    message("Press Enter when ready... ");
    cout.flush();

    char input[256];
    if (!fgets(input, sizeof(input), stdin)) {
        // fgets returned NULL - either EOF or interrupted by signal
        disable_browser_output();
        stop_generation = 0;
        return false;
    }

    int retries = 5;
    while (retries > 0) {
        if (check_browser_connected()) {
            message("\033[1;32m[Browser connected! Ready to proceed.]\033[0m");
            return true;
        }

        // When LLLM_OUTPUT=3 (stdout+browser), skip the "Still disconnected" retry loop
        if (get_output_mode() == 3) break;

        message("\033[1;35mStill disconnected. Press Enter to check again...\033[0m ");
        cout.flush();

        if (!fgets(input, sizeof(input), stdin)) {
            disable_browser_output();
            stop_generation = 0;
            return false;
        }
        retries--;
    }

    message("\n\033[1;35m[No browser detected. Output may be lost.]\033[0m\n");
    g_browser_warning_suppressed = true;  // Prevent "[Browser output disabled]" from appearing again this session
    return false;
}

// Check if think blocks should be output to stdout (modes 1, 2, or 3)
bool should_output_think_blocks() {
    int mode = get_output_mode();
    return mode == 1 || mode == 2 || mode == 3;
}

// LLLM_SHOW_TOOLS controls whether tool names are displayed in the browser.
// Default: 1 (on). Set to 0 to suppress. Independent of LLLM_DEBUG / is_debug.
static bool should_show_tools() {
    const char* env = getenv("LLLM_SHOW_TOOLS");
    if (env == nullptr) return true;
    int val = atoi(env);
    return val != 0;
}

template<typename... Args>
void console(Args&&... args) {
  if (should_output_to_stdout())
    ((cout << forward<Args>(args)), ...);
}

// Special function for think block output - always outputs to stdout in modes 1,2,3
// IMPORTANT: Includes color reset at start to prevent thinking output from appearing blue
template<typename... Args>
void console_think(Args&&... args) {
  if (should_output_think_blocks()) {
    cout << "\033[0m";  // Reset colors - thinking should NOT be blue
    ((cout << forward<Args>(args)), ...);
  }
}

static void consoleFlush() {
    if (should_output_to_stdout()) cout.flush();
}

static void consoleThinkFlush() {
    if (should_output_think_blocks()) cout.flush();
}

void init_output_stream() {
    // Create FIFO if it doesn't exist (ignores error if it already exists)
    mkfifo(FIFO_PATH, 0666);

    // TRICK: Open as O_RDWR | O_NONBLOCK.
    // This succeeds immediately even if the Python server isn't running yet,
    // and prevents SIGPIPE crashes if the Python server disconnects.
    pipe_fd = open(FIFO_PATH, O_RDWR | O_NONBLOCK);
}

static void ensure_pipe_open() {
    if (pipe_fd < 0) {
        pipe_fd = open(FIFO_PATH, O_RDWR | O_NONBLOCK);
    }
}

static void pipe_write(const char* data, size_t len) {
    ensure_pipe_open();
    if (pipe_fd >= 0) {
        ssize_t res = write(pipe_fd, data, len);
        if (res < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            close(pipe_fd);
            pipe_fd = -1;
        }
    }
}

// Segment prefix characters for viewer.html segment-based rendering.
// Each type renders independently so markdown in one never swallows another.
static const char SEG_LLM_TEXT   = '\x02';  // LLM-generated text (rendered through marked)
static const char SEG_TOOL_LABEL = '\x03';  // Tool name label (raw HTML)
static const char SEG_HTML       = '\x04';  // Any other raw HTML (tool results, user input, dividers)

void stream(const string& raw_token) {
    if (!should_output_to_browser()) return;
    // Thinking blocks are already filtered at source (see SAFE TERMINAL BUFFERING)
    // This function sends LLM text to browser with a segment prefix so the viewer
    // can render it independently from tool labels/results via marked.parse().

    // Filter out HTML closing tags that might leak from model output
    string filtered = raw_token;
    size_t pos = 0;
    while ((pos = filtered.find("</div>", pos)) != string::npos) {
        filtered.erase(pos, 6);
    }

    pipe_write(&SEG_LLM_TEXT, 1);
    pipe_write(filtered.c_str(), filtered.length());
}

void stream_tool_label(const string& tool_name) {
    if (!should_output_to_browser()) return;
    string html = "\n\n<div class='tool-label'><strong>" + tool_name + "</strong></div>\n\n";
    pipe_write(&SEG_TOOL_LABEL, 1);
    pipe_write(html.c_str(), html.length());
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

void clear_viewer() {
    if (!should_output_to_browser()) return;
    const char marker = 0x01; // SOH control character, never appears in normal text
    pipe_write(&marker, 1);
}


// --- Readline Headers ---
#include <readline/readline.h>
#include <readline/history.h>

using namespace std;
using namespace Tokens;

// --- Model Detection and Chat Template Selection ---
enum class ModelType {
    UNKNOWN,
    CHATML,      // Standard ChatML format (Nemotron, Mistral, etc.)
    LLAMA3       // Llama 3 format
};

// Function to get the appropriate chat template for a model type
string get_chat_template(ModelType model_type) {
    switch (model_type) {
        case ModelType::CHATML:
            return "chatml";
        case ModelType::LLAMA3:
            return "llama3";
        default:
            return "chatml";
    }
}

// Function to detect model type based on vocabulary tokens
ModelType detect_model_type(const llama_vocab * vocab) {
    int32_t n_tokens = llama_vocab_n_tokens(vocab);

    bool has_im_start = false;
    bool has_im_end = false;
    bool has_reasoning_start = false;
    bool has_reasoning_end = false;

    for (llama_token i = 0; i < n_tokens; i++) {
        const char* token_text = llama_vocab_get_text(vocab, i);
        if (token_text == nullptr) continue;

        string text(token_text);

        if (text.find(Tokens::TURN_START) != string::npos) has_im_start = true;
        if (text.find(Tokens::TURN_END) != string::npos) has_im_end = true;
        if (text.find(THINK_START) != string::npos) has_reasoning_start = true;
        if (text.find(THINK_END) != string::npos) has_reasoning_end = true;
    }

    if (has_reasoning_start && has_reasoning_end) return ModelType::CHATML;
    if (has_im_start && has_im_end) return ModelType::CHATML;

    return ModelType::CHATML;
}

// Forward declarations for diagnostic helpers (defined later with chat_log)
static void diag(const string& msg, const char* color);
static void diag_speed(const string& msg);

bool handle_llama_decode_error(llama_context *ctx, llama_batch batch, const char* error_msg = "KV Cache Exhausted. Type 'clear' to reset.", bool should_break = true) {
    int ret = llama_decode(ctx, batch);
    if (ret < -1) {
        diag(error_msg, "\033[31m");
        return false;
    } else if (ret == -1) {
        diag("Invalid input batch: " + std::string(error_msg), "\033[31m");
        return false;
    } else if (ret == 1 || ret == 2) {
        diag(ret == 1 ? error_msg : "Aborted", "\033[31m");
        if (should_break) return false;
        return true;
    }
    return true;
}

// Sync n_past with the actual KV cache position after a decode.
// Per llama.cpp API: on errors/aborts, partially-decoded ubatches may remain
// in the cache, and on ret==1 the cache is fully restored.  Blindly incrementing
// n_past before llama_decode() therefore drifts out of sync.  This helper
// queries the real cache position and corrects n_past accordingly.
static void sync_n_past(llama_context *ctx, int &n_past) {
    llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (max_pos >= 0) {
        n_past = (int)(max_pos + 1);  // next position to write
    }
}

// --- Global Interrupt Flags ---
volatile sig_atomic_t stop_generation = 0;
static volatile sig_atomic_t g_was_interrupted = 0;  // Track interrupt across loop iterations

// --- SEQUENTIAL INTERVENTION MESSAGES ---
static const string SYSTEM_PROMPT_REMINDER = "Follow the system prompt strictly.";

vector<string> loopMessages = {
    "You are in a loop.",
    "You already have this information.",
    "Please proceed.",
    "Continue.",
    "You did this already.",
    "Can we finish?",
    "Let's break out of this loop!"
};
int loopMessageIndex = 0;

string get_next_loop_message() {
    string msg = loopMessages[loopMessageIndex];
    loopMessageIndex = (loopMessageIndex + 1) % loopMessages.size();
    return SYSTEM_PROMPT_REMINDER + " " + msg;
}

// --- Signal Handler for Task Interruption ---
void sigint_handler(int sig) {
  stop_generation = 1;
  g_was_interrupted = 1;  // Track that we were interrupted (persists across loop iterations)
  // Clear any partial line and reset terminal state
  message("\r\033[K\033[0m\n");
  cout.flush();
}

// --- Dummy Log Callback to Silence Llama.cpp ---
bool first_prompt_displayed = false;
bool is_debug = false;
ofstream chat_log;

void dummy_log_callback(enum ggml_log_level level, const char * text, void * user_data) {}

void custom_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    if (first_prompt_displayed) return;
    cerr << text;
}

// --- Diagnostic Logging Helper Definitions ---
static void diag_impl(const string& formatted_line, const string& msg) {
    cout << formatted_line << endl;
    if (chat_log.is_open()) {
        chat_log << "[" << msg << "]" << "\n\n";
        chat_log.flush();
    }
}

static void diag(const string& msg, const char* color) {
    diag_impl(string(color) + "[" + msg + "]\033[0m", msg);
}

static void diag_speed(const string& msg) {
    diag_impl("\033[0m[" + msg + "]\033[0m", msg);
}

// --- Safe Multiline History Handlers ---
void load_history_safe(const char* filename) {
    ifstream in(filename);
    string line;
    while (getline(in, line)) {
        for (char& c : line) { if (c == '\x1E') c = '\n'; }
        add_history(line.c_str());
    }
}

void save_history_safe(const char* filename, const string& input) {
    ofstream out(filename, ios::app);
    string enc = input;
    for (char& c : enc) { if (c == '\n') c = '\x1E'; }
    out << enc << "\n";
}

// --- Macro-Loop Detection ---
class LoopDetector {
private:
    deque<size_t> tool_history;
    map<size_t, int> freq_map;  // O(1) occurrence counts, kept in sync with tool_history
    size_t max_window_size;

    // Consecutive same-tool-name tracking (secondary loop detection).
    // Pure tool-name equality only -- no fuzzy argument comparison.
    deque<string> tool_name_history;
    static const int CONSECUTIVE_THRESHOLD = 5;

    string normalize_str(const string& s) const {
        string tool_name;
        size_t fs = s.find(FUNC_START);
        if (fs != string::npos) {
            size_t gt = s.find('>', fs);
            if (gt != string::npos) {
                tool_name = s.substr(fs, gt - fs);
            }
        }

        // Extract parameter values using PARAM_START / PARAM_END constants.
        // Captures semantic intent regardless of whitespace/formatting differences.
        string param_values;
        string pstart(PARAM_START);
        string pend(PARAM_END);
        size_t ps = 0;
        while ((ps = s.find(pstart, ps)) != string::npos) {
            size_t pe = s.find('>', ps);
            if (pe == string::npos) break;
            size_t pc = s.find(pend, pe);
            if (pc == string::npos) break;
            string value = s.substr(pe + 1, pc - pe - 1);
            // Collapse whitespace within each value
            string collapsed;
            bool last_space = true;
            for (char c : value) {
                if (isspace(c)) {
                    if (!last_space) { collapsed += ' '; last_space = true; }
                } else { collapsed += c; last_space = false; }
            }
            param_values += collapsed + "|";
            ps = pc + pend.length();
        }

        // If no parameters were extracted (malformed tags), fall back to
        // stripping whitespace from the entire tool call body to avoid
        // hash collisions between different calls to the same tool.
        if (param_values.empty()) {
            string fallback;
            for (char c : s) { if (!isspace(c)) fallback += c; }
            return fallback;
        }

        return tool_name + ":" + param_values;
    }

    void add_to_map(size_t h) { freq_map[h]++; }
    void remove_from_map(size_t h) { if (--freq_map[h] == 0) freq_map.erase(h); }

public:
    LoopDetector(size_t window_size = 15) : max_window_size(window_size) {}

    // O(1): Block if this command has appeared >= 2 times in the window.
    // Catches direct repeats (A->A) and cycles (A->B->C->D->E->A).
    bool would_repeat(const string& tool_call) const {
        string norm_tool = normalize_str(tool_call);
        size_t tool_hash = hash<string>{}(norm_tool);
        auto it = freq_map.find(tool_hash);
        return (it != freq_map.end() && it->second >= 2);
    }

    // O(1): Record a tool call; return true if it now appears >= 3 times.
    bool record_and_check(const string& tool_call) {
        string norm_tool = normalize_str(tool_call);
        size_t tool_hash = hash<string>{}(norm_tool);

        tool_history.push_back(tool_hash);
        add_to_map(tool_hash);

        if (tool_history.size() > max_window_size) {
            remove_from_map(tool_history.front());
            tool_history.pop_front();
        }

        return (freq_map[tool_hash] >= 3);
    }

    // O(1): Count only CONSECUTIVE occurrences of the most recent hash.
    int get_loop_strikes() const {
        if (tool_history.empty()) return 0;
        size_t last = tool_history.back();
        int count = 0;
        for (auto it = tool_history.rbegin(); it != tool_history.rend(); ++it) {
            if (*it == last) count++;
            else break;
        }
        return count;
    }

    void clear_history() {
        tool_history.clear();
        freq_map.clear();
        tool_name_history.clear();
    }

    // Extract just the tool name from a tool call string (no parameters).
    string extract_tool_name(const string& tool_call) const {
        size_t fs = tool_call.find(FUNC_START);
        if (fs != string::npos) {
            size_t gt = tool_call.find('>', fs);
            if (gt != string::npos) {
                return tool_call.substr(fs, gt - fs);
            }
        }
        return "";
    }

    // Check if calling this tool would exceed the consecutive same-tool-name threshold.
    // Returns true if we should block. Pure tool-name equality only, no fuzzy matching.
    bool would_consecutive_loop(const string& tool_call) const {
        string tname = extract_tool_name(tool_call);
        if (tname.empty()) return false;
        if (tool_name_history.size() < CONSECUTIVE_THRESHOLD) return false;

        // Count how many times tname appears consecutively at the tail of tool_name_history
        int count = 0;
        for (auto it = tool_name_history.rbegin(); it != tool_name_history.rend(); ++it) {
            if (*it == tname) count++;
            else break;
        }
        return count >= CONSECUTIVE_THRESHOLD;
    }

    // Record a tool name for consecutive tracking. Caps history at 20 entries.
    void record_tool_name(const string& tool_call) {
        string tname = extract_tool_name(tool_call);
        if (tname.empty()) return;
        tool_name_history.push_back(tname);
        while (tool_name_history.size() > 20) {
            tool_name_history.pop_front();
        }
    }

    // Get the current consecutive count for a given tool name at the tail of history.
    int get_consecutive_count(const string& tool_call) const {
        string tname = extract_tool_name(tool_call);
        if (tname.empty()) return 0;
        int count = 0;
        for (auto it = tool_name_history.rbegin(); it != tool_name_history.rend(); ++it) {
            if (*it == tname) count++;
            else break;
        }
        return count;
    }

    static int get_consecutive_threshold() { return CONSECUTIVE_THRESHOLD; }
};

// --- Helper to unescape tags passed by the LLM ---
void replace_all_tags(string& str, const string& from, const string& to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

// --- Helper to validate path parameters (no embedded newlines) ---
static bool param_has_newline(const string& s) {
    return s.find('\n') != string::npos || s.find('\r') != string::npos;
}

static const string PATH_NEWLINE_ERROR = "System Error: Invalid tool format. The path parameter must not contain newlines.";

// --- Tool Execution Logic ---
string execute_tool_call(const string& tool_call, set<string>& clean_files, string& last_grep_req) {
  string result = "";
  string tool_name = "";
  size_t ns = tool_call.find(FUNC_START);
  if (ns != string::npos) {
      ns += string(FUNC_START).length();
      size_t ne = tool_call.find('>', ns);
      if (ne != string::npos) {
          tool_name = tool_call.substr(ns, ne - ns);
      }
  }

  // Check for interrupt before starting tool execution
  if (stop_generation) return "[Tool interrupted by user]";

  if (tool_name == "read_files") {
    vector<string> paths = extract_array_arg_bounded(tool_call, "paths");
    if (!paths.empty()) {
      FileSystemTools fs;
      NetworkTools net;
      result = "Files content:\n";

      // Separate local files from URLs for batched processing
      vector<string> local_paths_to_read;

      for (const auto& path : paths) {
        bool is_url = (path.find("http://") == 0 || path.find("https://") == 0);

        if (!is_url && clean_files.count(path)) {
          // Cache hit - already read this file
          result += "Path: " + path + "\n";
          result += "Content:\n[Content omitted: You already read this file and it has not been modified since your last read. If you need to search for specific code sections or variables, use the search_file tool instead. DO NOT call read_files on this file again.]\n";
          result += "---\n";
        } else if (is_url) {
          // Handle URL - fetch content from network
          auto url_results = net.fetch_urls({path});
          for (const auto& file : url_results) {
            result += "Path: " + file.at("path") + "\n";
            result += "Content:\n" + file.at("content") + "\n";
            if (!file.at("error").empty()) result += "Error: " + file.at("error") + "\n";
            result += "---\n";

            // Cache URL results similarly to files
            if (file.at("error").empty()) clean_files.insert(file.at("path"));
          }
        } else {
          // Local file that needs to be read - collect for batched processing
          local_paths_to_read.push_back(path);
        }
      }

      // Batch read all local files at once (enables PDF detection across all paths)
      if (!local_paths_to_read.empty()) {
        auto results = fs.read_files(local_paths_to_read);
        for (const auto& file : results) {
          result += "Path: " + file.at("path") + "\n";
          result += "Content:\n" + file.at("content") + "\n";
          if (!file.at("error").empty()) result += "Error: " + file.at("error") + "\n";
          result += "---\n";

          if (file.at("error").empty()) clean_files.insert(file.at("path"));
        }
      }
    } else {
      result = "Error: No paths provided to read_files";
    }
  } else if (tool_name == "search_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) return PATH_NEWLINE_ERROR;
    string text = extract_string_arg_bounded(tool_call, "text");
    string begin_str = extract_string_arg_bounded(tool_call, "begin");
    string end_str = extract_string_arg_bounded(tool_call, "end");
    int begin_line = 0;
    int end_line = 0;
    if (!begin_str.empty() && begin_str.find_first_not_of("0123456789") == string::npos) {
      begin_line = atoi(begin_str.c_str());
    }
    if (!end_str.empty() && end_str.find_first_not_of("0123456789") == string::npos) {
      end_line = atoi(end_str.c_str());
    }
    if (!path.empty()) {
      // For line range mode, use path:begin:end as the request key; for text search, use path:text
      string current_req;
      if (begin_line > 0 && end_line >= begin_line) {
        current_req = path + ":" + to_string(begin_line) + ":" + to_string(end_line);
      } else {
        current_req = path + ":" + text;
      }
      if (current_req == last_grep_req) {
          result = "System Error: You just ran this exact search_file. Do not repeat the same search.";
      } else {
          last_grep_req = current_req;
          FileSystemTools fs;
          result = fs.search_file(path, text, begin_line, end_line);
      }
    } else {
      result = "Error: path is required for search_file";
    }
  } else if (tool_name == "write_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) return PATH_NEWLINE_ERROR;
    string content = extract_string_arg_bounded(tool_call, "content");
    content = remove_trailing_spaces(content);
    clean_files.erase(path);
    last_grep_req = "";
    if (!path.empty()) {
      FileSystemTools fs;
      auto result_map = fs.write_file(path, content);
      result = "Status: " + result_map.at("status");
      if (result_map.find("error") != result_map.end()) result += ", Error: " + result_map.at("error");
    } else {
      result = "Error: No path provided to write_file";
    }
  } else if (tool_name == "edit_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) return PATH_NEWLINE_ERROR;
    string old_str = extract_string_arg_bounded(tool_call, "old");
    string new_str = extract_string_arg_bounded(tool_call, "new");
    new_str = remove_trailing_spaces(new_str);
    clean_files.erase(path);
    last_grep_req = "";
    if (!path.empty()) {
      FileSystemTools fs;
      auto result_map = fs.edit_file(path, old_str, new_str);
      result = "Status: " + result_map.at("status");
      if (result_map.find("changes") != result_map.end()) result += ", " + result_map.at("changes");
      if (result_map.find("error") != result_map.end()) result += ", Error: " + result_map.at("error");
    } else {
      result = "Error: No path provided to edit_file";
    }
  } else if (tool_name == "exec_shell") {
    string command = extract_string_arg_bounded(tool_call, "command");
    clean_files.clear();
    last_grep_req = "";
    if (!command.empty()) {
      FileSystemTools fs;
      result = fs.exec_shell(command);
    } else {
      result = "Error: No command provided to exec_shell";
    }
  } else if (tool_name == "web_search") {
    string query = extract_string_arg_bounded(tool_call, "query");
    clean_files.clear();
    last_grep_req = "";
    if (!query.empty()) {
      NetworkTools net;
      result = net.web_search(query);
    } else {
      result = "Error: No query provided to web_search";
    }
  } else {
    result = "Error: Unknown tool call";
  }

  return result;
}

// --- Helper to strip thinking and tool-call XML tags from a string ---
// Used when rendering thinking block output so that internal tool-call syntax
// (e.g., </parameter>) does not leak to stdout.
static void _strip_think_and_tool_tags(string& str) {
    static const vector<string> all_tags = {
        Tokens::THINK_START, Tokens::THINK_END,
        FUNC_START, FUNC_END,
        PARAM_START, PARAM_END
    };
    for (const auto& tag : all_tags) {
        size_t p;
        while ((p = str.find(tag)) != string::npos) {
            str.erase(p, tag.length());
        }
    }
}

// --- Helper to strip all occurrences of given tags from a string ---
static void strip_tags(string& str, const vector<string>& tags) {
    for (const auto& tag : tags) {
        size_t p;
        while ((p = str.find(tag)) != string::npos) {
            str.erase(p, tag.length());
        }
    }
}

string sanitize(string text) {
    vector<string> patterns = {FUNC_START, FUNC_END, TURN_START, TURN_END};
    for (const auto& pattern : patterns) {
      size_t pos = 0;
      while ((pos = text.find(pattern, pos)) != string::npos) {
            text.insert(pos + 1, "\\");
            pos += pattern.length() + 1;
        }
    }
    return text;
}

int main(int argc, char ** argv) {
  setlocale(LC_ALL, "");

  uid_t uid = getuid();
  struct passwd *pw = getpwuid(uid);
  if (pw == nullptr || strcmp(pw->pw_name, "ai") != 0) {
    cerr << "Error: This program must be run as user 'ai'" << endl;
    return 1;
  }

  // HOME is initialized at global scope via g_homeInit

  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != nullptr) {
    ofstream cwd_file(HOME+"/.cwd");
    if (cwd_file.is_open()) {
      cwd_file << cwd << endl;
      cwd_file.close();
    }
  }

  umask(0002);
  atexit([]() {
      cout << "\033[0m";  // Reset terminal colors on exit
      NetworkTools::cleanup_services();
      cleanup_lllm_server();
  });

  // Ignore SIGPIPE to prevent crashes when writing to a FIFO whose reader has disconnected.
  // Writes will fail with EPIPE/EAGAIN instead of terminating the process.
  signal(SIGPIPE, SIG_IGN);

  // Set up SIGINT handler without SA_RESTART so it interrupts blocking syscalls (e.g., fgets)
  struct sigaction sa{};
  sa.sa_handler = sigint_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;  // No SA_RESTART - allows signals to interrupt syscalls
  sigaction(SIGINT, &sa, nullptr);

  float temp = 0.7f;
  bool use_dummy_thought = false;

  if (argc < 2 || argc > 3) {
    cerr << "Usage: " << argv[0] << " <model_path> [temperature]" << endl;
    return 1;
  }

  // Parse optional temperature argument
  if (argc >= 3) {
    temp = atof(argv[2]);
    if (temp == 0.0f) {
      use_dummy_thought = true;
    }
  }

  const char* debug_env = getenv("LLLM_DEBUG");
  if (debug_env != nullptr && strcmp(debug_env, "1") == 0) {
    is_debug = true;
  }

  if (!is_debug) {
    llama_log_set(dummy_log_callback, nullptr);
  } else {
    llama_log_set(custom_log_callback, nullptr);
  }

  mkdir("log", 0775);
  int log_index = 1;
  string log_file_name;
  while (true) {
      log_file_name = "log/" + to_string(log_index);
      ifstream check_file(log_file_name.c_str());
      if (!check_file.good()) break;
      log_index++;
  }
  chat_log.open(log_file_name, ios::app);
  if (!chat_log.is_open()) {
    cerr << "Error: Failed to open log file. The directory isn't writeable by user ai." << endl;
    return 1;
  }

  // Initialize the fast stream pipe
  init_output_stream();

  // Start lllmServer.py if browser output is enabled
  if (should_output_to_browser()) {
      start_lllm_server_if_needed();
  }

  auto log_entry = [&](const string& role, const string& text) {
      if (chat_log.is_open()) {
          string clean_text = text;
          const vector<string> tags_to_remove = {FUNC_START, FUNC_END, TURN_START, TURN_END};
          strip_tags(clean_text, tags_to_remove);
          while (!clean_text.empty() && isspace(clean_text.back())) clean_text.pop_back();
          chat_log << "=== " << role << " ===\n" << clean_text << "\n\n";
          chat_log.flush();
      }
  };

  log_entry("SYSTEM", "Starting LLM Controller Session");

  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  auto mparams = llama_model_default_params();
  mparams.n_gpu_layers = 999;
  mparams.use_mmap = false;
  mparams.use_mlock = true;

  llama_model * model = llama_model_load_from_file(argv[1], mparams);
  if (!model) return 1;

  const llama_vocab * vocab = llama_model_get_vocab(model);
  ModelType model_type = detect_model_type(vocab);

  auto cparams = llama_context_default_params();
  cparams.n_ctx     = 262144;
  cparams.n_batch   = 2048;
  cparams.n_ubatch  = 512;
  cparams.n_threads = 8;
  cparams.n_threads_batch = 8;
  cparams.flash_attn_type = (llama_flash_attn_type)1;
  cparams.offload_kqv = true;
  cparams.type_k = GGML_TYPE_Q8_0;
  cparams.type_v = GGML_TYPE_Q8_0;

  llama_context * ctx = llama_init_from_model(model, cparams);
  if (!ctx) return 1;

  string system_prompt;

  ifstream prompt_file(HOME+"/prompt");
  if (prompt_file.is_open()) {
    stringstream buffer;
    buffer << prompt_file.rdbuf();
    system_prompt = buffer.str();
    prompt_file.close();
  }

  auto tokenize = [&](string text) { return common_tokenize(ctx, text, false, true); };

  string formatted_system_prompt = string(TURN_START) + "system\n" + system_prompt + TURN_END + "\n";
  vector<llama_token> system_tokens = common_tokenize(ctx, formatted_system_prompt, true, true);

  // Sampling parameters: instruct mode for general tasks
  float top_p = 0.8f;
  int32_t top_k = 20;
  float min_p = 0.0f;
  float penalty_present = 1.5f;
  float penalty_repeat = 1.0f;

  // If temperature is 0, disable thinking (dummy thought injection mode)
  if (use_dummy_thought) {
    top_k = 1;
    top_p = 1.0f;
  }

  llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
  llama_sampler * smpl = llama_sampler_chain_init(lparams);
  llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, penalty_repeat, 0.0f, penalty_present));
  llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
  llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
  int n_past = 0;
  batch.n_tokens = 0;
  for (size_t i = 0; i < (int)system_tokens.size(); i++)
    common_batch_add(batch, system_tokens[i], n_past++, {0}, (i == (int)system_tokens.size() - 1));

  if (!handle_llama_decode_error(ctx, batch)) return 1;

  bool auto_continue = false;
  bool reincarnate_mode = false;  // True when waiting for LLM to write ~/userprompt via reincarnate
  bool prev_was_interrupted = false;  // True if the previous generation turn was interrupted by Ctrl+C
  bool first_turn_done = false;  // Suppress Speed diagnostic on the very first turn only
  int last_t_count = 0;          // Cached stats printed right before >>>
  double last_elapsed = 0.0;
  int last_n_past = 0;

  const char* history_file = ".lllm_history";
  load_history_safe(history_file);

  set<string> clean_files;
  string last_grep_req = "";
  LoopDetector loop_guard(15);
  int invalid_tool_strikes = 0;  // Consecutive invalid/malformed tool calls

  // --- Shared helpers to avoid duplication across clear / reincarnate / input processing ---

  // Feed a vector of tokens into the KV cache, batching as needed.
  // Returns true on success, false on error or interrupt.
  auto feed_tokens = [&](const vector<llama_token>& toks) -> bool {
      batch.n_tokens = 0;
      for (size_t i = 0; i < (int)toks.size(); i++) {
          if (stop_generation) return false;
          common_batch_add(batch, toks[i], n_past++, {0}, (i == (int)toks.size() - 1));
          if (batch.n_tokens == (int)cparams.n_batch && i != (int)toks.size() - 1) {
              if (!handle_llama_decode_error(ctx, batch)) { sync_n_past(ctx, n_past); return false; }
              batch.n_tokens = 0;
          }
      }
      if (batch.n_tokens > 0) {
          if (!handle_llama_decode_error(ctx, batch, "KV Cache Exhausted. Type 'clear' to reset.", false)) {
              sync_n_past(ctx, n_past);
              return false;
          }
          // Even on apparent success, llama_decode may have returned 1 (context full)
          // with should_break=false, leaving n_past drifted ahead of the real cache.
          // Sync to keep our bookkeeping consistent with the actual KV cache position.
          sync_n_past(ctx, n_past);
      }
      return true;
  };

  // Clear the KV cache and re-encode the system prompt.
  auto clear_context = [&]() {
      llama_memory_clear(llama_get_memory(ctx), true);
      n_past = 0;
      feed_tokens(system_tokens);
  };

  // Reset only the LLM-facing state (lighter reset for mid-session recovery).
  // Does NOT touch web search cache, context usage, or browser warning.
  auto reset_llm_state = [&]() {
      clean_files.clear();
      last_grep_req = "";
      loop_guard.clear_history();
      invalid_tool_strikes = 0;
      llama_sampler_reset(smpl);
  };

  // Reset all session-level state variables (full reset for clear/reincarnate).
  auto reset_session_state = [&]() {
      reset_llm_state();
      NetworkTools().reset_search();
      NetworkTools::reset_context_usage();
      g_browser_warning_suppressed = false;
  };

  // --- MAIN CHAT TURN LOOP ---
  while (true) {
    string user_input = "";
    stop_generation = 0;
    g_was_interrupted = 0;  // Reset interrupt flag for this iteration

    if (!auto_continue) {
      // Print Speed from previous generation right before >>> (skip first turn)
      if (first_turn_done && last_t_count > 0) {
          double context_percent = (last_n_past / (double)cparams.n_ctx) * 100.0;
          ostringstream oss;
          oss << fixed << setprecision(1);
          oss << "Speed: " << (last_t_count / last_elapsed) << " t/s | Context: " << last_n_past << "/" << cparams.n_ctx << " (" << context_percent << "%) | Elapsed: " << last_elapsed << "s";
          string speed_line = oss.str();
          diag_speed(speed_line);
      }

      while (true) {
        const char* main_p = "\001\033[1;96m\002>>> \001\033[96m\002";
        const char* cont_p = "\001\033[1;96m\002... \001\033[96m\002";
        if (!first_prompt_displayed) first_prompt_displayed = true;

        // Clear any leftover terminal state from previous interrupt
        console("\r\033[K");
        consoleFlush();

        if (user_input.empty()) console("\n");
        char* input_c = readline(user_input.empty() ? main_p : cont_p);

        cout << "\033[0m" << endl;

        if (!input_c) {
          // EOF (Ctrl+D) or readline interrupted by SIGINT (Ctrl+C) - both return to prompt
          g_was_interrupted = 0;
          console("\r\033[K");
          consoleFlush();
          break;
        }

        string line(input_c);
        free(input_c);

        // Skip empty lines to avoid processing blank input
        if (line.empty()) {
          continue;
        }

        if (line == "quit" || line == "exit" || line == "clear" || line == "reset" || line == "reincarnate") {
          user_input = line;
          break;
        }

        if (!line.empty() && line.back() == '\\') {
          line.pop_back();
          user_input += line + "\n";
        } else {
          user_input += line;
          if (!user_input.empty()) {
            save_history_safe(history_file, user_input);
            add_history(user_input.c_str());
          }
          break;
        }
      }
    }

    if (user_input == "quit" || user_input == "exit") break;

    if (user_input == "clear") {
        clear_context();
        auto_continue = false;
        prev_was_interrupted = false;  // Context cleared, no orphaned turn
        reset_session_state();
        last_t_count = 0;
        last_elapsed = 0.0;
        last_n_past = 0;
        log_entry("SYSTEM", "Context Cleared");
        clear_viewer();
        diag("Context Cleared Successfully", "\033[32m");
        continue;
    }

    if (user_input == "reset") {
        reset_llm_state();
        log_entry("SYSTEM", "Loop Counter and File Cache Reset");
        diag("Loop Counter and File Cache Reset Successfully", "\033[32m");
        continue;
    }

    if (user_input == "reincarnate") {
        // Step 1: Read the reincarnate instruction file
        string reincarnate_path = string(HOME) + "/reincarnate";
        ifstream reincarnate_file(reincarnate_path);
        if (!reincarnate_file.is_open()) {
            diag("Reincarnate failed: Cannot open " + reincarnate_path, "\033[31m");
            continue;
        }
        stringstream reincarnate_buffer;
        reincarnate_buffer << "Use the write_file tool to write a new prompt to "
                      << HOME << "/userprompt. Read the following instructions and compose an appropriate prompt, then write it. "
                      << reincarnate_file.rdbuf();
        string reincarnate_text = reincarnate_buffer.str();
        reincarnate_file.close();

        // Truncate userprompt so the LLM doesn't read a stale/old session when composing the new one.
        string userprompt_check_path = string(HOME) + "/userprompt";
        {
            ofstream userprompt_clear(userprompt_check_path, ios::trunc);
            if (!userprompt_clear.is_open()) {
                diag("Warning: Cannot truncate " + userprompt_check_path + " before reincarnate.", "\033[33m");
            }
        }

        // Step 2: Feed the reincarnate request and let normal generation handle tool calls
        diag("Sending reincarnate request to LLM...", "\033[35m");
        log_entry("USER", "[reincarnate] " + reincarnate_text);

        // If the previous turn was interrupted, the assistant's partial response is still
        // open in the KV cache (no closing ). Close it first so the LLM sees
        // a clean turn boundary and gives the reincarnate instruction its undivided attention.
        string reincarnate_message;
        if (prev_was_interrupted) {
            reincarnate_message = string(TURN_END) + "\n" + string(TURN_START) + "user\n" + reincarnate_text + TURN_END + "\n" + TURN_START + "assistant\n";
        } else {
            reincarnate_message = string(TURN_START) + "user\n" + reincarnate_text + TURN_END + "\n" + TURN_START + "assistant\n";
        }
        prev_was_interrupted = false;  // Consumed
        vector<llama_token> reincarnate_tokens = tokenize(reincarnate_message);

        if (n_past + (int)reincarnate_tokens.size() >= (int)cparams.n_ctx) {
            string ctx_diag = n_past != last_n_past ? " (n_past=" + std::to_string(n_past) + " + " + std::to_string(reincarnate_tokens.size()) + ", last_n_past=" + std::to_string(last_n_past) + ")" : "";
            diag("Context Limit Reached! Cannot process reincarnate" + ctx_diag + ". Type 'clear' to reset.", "\033[31m");
            continue;
        }

        if (!feed_tokens(reincarnate_tokens)) {
            if (stop_generation) stop_generation = 0;
            continue;
        }

        auto_continue = true;
        reincarnate_mode = true;
        reset_session_state();
        continue;
    }

    if (user_input.empty() && !auto_continue) continue;

    // Check browser connection BEFORE processing prompt - ensures user loads browser first
    bool browser_connected = check_browser_connected();

    // If we were suppressed but now connected, reset the flag to restore original output mode
    if (browser_connected && g_browser_warning_suppressed) {
        g_browser_warning_suppressed = false;  // Restore browser output mode
    }

    if (should_output_to_browser()) {
        // If not suppressed and not connected, prompt for browser connection
        if (!g_browser_warning_suppressed && !browser_connected) {
            browser_connected = prompt_for_browser_connection();
            // If still disconnected after prompting, disable browser output for this generation
            // Note: We do NOT close pipe_fd here - just let stream() check should_output_to_browser()
            if (!browser_connected) {
                disable_browser_output();
            }
        }
    }

    if (!user_input.empty()) {
      if (!auto_continue) {
          log_entry("USER", user_input);
          // Write user input HTML directly to pipe with segment prefix
          if (should_output_to_browser() && pipe_fd >= 0) {
              ensure_pipe_open();
              if (pipe_fd >= 0) {
                  // Escape HTML entities for safe display inside <pre><code>
                  string escaped_user;
                  for (char c : user_input) {
                      if (c == '&') escaped_user += "&amp;";
                      else if (c == '<') escaped_user += "&lt;";
                      else if (c == '>') escaped_user += "&gt;";
                      else if (c == '"') escaped_user += "&quot;";
                      else escaped_user += c;
                  }
                  string user_html = "\n\n<div style=\"color: #79c0ff;\"><pre><code>" + escaped_user + "</code></pre></div>\n\n";
                  stream_html(user_html);
              }
          }
      }

      string user_message;

      // If the previous generation was interrupted, close the orphaned assistant turn
      // so the LLM sees a clean boundary and gives this input its full attention.
      string turn_close = prev_was_interrupted ? (string(TURN_END) + "\n") : "";
      prev_was_interrupted = false;

      if (use_dummy_thought) {
          // Dummy Thought Injection - disable thinking mode
          user_message = turn_close + string(TURN_START) + "user\n" +
                           user_input + TURN_END + "\n" +
                           TURN_START + "assistant\n"+THINK_START+"\nThe user wants a direct answer. I will output the requested data immediately without preamble.\n"+THINK_END+"\n";
      } else {
          // Normal thinking mode
          user_message = turn_close + string(TURN_START) + "user\n" + user_input + TURN_END + "\n" + TURN_START + "assistant\n";
      }
      vector<llama_token> tokens = tokenize(user_message);

      if (n_past + (int)tokens.size() >= (int)cparams.n_ctx) {
          string ctx_diag = n_past != last_n_past ? " (n_past=" + std::to_string(n_past) + " + " + std::to_string(tokens.size()) + ", last_n_past=" + std::to_string(last_n_past) + ")" : "";
          diag("Context Limit Reached! Cannot process input" + ctx_diag + ". Type 'clear' to reset.", "\033[31m");
          continue;
      }

      if (!feed_tokens(tokens)) {
        if (stop_generation) {
          diag("Input Evaluation Interrupted", "\033[31m");
          stop_generation = 0;
        }
        continue;
      }
    }

    auto start = chrono::high_resolution_clock::now();
    int t_count = 0;
    string generated_text = "";
    string unprinted_text = "";
    string full_response = "";
    size_t print_pos = 0;

    generated_text.reserve(32768);
    unprinted_text.reserve(1024);
    full_response.reserve(32768);

    bool in_tool_call_stream = false;
    size_t tool_start = string::npos;
    size_t tool_end = string::npos;
    bool trigger_tool_execution = false;
    size_t func_search_pos = 0;
    bool had_eog_recovery = false;  // If true, suppress speed diagnostic (elapsed includes wasted resampling)
    bool context_warned_this_turn = false;  // Track 90% warning per generation turn

    // Track thinking blocks (similar to tool calls)
    bool in_thinking_block = false;
    size_t think_start = string::npos;
    size_t think_end = string::npos;
    string think_buffer;        // Buffer for initial whitespace when LLLM_OUTPUT=2
    bool think_buffering = true;  // True while buffering (haven't found non-whitespace yet) in mode 2

    // Track if stdout ended with newline from previous iteration's tool output
    static bool prev_stdout_ended_with_newline = false;

    // --- INNER TOKEN GENERATION LOOP ---
    while (true) {
      if (stop_generation) {
        diag("Task Interrupted by User", "\033[31m");
        stream("\n\n[Task Interrupted by User]\n\n");
        stop_generation = 0;
        g_was_interrupted = 0;  // Reset for next iteration
        prev_was_interrupted = true;  // Mark that assistant turn was left unclosed
        auto_continue = false;
        reincarnate_mode = false;  // Abort any in-progress reincarnate so post-generation doesn't run on stale state
        // Force readline to reset its display state after interrupt
        rl_redisplay();
        break;
      }

      // Proactive context-full warning at 90%: give the user a chance to reincarnate or clear.
      {
          int context_90pct = (int)(cparams.n_ctx * 0.9);
          if (n_past >= context_90pct && !context_warned_this_turn) {
              context_warned_this_turn = true;

              if (!unprinted_text.empty()) {
                  console(unprinted_text.c_str());
                  consoleFlush();
                  stream(unprinted_text);
                  unprinted_text = "";
              }

              diag("Context approaching limit (" + std::to_string(n_past) + "/" + std::to_string(cparams.n_ctx) + "). Type 'reincarnate' to start a fresh session, or 'clear' to reset.", "\033[1;33m");
          }
      }

      if (n_past >= (int)cparams.n_ctx - 10) {
        string ctx_diag = n_past != last_n_past ? " (n_past=" + std::to_string(n_past) + ", last_n_past=" + std::to_string(last_n_past) + ", n_ctx=" + std::to_string(cparams.n_ctx) + ")" : "";
        diag("Context Window Exhausted!" + ctx_diag + ". Type 'clear' to reset.", "\033[31m");
        if (!unprinted_text.empty()) {
            console(unprinted_text.c_str());
            consoleFlush();
            stream(unprinted_text);
        }
        auto_continue = false;
        reincarnate_mode = false;  // Abort any in-progress reincarnate so post-generation doesn't run on stale state
        break;
      }

      if (batch.n_tokens < 1) {
          diag("Error: no tokens in batch to sample from", "\033[31m");
          auto_continue = false;
          reincarnate_mode = false;  // Abort any in-progress reincarnate so post-generation doesn't run on stale state
          break;
      }
      llama_token next_token = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);

      // --- EARLY EOG RECOVERY / SPURIOUS EOG WORKAROUND ---
      // When the model emits an EOG token, it may be spurious -- drawn by chance from a
      // distribution where EOG has elevated but not dominant probability (common in Q4
      // quantized models). Each call to llama_sampler_sample() produces a fresh random
      // draw from the same logits. If we get a non-EOG token on a subsequent draw, the
      // original EOG was spurious and we continue generating seamlessly.
      //
      // Resample up to 10 times (configurable via LLLM_EOG_RESAMPLE_MAX). In practice most
      // recoveries succeed on the first non-blocking check with zero latency cost.
      if (llama_vocab_is_eog(vocab, next_token)) {
          size_t active_ts = generated_text.rfind(FUNC_START);
          size_t active_te = generated_text.rfind(FUNC_END);

          bool inside_unclosed_tool = (active_ts != string::npos && (active_te == string::npos || active_ts > active_te));

          // --- DIAGNOSTIC COUNTERS ---
          static int eog_event_count = 0;
          static int total_poll_iters = 0;
          static int max_poll_iters = 0;
          static int last_printed = 0;
          int poll_iter_used = 0;

                    // No actual sleep between resamples -- sampling from pre-computed logits is CPU-only.
          // LLLM_EOG_RESAMPLE_MAX overrides the default (see below).
          static constexpr int DEFAULT_EOG_RESAMPLE_MAX = 100;
          static constexpr int EOG_RESAMPLE_HARD_CAP    = 200;

          const char* eog_env = getenv("LLLM_EOG_RESAMPLE_MAX");
          int max_iterations = (eog_env != nullptr && strlen(eog_env) > 0) ? atoi(eog_env) : DEFAULT_EOG_RESAMPLE_MAX;
          max_iterations = std::max(1, std::min(max_iterations, EOG_RESAMPLE_HARD_CAP));

          bool recovered = false;

          // Non-blocking first check: immediate resample before any sleep.
          // ~50% of spurious EOG recoveries succeed here with zero latency cost.
          {
              llama_token polled = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);
              if (!llama_vocab_is_eog(vocab, polled)) {
                  next_token = polled;
                  recovered = true;
                  poll_iter_used = 1;
              }
          }

          // If first check was also EOG, enter the polling loop -- no sleep needed.
          // llama_sampler_sample() draws from pre-computed logits (CPU-only operation).
          // There is nothing to wait for; sleeping only adds latency.
          if (!recovered) {
              for (int poll_iter = 0; poll_iter < max_iterations; ++poll_iter) {
                  if (stop_generation) break;
                  // No sleep needed -- logits are pre-computed, sampling is CPU-only.
                  llama_token polled = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);
                  if (!llama_vocab_is_eog(vocab, polled)) {
                      next_token = polled;
                      recovered = true;
                      poll_iter_used = poll_iter + 2; // account for initial non-blocking check
                      break;
                  }
              }
          }

          if (recovered) {
              eog_event_count++;
              total_poll_iters += poll_iter_used;
              max_poll_iters = std::max(max_poll_iters, poll_iter_used);

              string recovered_piece = common_token_to_piece(ctx, next_token);
              if (is_debug) {
                  message("\033[90m[EOG recovery: token=" + recovered_piece + " | polls=" + std::to_string(poll_iter_used) + "/" + std::to_string(max_iterations) + "]\033[0m\n");
                  cout.flush();

                  if (eog_event_count - last_printed >= 10) {
                      double avg = (double)total_poll_iters / eog_event_count;
                      char avg_buf[16];
                      snprintf(avg_buf, sizeof(avg_buf), "%.1f", avg);
                      message("\033[90m[EOG diagnostic: " + std::to_string(eog_event_count) + " spurious EOGs recovered | "
                          "avg polls: " + string(avg_buf) + "/" + std::to_string(max_iterations) +
                          " | max: " + std::to_string(max_poll_iters) + "/" + std::to_string(max_iterations) + "]\033[0m\n");
                      cout.flush();
                      last_printed = eog_event_count;
                  }
              }
              // Successfully recovered -- fall through to process the real token normally.
              had_eog_recovery = true;
          }

          // Polling exhausted. If inside an unclosed tool call, force recovery with a
          // premature closure so the partial tool is still executed.
          if (!recovered) {
              if (inside_unclosed_tool) {
                  if (is_debug) {
                      message("\033[31m[System: Premature End-Of-Turn detected after polling timeout. Auto-recovering tags...]\033[0m\n");
                      cout.flush();
                  }

                  size_t trailing_slash = generated_text.rfind("</");
                  if (trailing_slash != string::npos && trailing_slash > active_ts) {
                      size_t drop_len = generated_text.length() - trailing_slash;
                      generated_text.erase(trailing_slash);
                      full_response.erase(full_response.length() - drop_len);
                  }

                  string forced_close = "\n" + string(TURN_END) + "\n" + string(FUNC_END) + "\n";
                  generated_text += forced_close;
                  full_response += forced_close;

                  tool_start = active_ts;
                  tool_end = generated_text.find(FUNC_END, active_ts);
                  trigger_tool_execution = true;
              } else {
                  // Genuine end of turn (no more tokens and not inside a tool call)
                  auto_continue = false;
              }
              // All non-recovered EOG paths break out of the inner loop
              break;
          }
          // Recovered: fall through to process next_token as a normal token
      }

      string token_str = common_token_to_piece(ctx, next_token).c_str();
      generated_text += token_str;
      full_response += token_str;

      // --- THINKING BLOCK TRACKING ---
      // Must run BEFORE tool call tracking so we can skip tool calls inside think blocks.
      if (think_start == string::npos) {
          think_start = generated_text.find(Tokens::THINK_START);
      }
      if (think_start != string::npos && think_end == string::npos) {
          think_end = generated_text.find(Tokens::THINK_END, think_start);
      }
      in_thinking_block = (think_start != string::npos && think_end == string::npos);

      // --- PERF OPTIMIZATION: O(1) TRACKING OFFSETS ---
      if (tool_start == string::npos) {
          size_t search_from = func_search_pos;
          while (true) {
              tool_start = generated_text.find(FUNC_START, search_from);
              if (tool_start == string::npos) break;

              // Skip tool calls that appear inside a thinking block (open or closed).
              // Tool calls in <think>...</think> are internal reasoning, not actual invocations.
              if (think_start != string::npos &&
                  tool_start >= think_start &&
                  (think_end == string::npos || tool_start < think_end)) {
                  search_from = think_end;
                  continue;
              }
              break;
          }
          if (tool_start == string::npos) {
              func_search_pos = generated_text.length() > 20 ? generated_text.length() - 20 : 0;
          }
      }
      if (tool_start != string::npos && tool_end == string::npos) {
          tool_end = generated_text.find(FUNC_END, tool_start);
      }

      in_tool_call_stream = (tool_start != string::npos && tool_end == string::npos);

      // --- TARGETED SYNTAX TRAP (Stutter Fix) ---
      if (in_tool_call_stream && generated_text.length() >= 4 &&
          generated_text.compare(generated_text.length() - 4, 4, DOUBLE_OPEN) == 0) {

          diag("System: Infinite slash loop detected. Auto-recovering...", "\033[31m");

          size_t bad_pos = generated_text.rfind(DOUBLE_OPEN);
          if (bad_pos != string::npos && bad_pos > tool_start) {
              size_t drop_len = generated_text.length() - bad_pos;
              generated_text.erase(bad_pos);
              full_response.erase(full_response.length() - drop_len);
          }

          string forced_close = "\n" + string(TURN_END) + "\n" + string(FUNC_END) + "\n";
          generated_text += forced_close;
          full_response += forced_close;

          tool_end = generated_text.find(FUNC_END, tool_start);
          trigger_tool_execution = true;
          break;
      }

      // --- NORMAL TOOL COMPLETION ---
      if (tool_end != string::npos) {
          trigger_tool_execution = true;
          break; // The tool tag is closed. Execute!
      }

      // --- SAFE TERMINAL BUFFERING ---
      if (!in_tool_call_stream && !in_thinking_block) {
          size_t safe_len = generated_text.length();
          string fstart(FUNC_START);
          string tstart(Tokens::THINK_START);

          // CRITICAL: If we just exited a thinking block (think_end found), handle THINK_END
          if (think_start != string::npos && think_end != string::npos) {
              size_t think_block_end = think_end + string(Tokens::THINK_END).length();

              // Clear the buffer for all modes - empty blocks are always discarded
              think_buffer.clear();
              think_buffering = true;

              if (print_pos < think_block_end) {
                  // Only print the closing tag if we actually output content from this block
                  // (think_buffering == false means we found non-whitespace and started streaming)
                  if (!think_buffering && print_pos <= think_end) {
                      string close_tag = generated_text.substr(print_pos, think_block_end - print_pos);
                      console_think(close_tag.c_str());
                      consoleThinkFlush();
                  }
                  print_pos = think_block_end;
              }
          }

          // Hold back printing if the tail end is a partial match for a tool or thinking tag
          for (size_t len = 1; len <= max(fstart.length(), tstart.length()) && len <= generated_text.length(); ++len) {
              if (generated_text.compare(generated_text.length() - len, len, fstart, 0, len) == 0 ||
                  generated_text.compare(generated_text.length() - len, len, tstart, 0, len) == 0) {
                  safe_len = generated_text.length() - len;
                  break;
              }
          }

          if (safe_len > print_pos) {
              unprinted_text += generated_text.substr(print_pos, safe_len - print_pos);
              print_pos = safe_len;
          }

          if (!unprinted_text.empty() && (t_count % 10 == 0 || unprinted_text.back() == '\n')) {
              console(unprinted_text.c_str());
              consoleFlush();
              stream(unprinted_text);
              unprinted_text = "";
          }
      } else if (in_thinking_block) {
          // Inside a thinking block: Output to stdout only, not browser
          size_t safe_len = generated_text.length();
          string tstart(Tokens::THINK_START);
          string tend(Tokens::THINK_END);

          // Hold back printing if the tail end is a partial match for thinking end tag
          for (size_t len = 1; len <= tend.length() && len <= generated_text.length(); ++len) {
              if (generated_text.compare(generated_text.length() - len, len, tend, 0, len) == 0) {
                  safe_len = generated_text.length() - len;
                  break;
              }
          }

          if (safe_len > print_pos) {
              string think_output = generated_text.substr(print_pos, safe_len - print_pos);

              // Always buffer until first non-whitespace to suppress empty blocks regardless of LLLM_OUTPUT
              if (think_buffering) {
                  // Strip thinking tags and tool call tags before checking/accumulating
                  _strip_think_and_tool_tags(think_output);

                  // Check for first non-whitespace character
                  bool found_content = false;
                  size_t content_start = 0;
                  for (size_t i = 0; i < think_output.size(); ++i) {
                      if (!isspace(think_output[i])) {
                          found_content = true;
                          content_start = i;
                          break;
                      }
                  }

                  if (found_content) {
                      // Found non-whitespace: flush buffer, print from content_start onward, switch to streaming
                      think_buffer.clear();
                      think_output = think_output.substr(content_start);
                      console_think(think_output.c_str());
                      consoleThinkFlush();
                      think_buffering = false;  // Switch to direct streaming
                  } else {
                      // Still all whitespace - buffer it (will be discarded if block ends empty)
                      think_buffer += think_output;
                  }
              } else {
                  // Already found content - stream directly, stripping thinking and tool call tags
                  _strip_think_and_tool_tags(think_output);
                  console_think(think_output.c_str());
                  consoleThinkFlush();
              }
              // Do NOT send thinking blocks to browser - stdout only
              print_pos = safe_len;
          }
      } else {
          // Inside a tool call: Flush buffered text and mute the rest
          if (!unprinted_text.empty()) {
              console(unprinted_text.c_str());
              consoleFlush();
              stream(unprinted_text);
              unprinted_text = "";
          }
          print_pos = generated_text.length(); // Advance the cursor silently
      }

      t_count++;
      batch.n_tokens = 0;
      common_batch_add(batch, next_token, n_past++, {0}, true);
      if (!handle_llama_decode_error(ctx, batch)) { sync_n_past(ctx, n_past); reincarnate_mode = false; break; }

    } // END INNER TOKEN LOOP

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    // With segment-based rendering, each LLM text segment is parsed independently.
    // Unclosed code fences in one segment don't affect subsequent HTML segments.

    // Flush any remaining unprinted text before speed info using fold expression for streaming
    bool stdout_ended_with_newline = prev_stdout_ended_with_newline;  // Start with previous iteration's state
    if (!unprinted_text.empty()) {
        console(unprinted_text.back() != '\n' ? (unprinted_text + "\n").c_str() : unprinted_text.c_str());
        consoleFlush();
        stream(unprinted_text + (unprinted_text.back() != '\n' ? "\n" : ""));
        stdout_ended_with_newline = true;
        unprinted_text = "";
    }

    // Deterministic check: if stdout didn't end with newline, add one before Speed diagnostic
    if (!stdout_ended_with_newline) {
        cout << "\n";
    }

    // Save stats for Speed diagnostic before next prompt
    last_t_count = t_count;
    last_elapsed = elapsed;
    last_n_past = n_past;
    first_turn_done = true;

    // Save state for next iteration
    prev_stdout_ended_with_newline = true;

    if (stop_generation) {
      stop_generation = 0;
      auto_continue = false;
    }

    // --- TOOL EXECUTION BLOCK ---
    // If the generation loop exited and signaled a tool is ready, we process it here.
    if (trigger_tool_execution && tool_start != string::npos && tool_end != string::npos) {
        string tool_call = generated_text.substr(tool_start, tool_end - tool_start + string(FUNC_END).length());
        string preamble = "";
        if (tool_start > 0) preamble = generated_text.substr(0, tool_start);

        const vector<string> strip_tags_vec = {TURN_START, TURN_END};
        strip_tags(tool_call, strip_tags_vec);

        string tool_name_for_display = "";
        size_t name_start = tool_call.find(FUNC_START);
        if (name_start != string::npos) {
            name_start += string(FUNC_START).length();
            size_t name_end = tool_call.find('>', name_start);
            if (name_end != string::npos) {
                tool_name_for_display = tool_call.substr(name_start, name_end - name_start);
            }
        }

                // Show the tool name in the browser immediately (segment-based rendering).
        if (should_show_tools()) {
            string tool_label = tool_name_for_display.empty() ? "unknown" : tool_name_for_display;
            stream_tool_label(tool_label);
        }

        unprinted_text = "";

        bool is_real_tool = false;
        vector<string> valid_tools = {
            "read_files", "search_file", "exec_shell", "edit_file", "write_file", "web_search"
        };
        for (const auto& vn : valid_tools) {
          if (tool_name_for_display == vn) {
            is_real_tool = true;
            break;
          }
        }

        string tool_result = "";
        string display_result = "";
        bool was_loop = false;
        bool tool_blocked_by_loop = false;
        bool abort_auto = false;
        bool inject_auto_user_msg = false;
        string active_intervention_msg = "";

        string tool_name = tool_name_for_display;

        if (!is_real_tool) {
          invalid_tool_strikes++;

          // Show the raw malformed tool call in the browser so the user can see what went wrong.
          if (should_show_tools() && should_output_to_browser()) {
              string raw_display = tool_call;
              if (raw_display.length() > 500) {
                  raw_display = raw_display.substr(0, 497) + "...";
              }
              // Escape HTML to prevent injection from tool content
              string safe;
              for (char c : raw_display) {
                  if (c == '&') safe += "&amp;";
                  else if (c == '<') safe += "&lt;";
                  else if (c == '>') safe += "&gt;";
                  else safe += c;
              }
              string error_html = "\n\n<div class='tool-error'>Invalid Tool Call (Strike " + std::to_string(invalid_tool_strikes) + "):<pre><code>" + safe + "</code></pre></div>\n\n";
              stream_tool_result(error_html);
          }

          tool_result = "System Error: Invalid tool format or unsupported tool. You MUST use the strict XML schema.";
          display_result = tool_result;

          // After 2+ consecutive invalid tool calls, inject a system intervention
          // telling the LLM to follow the system prompt strictly.
          if (invalid_tool_strikes >= 5) {
              // Hard limit: intervention failed, eject to prompt.
              diag("System: " + std::to_string(invalid_tool_strikes) + " consecutive invalid tool calls. Intervention failed, ejecting to prompt.", "\033[1;31m");
              abort_auto = true;
          } else if (invalid_tool_strikes >= 2) {
              diag("System: " + std::to_string(invalid_tool_strikes) + " consecutive invalid tool calls. Injecting intervention.", "\033[1;31m");
              inject_auto_user_msg = true;
              active_intervention_msg = SYSTEM_PROMPT_REMINDER;
          }
        } else {
          // Valid tool call - reset the invalid strike counter
          invalid_tool_strikes = 0;
          bool is_mutating_tool = (tool_name == "edit_file" || tool_name == "write_file");

          // PRE-EXECUTION LOOP CHECK: Reject before running if this is an exact repeat.
          // This prevents wasting time on known-repeating commands.
          bool pre_loop = loop_guard.would_repeat(tool_call);

          // PRE-EXECUTION LOOP CHECK (2): Block consecutive same-tool-name calls.
          // Catches loops where the agent varies arguments slightly each time.
          bool pre_consecutive = loop_guard.would_consecutive_loop(tool_call);

          if (pre_loop) {
              // Record it to update strike count, but DON'T execute the tool.
              was_loop = loop_guard.record_and_check(tool_call);
              tool_blocked_by_loop = true;
              int current_strikes = loop_guard.get_loop_strikes();

              active_intervention_msg = get_next_loop_message();
              tool_result = "System Error: Loop Detected -- you already called this exact tool recently. " + active_intervention_msg + " If searching code, use search_file instead of exec_shell.";
              display_result = tool_result;

              diag("System: Pre-execution loop blocked (Strike " + std::to_string(current_strikes) + ").", "\033[35m");

              int max_attempts = loopMessages.size();

              if (current_strikes <= max_attempts) {
                  diag("System: Automating intervention (Attempt " + std::to_string(current_strikes) + "/" + std::to_string(max_attempts) + ").", "\033[35m");
                  abort_auto = false;
                  inject_auto_user_msg = true;
              } else {
                  diag("System: Circuit breaker -- intervention failed after " + std::to_string(max_attempts) + " strikes. Ejecting to prompt.", "\033[1;31m");
                  abort_auto = true;
              }
          } else if (pre_consecutive) {
              // Same tool called too many times in a row with varying arguments.
              // Block execution and inject an intervention message.
              loop_guard.record_and_check(tool_call);
              tool_blocked_by_loop = true;
              loop_guard.record_tool_name(tool_call);  // Keep incrementing so circuit breaker can eventually fire

              int consec = loop_guard.get_consecutive_count(tool_call);

              active_intervention_msg = get_next_loop_message();
              tool_result = "System Error: Consecutive Tool Loop Detected -- you have called " + tool_name + " " + std::to_string(consec) + " times in a row. " + active_intervention_msg + " Stop retrying and try a different approach.";
              display_result = tool_result;

              diag("System: Consecutive-tool loop blocked (" + tool_name + " x" + std::to_string(consec) + ").", "\033[35m");

              int max_attempts = loopMessages.size();

              if (consec <= max_attempts + LoopDetector::get_consecutive_threshold()) {
                  diag("System: Automating intervention (Attempt " + std::to_string(consec - LoopDetector::get_consecutive_threshold() + 1) + ").", "\033[35m");
                  abort_auto = false;
                  inject_auto_user_msg = true;
              } else {
                  diag("System: Circuit breaker -- consecutive intervention failed. Ejecting to prompt.", "\033[1;31m");
                  abort_auto = true;
              }
          } else {
              // Not a repeat -- execute the tool normally.
              tool_result = execute_tool_call(tool_call, clean_files, last_grep_req);

              // Record the execution for future loop detection (exact-match).
              was_loop = loop_guard.record_and_check(tool_call);

              // Also record for consecutive same-tool-name tracking.
              loop_guard.record_tool_name(tool_call);

              // Check for interrupt after tool execution completes
              if (stop_generation) {
                diag("Tool Interrupted by User", "\033[31m");
                stop_generation = 0;
                abort_auto = true;
                reincarnate_mode = false;  // Abort any in-progress reincarnate so post-generation doesn't run on stale state
              }

              if (!abort_auto && was_loop) {
                  // This is the second identical call in a row. On the NEXT call,
                  // pre-execution check will catch it. But we already executed this one.
                  // Still send an intervention message to steer the LLM away.
                  active_intervention_msg = get_next_loop_message();
                  tool_result = "System Warning: You just repeated a tool call. " + active_intervention_msg + " If searching code, use search_file instead of exec_shell.";
                  display_result = tool_result;

                  diag("System: Post-execution loop warning (Strike 2).", "\033[35m");
                  inject_auto_user_msg = true;
              } else if (!abort_auto) {
                  bool tool_failed = (tool_result.find("System Error:") != string::npos || tool_result.find("Error:") != string::npos);

                  if (is_mutating_tool && !tool_failed) loop_guard.clear_history();

                  bool is_expected_error = (tool_result.find("exact match not found") != string::npos ||
                                           tool_result.find("contains the 'old' string") != string::npos);
                  if (is_mutating_tool && is_expected_error) loop_guard.clear_history();

                  if (tool_name == "read_files") {
                      vector<string> paths = extract_array_arg_bounded(tool_call, "paths");
                      display_result = "Read files: ";
                      for (const auto& p : paths) display_result += p + " ";
                      if (tool_result.find("[Content omitted") != string::npos) display_result += "(Cache Hit)";
                  } else if (tool_name == "web_search") {
                      string q = extract_string_arg_bounded(tool_call, "query");
                      display_result = "Web search: " + q;
                  } else {
                      display_result = tool_result;
                  }
              }
          }
        }

        if (!abort_auto) {
            bool has_error = (display_result.find("Error:") != string::npos);
            bool has_match_count = (display_result.find("Match count:") != string::npos);

            if (is_debug) {
              console("\n\033[92m[Tool Result]\033[0m\n");
              string result_to_print = display_result;
              size_t p = 0;
              while ((p = result_to_print.find('\n')) != string::npos) {
                console("  ", (int)p, result_to_print.c_str(),"\n");
                  result_to_print.erase(0, p + 1);
              }
              if (!result_to_print.empty()) console("  ", result_to_print.c_str(),"\n");
            }

            // Always send tool results to the browser (independent of debug mode)
            // Truncate to last 500 chars to avoid overwhelming the viewer (unless debug mode)
            string display_for_browser = display_result;
            if (!is_debug && display_for_browser.length() > 500) {
                display_for_browser = "..." + display_for_browser.substr(display_for_browser.length() - 497);
            }

            // Escape HTML in tool result for safe browser display (no markdown code fences)
            string safe_result;
            for (char c : display_for_browser) {
                if (c == '&') safe_result += "&amp;";
                else if (c == '<') safe_result += "&lt;";
                else if (c == '>') safe_result += "&gt;";
                else safe_result += c;
            }

            string result_html = "\n\n<div class='tool-result'>Tool Result:<pre><code>" + safe_result + "</code></pre></div>\n\n";
            stream_tool_result(result_html);
            consoleFlush();
            prev_stdout_ended_with_newline = true;  // Tool output printed, ends with \n

            chat_log << "\n";
            generated_text = ""; unprinted_text = "";
            // Reset tracking for new generation phase
            tool_start = string::npos; tool_end = string::npos;
            think_start = string::npos; think_end = string::npos;
            in_tool_call_stream = false; in_thinking_block = false;
            think_buffer.clear();
            think_buffering = true;

            string tool_result_section = string(TURN_START) + "user\n[Tool Result]\n" + sanitize(tool_result) + TURN_END + "\n";
            string tool_msg = tool_result_section;

            if (tool_blocked_by_loop || inject_auto_user_msg) {
                // Reconstruct a clean single user turn with error + intervention,
                // then open a fresh assistant turn. This forces the LLM to start from scratch
                // rather than continuing its previous trajectory (which caused the loop or
                // invalid tool calls). Equivalent to the user hitting Ctrl-C and typing a message.
                string clean_user_turn = string(TURN_START) + "user\n[Tool Result]\n" + sanitize(tool_result);
                if (inject_auto_user_msg && !active_intervention_msg.empty()) {
                    clean_user_turn += "\n" + active_intervention_msg;
                }
                clean_user_turn += string(TURN_END) + "\n";
                tool_msg = clean_user_turn + string(TURN_START) + "assistant\n";
            } else {
                tool_msg += string(TURN_START) + "assistant\n";
            }

            vector<llama_token> t_tokens = tokenize(tool_msg);
            if (n_past + (int)t_tokens.size() >= (int)cparams.n_ctx) {
                string ctx_diag = n_past != last_n_past ? " (n_past=" + std::to_string(n_past) + " + " + std::to_string(t_tokens.size()) + " tokens, last_n_past=" + std::to_string(last_n_past) + ", n_ctx=" + std::to_string(cparams.n_ctx) + ")" : "";
                diag("Context limit exhausted" + ctx_diag + ". Type 'clear' to reset.", "\033[31m");
                auto_continue = false;
            } else if (!feed_tokens(t_tokens)) {
                abort_auto = true;
            } else {
                auto_continue = true;
                continue; // Skip the standard logging at the bottom and go straight to the next token generation loop
            }
        } else {
            // Circuit breaker fired: feed the error back to the LLM so it knows
            // its tool call was blocked. Without this, the LLM sees silence and
            // generates the same call again -> infinite loop.
            auto_continue = false;
            generated_text = ""; unprinted_text = "";

            string abort_msg;
            if (invalid_tool_strikes >= 5) {
                abort_msg = "System Error: You are generating malformed tool calls. Your XML schema is incorrect. Stop and carefully review the required format. Do NOT wrap tool calls in markdown code blocks or other formatting.";
            } else {
                abort_msg = "System Error: Tool call blocked -- you are repeating yourself. Stop retrying and try a different approach (e.g., use search_file instead of exec_shell for code searches).";
            }
            string tool_result_section = string(TURN_START) + "user\n[Tool Result]\n" + abort_msg + TURN_END + "\n";
            string tool_msg = tool_result_section;  // Close the turn -- do not leave open assistant tag

            vector<llama_token> t_tokens = tokenize(tool_msg);
            if (n_past + t_tokens.size() < cparams.n_ctx) {
                feed_tokens(t_tokens);
            }
        }
    }

    if (!auto_continue && !generated_text.empty()) log_entry("ASSISTANT", generated_text);

    // --- REINCARNATE POST-GENERATION HANDLING ---
    // After reincarnate generation completes, verify userprompt is nonempty before proceeding.
    if (reincarnate_mode && !auto_continue) {
        reincarnate_mode = false;

        string userprompt_path = string(HOME) + "/userprompt";

        // Verify that userprompt exists and is nonempty.
        // We truncated it before generation, so a nonempty file means the LLM wrote a new prompt.
        ifstream userprompt_file(userprompt_path);
        if (!userprompt_file.is_open()) {
            diag("Reincarnate failed: LLM did not write " + userprompt_path + ". Session will not be reincarnated.", "\033[31m");
            log_entry("SYSTEM", "Reincarnate failed: userprompt was not written by LLM");
            continue;
        }

        // Check that userprompt is nonempty by reading lines until we find a character.
        bool found_content = false;
        string line;
        while (getline(userprompt_file, line)) {
            for (char c : line) {
                if (!isspace(c)) { found_content = true; break; }
            }
            if (found_content) break;
        }
        userprompt_file.close();

        if (!found_content) {
            diag("Reincarnate failed: " + userprompt_path + " is empty. Session will not be reincarnated.", "\033[31m");
            log_entry("SYSTEM", "Reincarnate failed: userprompt is empty");
            continue;
        }

        diag("Clearing context and starting reincarnated session...", "\033[35m");
        clear_context();
        // NOTE: Do NOT call clear_viewer() here. The explicit "clear" command clears the browser screen,
        // but reincarnate should preserve the visible conversation history so the user can see what happened.

        // Output a visual divider to the browser to mark the start of the new session
        if (should_output_to_browser()) {
            string divider =
                "\n\n<div style=\"text-align:center;margin:24px 0;\">\n"
                "  <hr style=\"border:none;border-top:2px dashed #555;width:80%;margin:0 auto;padding:0;\">\n"
                "  <span style=\"color:#aaa;font-size:13px;font-weight:bold;margin-top:6px;display:inline-block;\">-- New Session (Reincarnated) --</span>\n"
                "</div>\n\n";
            stream_html(divider);
        }

        string follow_prompt = "Follow the prompt in " + string(HOME) + "/userprompt";
        log_entry("USER", "[reincarnated session] " + follow_prompt);

        string new_session_message = string(TURN_START) + "user\n" + follow_prompt + TURN_END + "\n" + TURN_START + "assistant\n";
        vector<llama_token> new_session_tokens = tokenize(new_session_message);

        if (n_past + (int)new_session_tokens.size() >= (int)cparams.n_ctx) {
            string ctx_diag = n_past != last_n_past ? " (n_past=" + std::to_string(n_past) + " + " + std::to_string(new_session_tokens.size()) + ", last_n_past=" + std::to_string(last_n_past) + ")" : "";
            diag("Context Limit Reached! Cannot process reincarnated prompt" + ctx_diag + ".", "\033[31m");
            continue;
        }

        if (!feed_tokens(new_session_tokens)) {
            if (stop_generation) stop_generation = 0;
            diag("Failed to feed reincarnated session tokens. Type 'clear' to reset.", "\033[31m");
            continue;
        }

        auto_continue = true;
        reset_session_state();
        log_entry("SYSTEM", "Context Cleared and Reincarnated with New Prompt");
        continue;
    }
  }

  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
