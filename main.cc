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
#include <chrono>
#include <signal.h>
#include <cctype>
#include <set>
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


// --- Web Viewer Output Support ---
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

int pipe_fd = -1;
const char* FIFO_PATH = "/tmp/lllm.fifo";

// --- LLLM_OUTPUT Environment Variable Control ---
// Modes: 3=both stdout+browser, 2=browser only (no stdout), 1=stdout only (no browser), 0=no output (system messages still go to stdout)
static int get_output_mode() {
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

// Check if think blocks should be output to stdout (modes 1, 2, or 3)
bool should_output_think_blocks() {
    int mode = get_output_mode();
    return mode == 1 || mode == 2 || mode == 3;
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

void stream(const string& raw_token) {
    if (!should_output_to_browser()) return;
    // Thinking blocks are already filtered at source (see SAFE TERMINAL BUFFERING)
    // This function just sends pre-filtered text to browser

    // Filter out HTML closing tags that might leak from model output
    string filtered = raw_token;
    size_t pos = 0;
    while ((pos = filtered.find("</div>", pos)) != string::npos) {
        filtered.erase(pos, 6);
    }

    if (pipe_fd < 0) {
        pipe_fd = open(FIFO_PATH, O_RDWR | O_NONBLOCK);
    }
    if (pipe_fd >= 0) {
        ssize_t res = write(pipe_fd, filtered.c_str(), filtered.length());

        // If write fails with EAGAIN or EWOULDBLOCK, the pipe buffer is just full
        // (Python is reading too slowly). We just drop the token but keep the pipe open.
        // If it's a real structural error, we close it to try again later.
        if (res < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            close(pipe_fd);
            pipe_fd = -1;
        }
    }
}

void clear_viewer() {
    if (!should_output_to_browser()) return;
    if (pipe_fd < 0) {
        pipe_fd = open(FIFO_PATH, O_RDWR | O_NONBLOCK);
    }
    if (pipe_fd >= 0) {
        const char* cmd = "[[CLEAR]]";
        ssize_t res = write(pipe_fd, cmd, strlen(cmd));
        if (res < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            close(pipe_fd);
            pipe_fd = -1;
        }
    }
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

bool handle_llama_decode_error(llama_context *ctx, llama_batch batch, const char* error_msg = "KV Cache Exhausted. Type 'clear' to reset.", bool should_break = true) {
    int ret = llama_decode(ctx, batch);
    if (ret < -1) {
        message("\n\033[31m[" + std::string(error_msg) + "]\033[0m\n");
        cout.flush();
        return false;
    } else if (ret == -1) {
        message("\n\033[31m[Invalid input batch: " + std::string(error_msg) + "]\033[0m\n");
        cout.flush();
        return false;
    } else if (ret == 1 || ret == 2) {
        message("\n\033[31m[" + std::string(ret == 1 ? error_msg : "Aborted") + "]\033[0m\n");
        cout.flush();
        if (should_break) return false;
        return true;
    }
    return true;
}

// --- Global Interrupt Flag ---
volatile sig_atomic_t stop_generation = 0;

// --- SEQUENTIAL INTERVENTION MESSAGES ---
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
    return msg;
}

// --- Signal Handler for Task Interruption ---
void sigint_handler(int sig) {
  stop_generation = 1;
  message("\033[0m\n");
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
    size_t max_window_size;

    string normalize_str(const string& s) {
        string res;
        for (char c : s) { if (!isspace(c)) res += c; }
        return res;
    }

public:
    LoopDetector(size_t window_size = 15) : max_window_size(window_size) {}

    bool check_for_loop(const string& preamble, const string& tool_call) {
        string norm_tool = normalize_str(tool_call);
        size_t tool_hash = hash<string>{}(norm_tool);

        tool_history.push_back(tool_hash);
        if (tool_history.size() > max_window_size) {
            tool_history.pop_front();
        }

        int occurrence_count = 0;
        for (size_t past_hash : tool_history) {
            if (past_hash == tool_hash) occurrence_count++;
        }
        return (occurrence_count >= 3);
    }

    int get_loop_strikes() const {
        if (tool_history.empty()) return 0;
        size_t last = tool_history.back();
        int count = 0;
        for (size_t past_hash : tool_history) {
            if (past_hash == last) count++;
        }
        return count;
    }

    void clear_history() {
        tool_history.clear();
    }
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
    string text = extract_string_arg_bounded(tool_call, "text");
    replace_all_tags(text, PARAM_END_ESC, PARAM_END); // Unescape
    if (!path.empty()) {
      string current_req = path + ":" + text;
      if (current_req == last_grep_req) {
          result = "System Error: You just ran this exact search_file. Do not repeat the same search. Try a different text.";
      } else {
          last_grep_req = current_req;
          FileSystemTools fs;
          result = fs.search_file(path, text);
      }
    } else {
      result = "Error: path and text are required for search_file";
    }
  } else if (tool_name == "write_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    string content = extract_string_arg_bounded(tool_call, "content");
    content = remove_trailing_spaces(content);
    replace_all_tags(content, PARAM_END_ESC, PARAM_END); // Unescape
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
    string old_str = extract_string_arg_bounded(tool_call, "old");
    string new_str = extract_string_arg_bounded(tool_call, "new");
    old_str = remove_trailing_spaces(old_str);
    new_str = remove_trailing_spaces(new_str);
    replace_all_tags(old_str, PARAM_END_ESC, PARAM_END); // Unescape
    replace_all_tags(new_str, PARAM_END_ESC, PARAM_END); // Unescape
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

  HOME=getenv("HOME");

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
  signal(SIGINT, sigint_handler);

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
          for (const auto& tag : tags_to_remove) {
              size_t p;
              while ((p = clean_text.find(tag)) != string::npos) {
                  clean_text.erase(p, tag.length());
              }
          }
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
  cparams.n_batch   = 4096;
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
  const char* history_file = ".llm_history";
  load_history_safe(history_file);

  set<string> clean_files;
  string last_grep_req = "";
  LoopDetector loop_guard(15);
  int intra_loop_strikes = 0;

  // --- MAIN CHAT TURN LOOP ---
  while (true) {
    string user_input = "";
    stop_generation = 0;

    if (!auto_continue) {
      while (true) {
        const char* main_p = "\001\033[1;34m\002>>> \001\033[34m\002";
        const char* cont_p = "\001\033[1;34m\002... \001\033[34m\002";
        if (!first_prompt_displayed) first_prompt_displayed = true;

        if (user_input.empty()) console("\n");
        char* input_c = readline(user_input.empty() ? main_p : cont_p);

        console("\033[0m");
        consoleFlush();

        if (!input_c) {
          if (stop_generation) {
            stop_generation = 0;
            user_input = "";
            break;
          } else {
            if (user_input.empty()) user_input = "quit";
            break;
          }
        }

        string line(input_c);
        free(input_c);

        if (line == "quit" || line == "exit" || line == "clear" || line == "reset") {
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
        llama_free(ctx);
        ctx = llama_init_from_model(model, cparams);
        n_past = 0;
        batch.n_tokens = 0;
        for (size_t i = 0; i < (int)system_tokens.size(); i++) {
            common_batch_add(batch, system_tokens[i], n_past++, {0}, (i == (int)system_tokens.size() - 1));
            if (batch.n_tokens == (int)cparams.n_batch && i != (int)system_tokens.size() - 1) {
                if (!handle_llama_decode_error(ctx, batch)) break;
                batch.n_tokens = 0;
            }
        }
        if (batch.n_tokens > 0) handle_llama_decode_error(ctx, batch, "KV Cache Exhausted. Type 'clear' to reset.", false);

        auto_continue = false;
        clean_files.clear();
        last_grep_req = "";
        loop_guard.clear_history();
        intra_loop_strikes = 0;
        llama_sampler_reset(smpl);
        NetworkTools().reset_search();
        NetworkTools::reset_context_usage();
        log_entry("SYSTEM", "Context Cleared");
        clear_viewer();
        message("\n\033[32m[Context Cleared Successfully]\033[0m\n");
        cout.flush();
        continue;
    }

    if (user_input == "reset") {
        clean_files.clear();
        last_grep_req = "";
        loop_guard.clear_history();
        intra_loop_strikes = 0;
        llama_sampler_reset(smpl);
        NetworkTools().reset_search();
        log_entry("SYSTEM", "Loop Counter and File Cache Reset");
        message("\n\033[32m[Loop Counter and File Cache Reset Successfully]\033[0m\n");
        cout.flush();
        continue;
    }

    if (user_input.empty() && !auto_continue) continue;

    if (!user_input.empty()) {
      if (!auto_continue) {
          log_entry("USER", user_input);
          // Write user input HTML directly to pipe (bypass stream() which strips </div>)
          if (should_output_to_browser()) {
              if (pipe_fd < 0) pipe_fd = open(FIFO_PATH, O_RDWR | O_NONBLOCK);
              if (pipe_fd >= 0) {
                  string user_html = "\n\n<div style=\"color: #007bff;\">\n\n```" + user_input + "\n```\n\n</div>\n\n";
                  write(pipe_fd, user_html.c_str(), user_html.length());
              }
          }
      }

      string user_message;

      if (use_dummy_thought) {
          // Dummy Thought Injection - disable thinking mode
          user_message = string(TURN_START) + "user\n" +
                           user_input + TURN_END + "\n" +
                           TURN_START + "assistant\n"+THINK_START+"\nThe user wants a direct answer. I will output the requested data immediately without preamble.\n"+THINK_END+"\n";
      } else {
          // Normal thinking mode
          user_message = string(TURN_START) + "user\n" + user_input + TURN_END + "\n" + TURN_START + "assistant\n";
      }
      vector<llama_token> tokens = tokenize(user_message);

      if (n_past + tokens.size() >= cparams.n_ctx) {
          message("\n\033[31m[Context Limit Reached! Cannot process input. Type 'clear' to reset.]\033[0m\n");
          continue;
      }

      batch.n_tokens = 0;
      bool input_interrupted = false;

      for (size_t i = 0; i < (int)tokens.size(); i++) {
        if (stop_generation) {
          message("\n\033[31m[Input Evaluation Interrupted]\033[0m\n");
          cout.flush();
          stop_generation = 0;
          input_interrupted = true;
          break;
        }

        common_batch_add(batch, tokens[i], n_past++, {0}, (i == (int)tokens.size() - 1));

        if (batch.n_tokens == (int)cparams.n_batch && i != (int)tokens.size() - 1) {
          if (!handle_llama_decode_error(ctx, batch)) {
            stop_generation = 0;
            input_interrupted = true;
            break;
          }
          batch.n_tokens = 0;
        }
      }
      if (batch.n_tokens > 0 && !input_interrupted) {
        if (!handle_llama_decode_error(ctx, batch, "KV Cache Exhausted. Type 'clear' to reset.", false)) {
          stop_generation = 0;
          input_interrupted = true;
        }
      }
      if (input_interrupted) continue;
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

    // Track thinking blocks (similar to tool calls)
    bool in_thinking_block = false;
    size_t think_start = string::npos;
    size_t think_end = string::npos;

    // Track if stdout ended with newline from previous iteration's tool output
    static bool prev_stdout_ended_with_newline = false;

    // --- INNER TOKEN GENERATION LOOP ---
    while (true) {
      if (stop_generation) {
        message("\n\033[31m[Task Interrupted by User]\033[0m\n");
        cout.flush();
        stream("\n\n*[Task Interrupted by User]*\n\n");
        stop_generation = 0;
        auto_continue = false;
        break;
      }

      if (n_past >= (int)cparams.n_ctx - 10) {
        message("\n\033[31m[Context Window Exhausted! Type 'clear' to reset.]\033[0m\n");
        if (!unprinted_text.empty()) {
            console(unprinted_text.c_str());
            consoleFlush();
            stream(unprinted_text);
        }
        auto_continue = false;
        break;
      }

      llama_token next_token = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);

      // --- EARLY EOG RECOVERY ---
      if (llama_vocab_is_eog(vocab, next_token)) {
          size_t active_ts = generated_text.rfind(FUNC_START);
          size_t active_te = generated_text.rfind(FUNC_END);

          if (active_ts != string::npos && (active_te == string::npos || active_ts > active_te)) {
              message("\033[33m[System: Premature End-Of-Turn detected. Auto-recovering tags...]\033[0m\n");
              cout.flush();

              size_t trailing_slash = generated_text.rfind("</");
              if (trailing_slash != string::npos && trailing_slash > active_ts) {
                  size_t drop_len = generated_text.length() - trailing_slash;
                  generated_text.erase(trailing_slash);
                  full_response.erase(full_response.length() - drop_len);
              }

              string forced_close = "\n" + string(PARAM_END) + "\n" + string(FUNC_END) + "\n";
              generated_text += forced_close;
              full_response += forced_close;

              tool_start = active_ts;
              tool_end = generated_text.find(FUNC_END, active_ts);
              trigger_tool_execution = true;
              break;
          }

          auto_continue = false;
          break;
      }

      string token_str = common_token_to_piece(ctx, next_token).c_str();
      generated_text += token_str;
      full_response += token_str;

      // --- PERF OPTIMIZATION: O(1) TRACKING OFFSETS ---
      if (tool_start == string::npos) {
          tool_start = generated_text.find(FUNC_START, func_search_pos);
          if (tool_start == string::npos) {
              func_search_pos = generated_text.length() > 20 ? generated_text.length() - 20 : 0;
          }
      }
      if (tool_start != string::npos && tool_end == string::npos) {
          tool_end = generated_text.find(FUNC_END, tool_start);
      }

      in_tool_call_stream = (tool_start != string::npos && tool_end == string::npos);

      // --- THINKING BLOCK TRACKING ---
      if (think_start == string::npos) {
          think_start = generated_text.find(Tokens::THINK_START);
      }
      if (think_start != string::npos && think_end == string::npos) {
          think_end = generated_text.find(Tokens::THINK_END, think_start);
      }
      in_thinking_block = (think_start != string::npos && think_end == string::npos);

      // --- TARGETED SYNTAX TRAP (Stutter Fix) ---
      if (in_tool_call_stream && generated_text.length() >= 4 &&
          generated_text.compare(generated_text.length() - 4, 4, DOUBLE_OPEN) == 0) {

          message("\n\033[33m[System: Infinite slash loop detected. Auto-recovering...]\033[0m\n");
          cout.flush();

          size_t bad_pos = generated_text.rfind(DOUBLE_OPEN);
          if (bad_pos != string::npos && bad_pos > tool_start) {
              size_t drop_len = generated_text.length() - bad_pos;
              generated_text.erase(bad_pos);
              full_response.erase(full_response.length() - drop_len);
          }

          string forced_close = "\n" + string(PARAM_END) + "\n" + string(FUNC_END) + "\n";
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

      // --- INTRA-TURN LOOP DETECTION (Babbling Prevention) ---
      if (t_count > 0 && t_count % 10 == 0 && generated_text.length() > 100) {
          size_t n = generated_text.length();
          bool intra_loop = false;
          size_t max_len = min((size_t)3000, n / 2);
          char last_char = generated_text.back();
          for (size_t len = 50; len <= max_len; len += 10) {
              if (generated_text[n - len - 1] == last_char) {
                  if (generated_text.compare(n - len, len, generated_text, n - 2 * len, len) == 0) {
                      size_t spaces = 0;
                      for(size_t i = n - len; i < n; ++i) { if(generated_text[i] == ' ') spaces++; }
                      if (spaces > 5) {
                          intra_loop = true;
                          break;
                      }
                  }
              }
          }

          if (!intra_loop && n > 300 && !in_tool_call_stream) {
              size_t suffix_len = 250;
              string suffix = generated_text.substr(n - suffix_len);
              size_t prev_pos = generated_text.rfind(suffix, n - suffix_len - 1);
              if (prev_pos != string::npos) intra_loop = true;
          }

          if (intra_loop) {
              intra_loop_strikes++;
              if (intra_loop_strikes >= 5) {
                  message("\n\033[1;31m[System: Agent stubbornly babbling. Ejecting to manual prompt.]\033[0m\n");
                  cout.flush();
                  auto_continue = false;
                  break;
              }

              message("\n\033[35m[System: Intra-turn Generation Loop Detected. Injecting intervention.]\033[0m\n");
              cout.flush();

              if (!in_tool_call_stream && !unprinted_text.empty()) {
                  console(unprinted_text.c_str());
                  consoleFlush();
                  stream(unprinted_text);
              }
              unprinted_text = "";
              log_entry("ASSISTANT (Interrupted Reasoning Loop)", generated_text);

              string active_intervention_msg = get_next_loop_message();
              string msg = string(TURN_END) + "\n" + TURN_START + "user\n" + active_intervention_msg + "\n" + TURN_END + "\n" + TURN_START + "assistant\n";
              vector<llama_token> t_tokens = tokenize(msg);

              if (n_past + t_tokens.size() >= cparams.n_ctx) {
                  message("\n\033[31m[Context limit exhausted. Type 'clear' to reset.]\033[0m\n");
                  cout.flush();
                  auto_continue = false;
                  break;
              }

              batch.n_tokens = 0;
              for (size_t i = 0; i < t_tokens.size(); i++) {
                  common_batch_add(batch, t_tokens[i], n_past++, {0}, (i == t_tokens.size() - 1));
                  if (batch.n_tokens == (int)cparams.n_batch && i != t_tokens.size() - 1) {
                      if (!handle_llama_decode_error(ctx, batch)) { auto_continue = false; break; }
                      batch.n_tokens = 0;
                  }
              }
              if (batch.n_tokens > 0) {
                if (!handle_llama_decode_error(ctx, batch)) { auto_continue = false; break; }
              }

              auto_continue = true;
              break;
          }
      }

      // --- SAFE TERMINAL BUFFERING ---
      if (!in_tool_call_stream && !in_thinking_block) {
          size_t safe_len = generated_text.length();
          string fstart(FUNC_START);
          string tstart(Tokens::THINK_START);

          // CRITICAL: If we just exited a thinking block (think_end found), skip past THINK_END
          // to prevent it from leaking into normal text output
          if (think_start != string::npos && think_end != string::npos) {
              size_t think_block_end = think_end + string(Tokens::THINK_END).length();
              if (print_pos < think_block_end) {
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

              // In browser-only mode (LLLM_OUTPUT=2), strip both <think> and </think> tags
              // In combined mode (LLLM_OUTPUT=3), keep the tags visible
              int mode = get_output_mode();
              if (mode == 2) {
                  // Strip opening tag from output
                  size_t open_tag_pos = think_output.find(tstart);
                  if (open_tag_pos != string::npos) {
                      think_output.erase(open_tag_pos, tstart.length());
                  }
                  // Strip closing tag from output
                  size_t close_tag_pos = think_output.find(tend);
                  if (close_tag_pos != string::npos) {
                      think_output.erase(close_tag_pos, tend.length());
                  }
              }

              console_think(think_output.c_str());
              consoleThinkFlush();
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
      if (!handle_llama_decode_error(ctx, batch)) break;

    } // END INNER TOKEN LOOP

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

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

    if (t_count > 0) {
        cout << "\033[34m[Speed: " << fixed << setprecision(2) << (t_count / elapsed) << " t/s | Elapsed: " << elapsed << "s]\033[0m" << endl;
    }

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

        const vector<string> strip_tags = {TURN_START, TURN_END};
        for (const auto& tag : strip_tags) {
            size_t p;
            while ((p = tool_call.find(tag)) != string::npos) tool_call.erase(p, tag.length());
        }

        string tool_name_for_display = "";
        size_t name_start = tool_call.find(FUNC_START);
        if (name_start != string::npos) {
            name_start += string(FUNC_START).length();
            size_t name_end = tool_call.find('>', name_start);
            if (name_end != string::npos) {
                tool_name_for_display = tool_call.substr(name_start, name_end - name_start);
            }
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

        int bticks = 0; size_t pos = 0;
        while ((pos = full_response.find("```", pos)) != string::npos) { bticks++; pos += 3; }
        if (bticks % 2 != 0) is_real_tool = false;

        size_t global_ts = full_response.rfind(FUNC_START);
        if (global_ts != string::npos && global_ts > 0) {
          char prev_char = full_response[global_ts - 1];
          if (prev_char == '`' || prev_char == '\\') is_real_tool = false;
        }

        string tool_result = "";
        string display_result = "";
        bool was_loop = false;
        bool abort_auto = false;
        bool inject_auto_user_msg = false;
        string active_intervention_msg = "";

        string tool_name = tool_name_for_display;

        if (!is_real_tool) {
          tool_result = "System Error: Invalid tool format or unsupported tool. You MUST use the strict XML schema. Supported tools: read_files, write_file, edit_file, exec_shell, search_file, web_search. Please try again.";
          display_result = tool_result;
        } else {
          bool is_mutating_tool = (tool_name == "edit_file" || tool_name == "write_file");

          was_loop = loop_guard.check_for_loop(preamble, tool_call);
          int current_strikes = loop_guard.get_loop_strikes();

          tool_result = execute_tool_call(tool_call, clean_files, last_grep_req);

          // Check for interrupt after tool execution completes
          if (stop_generation) {
            message("\n\033[31m[Tool Interrupted by User]\033[0m\n");
            cout.flush();
            stop_generation = 0;
            abort_auto = true;
          }

          if (!abort_auto) {
              bool tool_failed = (tool_result.find("System Error:") != string::npos || tool_result.find("Error:") != string::npos);

              if (was_loop) {
                  active_intervention_msg = get_next_loop_message();
                  tool_result = active_intervention_msg;

                  display_result = tool_result;

                  int max_attempts = loopMessages.size();
                  int attempt_num = current_strikes - 2;

                  if (attempt_num <= max_attempts) {
                      message("\n\033[35m[System: Loop Detected. Automating intervention (Attempt " + std::to_string(attempt_num) + "/" + std::to_string(max_attempts) + ").]\033[0m\n");
                      cout.flush();
                      abort_auto = false;
                      inject_auto_user_msg = true;
                  } else {
                      message("\n\033[1;31m[System: Intervention failed after " + std::to_string(max_attempts) + " attempts. Agent is stuck. Ejecting to prompt.]\033[0m\n");
                      cout.flush();
                      abort_auto = true;
                      intra_loop_strikes = 0;
                  }
              } else {
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
              // Truncate display_result to first 500 chars for debug output
              string truncated_display = display_result;
              if (truncated_display.length() > 500) {
                  truncated_display = truncated_display.substr(0, 497) + "...";
              }

              console("\n\033[92m[Tool Result]\033[0m\n");
              string result_to_print = truncated_display;
              size_t p = 0;
              while ((p = result_to_print.find('\n')) != string::npos) {
                console("  ", (int)p, result_to_print.c_str(),"\n");
                  result_to_print.erase(0, p + 1);
              }
              if (!result_to_print.empty()) console("  ", result_to_print.c_str(),"\n");
              stream("\n\n> **Tool Result:**\n> ```\n> " + truncated_display + "```\n\n");
            }
            consoleFlush();
            prev_stdout_ended_with_newline = true;  // Tool output printed, ends with \n

            chat_log << "\n";
            generated_text = ""; unprinted_text = "";
            // Reset tracking for new generation phase
            tool_start = string::npos; tool_end = string::npos;
            think_start = string::npos; think_end = string::npos;
            in_tool_call_stream = false; in_thinking_block = false;

            string tool_result_section = string(TURN_START) + "user\n[Tool Result]\n" + sanitize(tool_result) + TURN_END + "\n";
            string tool_msg = tool_result_section + TURN_START + "assistant\n";

            if (inject_auto_user_msg) {
                tool_msg += active_intervention_msg + string(TURN_END) + "\n" + TURN_START + "assistant\n";
            }

            vector<llama_token> t_tokens = tokenize(tool_msg);
            if (n_past + t_tokens.size() >= cparams.n_ctx) {
                message("\n\033[31m[Context limit exhausted. Type 'clear' to reset.]\033[0m\n");
                cout.flush();
                auto_continue = false;
            } else {
                batch.n_tokens = 0;
                for (size_t i = 0; i < t_tokens.size(); i++) {
                    common_batch_add(batch, t_tokens[i], n_past++, {0}, (i == t_tokens.size() - 1));
                    if (batch.n_tokens == (int)cparams.n_batch && i != t_tokens.size() - 1) {
                        if (!handle_llama_decode_error(ctx, batch)) {
                            abort_auto = true;
                            break;
                        }
                        batch.n_tokens = 0;
                    }
                }
                if (!abort_auto && batch.n_tokens > 0) {
                    if (!handle_llama_decode_error(ctx, batch)) {
                        abort_auto = true;
                    }
                }

                if (!abort_auto) {
                    auto_continue = true;
                    continue; // Skip the standard logging at the bottom and go straight to the next token generation loop
                }
            }
        } else {
            auto_continue = false;
        }
    }

    if (!auto_continue && !generated_text.empty()) log_entry("ASSISTANT", generated_text);
  }

  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
