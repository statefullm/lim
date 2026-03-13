#include "llama.h"
#include "common.h"
#include "parsers.h"
#include "filesystem.h"
#include "network.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
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

// --- Readline Headers ---
#include <readline/readline.h>
#include <readline/history.h>

std::string tokenStart = "<|im_start|>";
std::string tokenEnd = "<|im_end|>";
std::string toolStart = "<tool_call>";
std::string toolEnd = "</tool_call>";

bool handle_llama_decode_error(llama_context *ctx, llama_batch batch, const char* error_msg = "KV Cache Exhausted. Type 'clear' to reset.", bool should_break = true) {
    int ret = llama_decode(ctx, batch);
    if (ret < -1) {
        // Fatal error - should break
        printf("\n\033[31m[%s]\033[0m\n", error_msg);
        fflush(stdout);
        return false;
    } else if (ret == -1) {
        // Invalid input batch - should break
        printf("\n\033[31m[Invalid input batch: %s]\033[0m\n", error_msg);
        fflush(stdout);
        return false;
    } else if (ret == 1) {
        // KV cache exhausted - always inform the user, but only break if should_break is true
        printf("\n\033[31m[%s]\033[0m\n", error_msg);
        fflush(stdout);
        if (should_break) {
            return false;  // break
        }
        return true;  // continue, but user was informed
    } else if (ret == 2) {
        // Aborted - always inform the user, but only break if should_break is true
        printf("\n\033[31m[Aborted: %s]\033[0m\n", error_msg);
        fflush(stdout);
        if (should_break) {
            return false;  // break
        }
        return true;  // continue, but user was informed
    }
    // ret == 0, success
    return true;
}

// --- Global Interrupt Flag ---
volatile sig_atomic_t stop_generation = 0;

using namespace std;

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
  printf("\033[0m\n");
  fflush(stdout);
}

// --- Dummy Log Callback to Silence Llama.cpp ---
bool first_prompt_displayed = false;

// Global debug flag - accessible across all modules
bool is_debug = false;

// Chat log file stream - accessible across all modules for tool diagnostics
std::ofstream chat_log;

void dummy_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
  // Suppress llama.cpp terminal output
}

bool should_suppress_logs() {
    // Suppress logs after first prompt is displayed, even in debug mode
    return first_prompt_displayed;
}

// Custom log callback that ensures all output is suppressed after first prompt
void custom_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    // Always suppress all ggml diagnostics after first prompt is displayed
    if (first_prompt_displayed) {
        return; // Suppress all output
    }
    // Before first prompt, allow output
    fprintf(stderr, "%s", text);
}

// --- Safe Multiline History Handlers ---
void load_history_safe(const char* filename) {
    ifstream in(filename);
    string line;
    while (getline(in, line)) {
        for (char& c : line) {
            if (c == '\x1E') c = '\n';
        }
        add_history(line.c_str());
    }
}

void save_history_safe(const char* filename, const string& input) {
    ofstream out(filename, ios::app);
    string enc = input;
    for (char& c : enc) {
        if (c == '\n') c = '\x1E';
    }
    out << enc << "\n";
}

// --- Macro-Loop Detection ---
class LoopDetector {
private:
    deque<size_t> tool_history;
    size_t max_window_size;

    string normalize_str(const string& s) {
        string res;
        for (char c : s) {
            if (!isspace(c)) res += c; // Ignore spacing differences
        }
        return res;
    }

public:
    // Expanded window size to 15 to catch wide multi-step loops
    LoopDetector(size_t window_size = 15) : max_window_size(window_size) {}

    bool check_for_loop(const string& preamble, const string& tool_call) {
        // We drop preamble hashing entirely. Behavior (tool calls) is what matters.
        string norm_tool = normalize_str(tool_call);
        size_t tool_hash = hash<string>{}(norm_tool);

        // 1. Add to sliding window
        tool_history.push_back(tool_hash);
        if (tool_history.size() > max_window_size) {
            tool_history.pop_front();
        }

        // 2. Count occurrences of this exact tool call in the recent window
        int occurrence_count = 0;
        for (size_t past_hash : tool_history) {
            if (past_hash == tool_hash) occurrence_count++;
        }

        // Trigger intervention if this exact action has occurred 3 times recently
        return (occurrence_count >= 3);
    }

    int get_loop_strikes() const {
        if (tool_history.empty()) return 0;
        size_t last = tool_history.back();
        int count = 0;
        for (size_t past_hash : tool_history) {
            if (past_hash == last) count++;
        }
        return count; // Feeds seamlessly into your existing attempt_num logic
    }

    void clear_history() {
        tool_history.clear();
    }
};

// --- Tool Execution Logic ---

string execute_tool_call(const string& tool_call, set<string>& clean_files, string& last_grep_req) {
  string result = "";

  string tool_name = tool_call.substr(0, tool_call.find('('));

  if (tool_name == "read_files") {
    vector<string> paths = extract_array_arg_bounded(tool_call, "paths");
    if (!paths.empty()) {
      FileSystemTools fs;
      result = "Files content:\n";
      for (const auto& path : paths) {
        if (clean_files.count(path)) {
          result += "Path: " + path + "\n";
          result += "Content:\n[Content omitted: You already read this file and it has not been modified since your last read. If you need to search for specific code sections or variables, use the search_file tool instead. DO NOT call read_files on this file again.]\n";
          result += "---\n";
        } else {
          auto results = fs.read_files({path});
          for (const auto& file : results) {
            result += "Path: " + file.at("path") + "\n";
            result += "Content:\n" + file.at("content") + "\n";
            if (!file.at("error").empty()) result += "Error: " + file.at("error") + "\n";
            result += "---\n";

            if (file.at("error").empty()) {
                clean_files.insert(file.at("path"));
            }
          }
        }
      }
    } else {
      result = "Error: No paths provided to read_files";
    }
  } else if (tool_name == "search_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    string text = extract_string_arg_bounded(tool_call, "text");
    // Removed duplicate text.empty() check - let filesystem.cc handle validation
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
    // Apply remove_trailing_spaces to both strings for consistency
    old_str = remove_trailing_spaces(old_str);
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
  } else if (tool_name == "chmod") {
    string path = extract_string_arg_bounded(tool_call, "path");
    int mode = extract_int_arg_bounded(tool_call, "mode");
    clean_files.erase(path);
    last_grep_req = "";
    if (!path.empty()) {
      FileSystemTools fs;
      auto result_map = fs.chmod_file(path, mode);
      result = "Status: " + result_map.at("status");
      if (result_map.find("error") != result_map.end()) result += ", Error: " + result_map.at("error");
    } else {
      result = "Error: No path provided to chmod";
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
    // Insert a backslash to break the LLM's pattern matching for these
    // specific control tokens.
    vector<string> patterns = {
      toolStart,
      toolEnd,
      tokenStart,
      tokenEnd
    };

    for (const auto& pattern : patterns) {
      size_t pos = 0;
      while ((pos = text.find(pattern, pos)) != std::string::npos) {
            // Insert backslash after first character of the match
            text.insert(pos + 1, "\\");
            pos += pattern.length() + 1;
        }
    }

    return text;
}

int main(int argc, char ** argv) {
  setlocale(LC_ALL, "");

  // --- CHECK IF RUNNING AS USER 'ai' ---
  uid_t uid = getuid();
  struct passwd *pw = getpwuid(uid);
  if (pw == nullptr || strcmp(pw->pw_name, "ai") != 0) {
    fprintf(stderr, "Error: This program must be run as user 'ai'\n");
    return 1;
  }

  // --- WRITE CURRENT DIRECTORY TO .CWD FILE ON STARTUP ---
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != nullptr) {
    ofstream cwd_file("/home/ai/.cwd");
    if (cwd_file.is_open()) {
      cwd_file << cwd << endl;
      cwd_file.close();
    }
  }

  // --- SET UMASK FOR GROUP WRITABILITY ---
  umask(0002);

  // --- AUTOMATIC PROCESS CLEANUP ---
  std::atexit(NetworkTools::cleanup_searxng);

  signal(SIGINT, sigint_handler);

  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
    return 1;
  }

// Set up log callback that suppresses GGML diagnostics after first prompt
const char* debug_env = getenv("LLM_DEBUG");
if (debug_env != nullptr && (strcmp(debug_env, "1") == 0 || strcasecmp(debug_env, "true") == 0)) {
  is_debug = true;
}

if (!is_debug) {
  llama_log_set(dummy_log_callback, nullptr);
} else {
  // In debug mode, use custom callback that suppresses all output after first prompt
  llama_log_set(custom_log_callback, nullptr);
}

  // --- Sequential Logger Initialization ---
  // Create log subdirectory if it doesn't exist
  mkdir("log", 0755);

  int log_index = 1;
  string log_file_name;
  while (true) {
      log_file_name = "log/" + to_string(log_index);
      ifstream check_file(log_file_name.c_str());
      if (!check_file.good()) {
          break;
      }
      log_index++;
  }
  chat_log.open(log_file_name, ios::app);

  // --- Sanitized Logging Lambda ---
  auto log_entry = [&](const string& role, const string& text) {
      if (chat_log.is_open()) {
          string clean_text = text;
          const vector<string> tags_to_remove = {
              toolStart, toolEnd,
              tokenStart, tokenEnd
          };

          for (const auto& tag : tags_to_remove) {
              size_t p;
              while ((p = clean_text.find(tag)) != string::npos) {
                  clean_text.erase(p, tag.length());
              }
          }

          while (!clean_text.empty() && isspace(clean_text.back())) {
              clean_text.pop_back();
          }

          chat_log << "=== " << role << " ===\n" << clean_text << "\n\n";
          chat_log.flush();
      }
  };

  log_entry("SYSTEM", "Starting LLM Controller Session");

  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);

  auto mparams = llama_model_default_params();
  mparams.n_gpu_layers = 999;
  mparams.use_mmap = false;
  mparams.use_mlock = true;

  llama_model * model = llama_model_load_from_file(argv[1], mparams);
  if (!model) return 1;

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
  HOME=getenv("HOME");

  ifstream prompt_file(HOME+"/prompt");
  if (prompt_file.is_open()) {
    stringstream buffer;
    buffer << prompt_file.rdbuf();
    system_prompt = buffer.str();
    prompt_file.close();
  }

  const llama_vocab * vocab = llama_model_get_vocab(model);
  auto tokenize = [&](string text) { return common_tokenize(ctx, text, false, true); };

  vector<llama_token> system_tokens = common_tokenize(ctx, tokenStart + "system\n" + system_prompt + tokenEnd + "\n", true, true);

  // Sampling parameters: instruct mode for general tasks
  float temp = 0.7f;
  float top_p = 0.8f;
  int32_t top_k = 20;
  float min_p = 0.0f;
  float penalty_present = 1.5f;
  float penalty_repeat = 1.0f;

  /*
  // Sampling parameters: instruct mode for reasoning
  float temp = 1.0f;
  float top_p = 1.0f;
  int32_t top_k = 40;
  float min_p = 0.0f;
  float penalty_present = 2.0f;
  float penalty_repeat = 1.0f;
  */

  // Initialize sampler chain with the specified parameters
  llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
  llama_sampler * smpl = llama_sampler_chain_init(lparams);

  // Add samplers to the chain in order
  // Penalties sampler for repeat penalty
   llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, penalty_repeat, 0.0f, penalty_present));

  // Top-K sampling
  llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
  // Top-P (nucleus) sampling
  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
  // Min-P sampling
  llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
  // Temperature sampling
  llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
  // Final sampler - distribution sampling
  llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
  llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
  int n_past = 0;

  batch.n_tokens = 0;
  for (size_t i = 0; i < (int)system_tokens.size(); i++)
    common_batch_add(batch, system_tokens[i], n_past++, {0}, (i == (int)system_tokens.size() - 1));

  if (!handle_llama_decode_error(ctx, batch)) {
    return 1;
  }

  bool auto_continue = false;
  const char* history_file = ".llm_history";

  load_history_safe(history_file);

  // --- STATE TRACKING ---
  set<string> clean_files;
  string last_grep_req = "";
  LoopDetector loop_guard(15);
  int intra_loop_strikes = 0;

  while (true) {
    string user_input = "";
    stop_generation = 0;

    if (!auto_continue) {
      while (true) {
        const char* main_p = "\001\033[1;34m\002>>> \001\033[34m\002";
        const char* cont_p = "\001\033[1;34m\002... \001\033[34m\002";

        // Set flag when first prompt is displayed
        if (!first_prompt_displayed) {
          first_prompt_displayed = true;
        }

        if (user_input.empty()) printf("\n");
        char* input_c = readline(user_input.empty() ? main_p : cont_p);

        printf("\033[0m");
        fflush(stdout);

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

        if (line == "quit" || line == "exit") {
          user_input = line;
          break;
        }

        if (line == "clear") {
          user_input = "clear";
          break;
        }

        if (line == "reset") {
          user_input = "reset";
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
    } else {
// Use consolidated logging for agent processing message
if (is_debug) {
      log_diagnostic("Agent processing tool results...");
    }
    }

    if (user_input == "quit" || user_input == "exit") break;

    // --- Context Reset Command ---
    if (user_input == "clear") {
        llama_free(ctx);
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "\nError: Failed to re-initialize context during clear.\n");
            break;
        }
        n_past = 0;

        batch.n_tokens = 0;
        for (size_t i = 0; i < (int)system_tokens.size(); i++) {
            common_batch_add(batch, system_tokens[i], n_past++, {0}, (i == (int)system_tokens.size() - 1));
            if (batch.n_tokens == (int)cparams.n_batch && i != (int)system_tokens.size() - 1) {
                if (!handle_llama_decode_error(ctx, batch)) {
                    break;
                }
                batch.n_tokens = 0;
            }
        }
        if (batch.n_tokens > 0) {
            if (!handle_llama_decode_error(ctx, batch, "KV Cache Exhausted. Type 'clear' to reset.", false)) {
                // KV cache exhausted, but we should not break here
                // Just continue to the next iteration
            }
        }

        auto_continue = false;
        clean_files.clear();
        last_grep_req = "";
        loop_guard.clear_history();
        intra_loop_strikes = 0;
        log_entry("SYSTEM", "Context Cleared");
        printf("\033[32m[Context Cleared Successfully]\033[0m\n");
        continue;
    }

    // --- Reset Command (only resets loop counter and file cache, not context) ---
    if (user_input == "reset") {
        clean_files.clear();
        last_grep_req = "";
        loop_guard.clear_history();
        intra_loop_strikes = 0;
        log_entry("SYSTEM", "Loop Counter and File Cache Reset");
        printf("\033[32m[Loop Counter and File Cache Reset Successfully]\033[0m\n");
        continue;
    }

    if (user_input.empty() && !auto_continue) continue;

    if (!user_input.empty()) {
      if (!auto_continue) log_entry("USER", user_input);

      vector<llama_token> tokens = tokenize(tokenStart + "user\n" + user_input + tokenEnd + "\n" + tokenStart + "assistant\n");

      if (n_past + tokens.size() >= cparams.n_ctx) {
          printf("\n\033[31m[Context Limit Reached! Cannot process input. Type 'clear' to reset.]\033[0m\n");
          continue;
      }

      batch.n_tokens = 0;
      bool input_interrupted = false;

      for (size_t i = 0; i < (int)tokens.size(); i++) {
        if (stop_generation) {
          printf("\n\033[31m[Input Evaluation Interrupted]\033[0m\n");
          fflush(stdout);
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

    while (true) {
      if (stop_generation) {
        printf("\n\033[31m[Task Interrupted by User]\033[0m\n");
        fflush(stdout);
        stop_generation = 0;
        auto_continue = false;
        break;
      }

      if (n_past >= (int)cparams.n_ctx - 10) {
        printf("\n\033[31m[Context Window Exhausted! Type 'clear' to reset.]\033[0m\n");
        if (!unprinted_text.empty()) { printf("%s", unprinted_text.c_str()); fflush(stdout); }
        auto_continue = false;
        break;
      }

      llama_token next_token = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);
      string token_str = common_token_to_piece(ctx, next_token).c_str();
      generated_text += token_str;
      unprinted_text += token_str;
      full_response += token_str;

      // --- INTRA-TURN LOOP DETECTION ---
      if (t_count > 0 && t_count % 10 == 0 && generated_text.length() > 100) {
          size_t n = generated_text.length();
          bool intra_loop = false;

          size_t max_len = min((size_t)3000, n / 2);
          char last_char = generated_text.back();
          for (size_t len = 50; len <= max_len; ++len) {
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

          bool in_tool_call_stream = false;
          size_t t_start = generated_text.rfind(toolStart);
          size_t t_end = generated_text.rfind(toolEnd);
          if (t_start != string::npos && (t_end == string::npos || t_start > t_end)) {
              in_tool_call_stream = true;
          }

          bool in_reasoning_block = false;
          size_t think_start = generated_text.rfind(tokenStart);
          size_t think_end = generated_text.rfind(tokenEnd);
          if (think_start != string::npos && (think_end == string::npos || think_start > think_end)) {
              in_reasoning_block = true;
          }

          // Only do suffix-based loop detection when NOT in reasoning or tool blocks
          if (!intra_loop && n > 300 && !in_tool_call_stream && !in_reasoning_block) {
              size_t suffix_len = 250;
              string suffix = generated_text.substr(n - suffix_len);
              size_t prev_pos = generated_text.rfind(suffix, n - suffix_len - 1);
              if (prev_pos != string::npos) {
                  intra_loop = true;
              }
          }

          if (intra_loop) {
              intra_loop_strikes++;

              if (intra_loop_strikes >= 5) {
                  printf("\n\033[1;31m[System: Agent stubbornly babbling. Ejecting to manual prompt.]\033[0m\n");
                  fflush(stdout);
                  auto_continue = false;
                  break;
              }

              printf("\n\033[35m[System: Intra-turn Generation Loop Detected. Injecting intervention.]\033[0m\n");
              fflush(stdout);

              bool in_tool = false;
              size_t ts = generated_text.find(toolStart);
              if (ts != string::npos && generated_text.find(toolEnd, ts) == string::npos) {
                  in_tool = true;
              }

              if (!unprinted_text.empty() && !in_tool) {
                  printf("%s", unprinted_text.c_str());
                  fflush(stdout);
              }
              unprinted_text = "";

              log_entry("ASSISTANT (Interrupted Reasoning Loop)", generated_text);

              string active_intervention_msg = get_next_loop_message();
              string msg = tokenEnd + "\n" + tokenStart + "user\n" + active_intervention_msg + "\n" + tokenEnd + "\n" + tokenStart + "assistant\n";
              vector<llama_token> t_tokens = tokenize(msg);

              if (n_past + t_tokens.size() >= cparams.n_ctx) {
                  printf("\n\033[31m[Context limit exhausted. Type 'clear' to reset.]\033[0m\n");
                  auto_continue = false;
                  break;
              }

              batch.n_tokens = 0;
              for (size_t i = 0; i < t_tokens.size(); i++) {
                  common_batch_add(batch, t_tokens[i], n_past++, {0}, (i == t_tokens.size() - 1));
                  if (batch.n_tokens == (int)cparams.n_batch && i != t_tokens.size() - 1) {
                      if (!handle_llama_decode_error(ctx, batch)) {
                        auto_continue = false;
                        break;
                      }
                      batch.n_tokens = 0;
                  }
              }
              if (batch.n_tokens > 0) {
                if (!handle_llama_decode_error(ctx, batch)) {
                  auto_continue = false;
                  break;
                }
              }

              auto_continue = true;
              break;
          }
      }

      size_t tool_start = generated_text.find(toolStart);
      size_t tool_end = string::npos;
      if (tool_start != string::npos) {
          tool_end = generated_text.find(toolEnd, tool_start);
      }

execute_tool:
      if (tool_end != string::npos) {
          string tool_call = generated_text.substr(tool_start + 11, tool_end - tool_start - 11);

          // Strip anything after the closing parenthesis of the tool call to prevent circumvention of loop detection
          // We need to find the matching closing parenthesis for the tool call itself,
          // not parentheses inside string arguments
          int bracket_level = 0;
          size_t close_paren_pos = string::npos;
          bool in_string = false;
          char string_quote = '\0';

          for (size_t i = 0; i < tool_call.length(); ++i) {
              char c = tool_call[i];

              // Handle escaped characters
              if (c == '\\' && i + 1 < tool_call.length()) {
                  i++; // Skip next character
                  continue;
              }

              // Handle string literals
              if (!in_string && (c == '"' || c == '\'')) {
                  in_string = true;
                  string_quote = c;
              } else if (in_string && c == string_quote) {
                  in_string = false;
              } else if (!in_string) {
                  // Track bracket levels
                  if (c == '(') bracket_level++;
                  else if (c == ')') {
                      bracket_level--;
                      if (bracket_level == 0) {
                          close_paren_pos = i;
                          break;
                      }
                  }
              }
          }

          // If we found the matching closing parenthesis, strip everything after it
          if (close_paren_pos != string::npos && close_paren_pos + 1 < tool_call.length()) {
              tool_call = tool_call.substr(0, close_paren_pos + 1);
          }

          string preamble = "";
          if (tool_start > 0) {
              preamble = generated_text.substr(0, tool_start);
          }

          const vector<string> strip_tags = {tokenStart, tokenEnd};
          for (const auto& tag : strip_tags) {
              size_t p;
              while ((p = tool_call.find(tag)) != string::npos) {
                  tool_call.erase(p, tag.length());
              }
          }

          size_t t_start_in_unprinted = unprinted_text.find(toolStart);
          if (t_start_in_unprinted != string::npos && t_start_in_unprinted > 0) {
              printf("%s", unprinted_text.substr(0, t_start_in_unprinted).c_str());
          }

          // Extract tool name for display formatting
          string tool_name_for_display = tool_call.substr(0, tool_call.find('('));

          // Display formatting based on tool name
          string display_call = tool_call;  // Default to full tool call
          if (tool_name_for_display == "edit_file") {
              string path = extract_string_arg_bounded(tool_call, "path");
              display_call = "edit_file(path=\"" + path + "\")";
          } else if (tool_name_for_display == "write_file") {
              string path = extract_string_arg_bounded(tool_call, "path");
              display_call = "write_file(path=\"" + path + "\")";
          }

          // Log tool call directly without header for cleaner output
          string clean_log = preamble;
          if (!clean_log.empty() && clean_log.back() != '\n') clean_log += "\n";
          clean_log += display_call;
          if (chat_log.is_open()) {
              chat_log << clean_log << "\n\n";
              chat_log.flush();
          }

          printf("%s\n", display_call.c_str());
          fflush(stdout);
          unprinted_text = "";

          bool is_real_tool = false;
          // Removed "read_file_snippet(" from valid tools
          vector<string> valid_tools = {
              "read_files(", "search_file(",
              "exec_shell(", "edit_file(", "write_file(", "chmod(",
              "web_search("
          };
          for (const auto& vn : valid_tools) {
            // Find the actual tool name in the call (skip leading whitespace)
            size_t tool_start_pos = 0;
            while (tool_start_pos < tool_call.length() && isspace(tool_call[tool_start_pos])) {
              tool_start_pos++;
            }
            string trimmed_tool_call = tool_call.substr(tool_start_pos);

            if (trimmed_tool_call.find(vn) == 0) {
              is_real_tool = true;
              break;
            }
          }

          int bticks = 0; size_t pos = 0;
          while ((pos = full_response.find("```", pos)) != string::npos) { bticks++; pos += 3; }
          if (bticks % 2 != 0) is_real_tool = false;

          size_t global_ts = full_response.rfind(toolStart);
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

          string tool_name = tool_call.substr(0, tool_call.find('('));
          if (!is_real_tool) {
            tool_result = "System Error: Invalid tool format or unsupported tool. Supported tools: read_files, write_file, edit_file, chmod, exec_shell, search_file, web_search. Please try again.";
            display_result = tool_result;
          } else {
            // Removed chmod from mutating tools
            bool is_mutating_tool = (tool_name == "edit_file" || tool_name == "write_file");

            was_loop = loop_guard.check_for_loop(preamble, tool_call);
            int current_strikes = loop_guard.get_loop_strikes();

            // Execute the tool REGARDLESS of loop, to fetch actual results or specific duplicate errors
            tool_result = execute_tool_call(tool_call, clean_files, last_grep_req);

            if (stop_generation) {
              printf("\n\033[31m[Tool Interrupted by User]\033[0m\n");
              fflush(stdout);
              stop_generation = 0;
              abort_auto = true;
              break;
            }

            bool tool_failed = (tool_result.find("System Error:") != string::npos || tool_result.find("Error:") != string::npos);

            if (was_loop) {
                active_intervention_msg = get_next_loop_message();

                // Preserve match count information from tool_result before intervention
                bool has_match_count = (tool_result.find("replacement(s)") != string::npos);

                if (tool_result.find("exact grep") == string::npos) {
                    // Only use intervention message if not already showing successful edits
                    if (!has_match_count) {
                        tool_result = active_intervention_msg;
                    } else {
                        // Keep the tool result to show match count, but add intervention note
                        tool_result += "\n" + active_intervention_msg;
                    }
                }

                display_result = tool_result;

                // --- THE CIRCUIT BREAKER ---
                int max_attempts = loopMessages.size();
                int attempt_num = current_strikes - 2; // Occurrence 3 = Attempt 1

                if (attempt_num <= max_attempts) {
                    printf("\n\033[35m[System: Loop Detected. Automating intervention (Attempt %d/%d).]\033[0m\n", attempt_num, max_attempts);
                    fflush(stdout);
                    abort_auto = false;
                    inject_auto_user_msg = true; // Signal to append the user message
                } else {
                    printf("\n\033[1;31m[System: Intervention failed after %d attempts. Agent is stuck. Ejecting to prompt.]\033[0m\n", max_attempts);
                    fflush(stdout);
                    abort_auto = true;
                    intra_loop_strikes = 0;  // Reset attempt counter on eject
                }
            } else {
                // Clear loop history on successful mutations
                if (is_mutating_tool && !tool_failed) {
                    loop_guard.clear_history();
                }

                // Also clear loop history for expected error patterns that indicate the LLM is learning,
                // not actually looping. These are common patterns during editing tasks.
                bool is_expected_error = (tool_result.find("exact match not found") != string::npos ||
                                         tool_result.find("contains the 'old' string") != string::npos);

                if (is_mutating_tool && is_expected_error) {
                    loop_guard.clear_history();
                }

                // Display result based on tool name
                if (tool_name == "read_files") {
                    vector<string> paths = extract_array_arg_bounded(tool_call, "paths");
                    display_result = "Read files: ";
                    for (const auto& p : paths) display_result += p + " ";
                    if (tool_result.find("[Content omitted") != string::npos) {
                        display_result += "(Cache Hit)";
                    }
                } else if (tool_name == "web_search") {
                    string q = extract_string_arg_bounded(tool_call, "query");
                    display_result = "Web search: " + q;
                } else {
                    display_result = tool_result;
                }
            }
          }

          // Use consolidated logging for tool results
          // In debug mode, show full result; in non-debug mode, only show errors/match counts
          bool has_error = (display_result.find("Error:") != string::npos);
          bool has_match_count = (display_result.find("Match count:") != string::npos);

          // Build the log message for diagnostic output
          string log_message = "Tool Result:\n" + display_result;

          // Output to terminal based on debug mode and content type
          if (is_debug) {
            // In debug mode, show full tool result with header
            printf("\033[92m[Tool Result]\033[0m\n");
            string result_to_print = display_result;
            size_t pos = 0;
            while ((pos = result_to_print.find('\n')) != string::npos) {
                printf("  %.*s\n", (int)pos, result_to_print.c_str());
                result_to_print.erase(0, pos + 1);
            }
            if (!result_to_print.empty()) {
                printf("  %s\n", result_to_print.c_str());
            }
          } else {
            // In non-debug mode, only show errors or match count
            if (has_error || has_match_count) {
              printf("\033[92m[Tool Result]\033[0m\n");
              if (tool_name == "edit_file") {
                // For edit_file: show first line only (contains match count like "Applied 3 replacement(s)")
                string first_line = display_result;
                size_t pos = first_line.find('\n');
                if (pos != string::npos) {
                  first_line.erase(pos);
                }
                printf("  %s\n", first_line.c_str());
              } else {
                // For other tools, show truncated result (first line only)
                string result_to_print = display_result;
                size_t pos = 0;
                if ((pos = result_to_print.find('\n')) != string::npos) {
                    printf("  %.*s\n", (int)pos, result_to_print.c_str());
                } else {
                  printf("  %s\n", result_to_print.c_str());
                }
              }
            }
          }
          fflush(stdout);

          // Simplified TOOL RESULT logging - just the display_result without redundant "Tool Result:" prefix
          log_entry("TOOL RESULT", display_result);

          generated_text = ""; unprinted_text = "";

          // --- 1. ALWAYS FULFILL THE TOOL CONTRACT FIRST ---
          string tool_msg = tokenEnd + "\n" + tokenStart + "user\n[Tool Result]\n" + sanitize(tool_result) + "\n" + tokenEnd + "\n";

          // --- 2. OPEN THE ASSISTANT BLOCK (Crucial for ChatML alternating roles) ---
          tool_msg += tokenStart + "assistant\n";

          // --- 3. APPEND THE AUTOMATED INTERVENTION AS AN INTERRUPTION ---
          // By immediately injecting a user tag after the assistant tag, we perfectly mimic
          // the exact state of dropping to the prompt and the user typing a manual override.
          if (inject_auto_user_msg) {
            tool_msg += tokenStart + "user\n" + active_intervention_msg + "\n" + tokenEnd;
              log_entry("USER (Automated)", active_intervention_msg);
          }

          // Open the assistant block to trigger the next response
          tool_msg += tokenStart + "assistant\n";

          vector<llama_token> t_tokens = tokenize(tool_msg);

          if (n_past + t_tokens.size() >= cparams.n_ctx) {
              printf("\n\033[31m[Tool result too large! Context limit exhausted. Type 'clear' to reset.]\033[0m\n");
              auto_continue = false;
              break;
          }

          batch.n_tokens = 0;
          bool tool_eval_interrupted = false;

          for (size_t i = 0; i < (int)t_tokens.size(); i++) {
            if (stop_generation) {
              printf("\n\033[31m[Tool Evaluation Interrupted]\033[0m\n");
              fflush(stdout);
              stop_generation = 0;
              abort_auto = true;
              tool_eval_interrupted = true;
              break;
            }

            common_batch_add(batch, t_tokens[i], n_past++, {0}, (i == (int)t_tokens.size() - 1));

            if (batch.n_tokens == (int)cparams.n_batch && i != (int)t_tokens.size() - 1) {
              if (!handle_llama_decode_error(ctx, batch)) {
                stop_generation = 0;
                abort_auto = true;
                tool_eval_interrupted = true;
                break;
              }
              batch.n_tokens = 0;
            }
          }
          if (batch.n_tokens > 0 && !tool_eval_interrupted) {
            if (!handle_llama_decode_error(ctx, batch, "KV Cache Exhausted. Type 'clear' to reset.", false)) {
              stop_generation = 0;
              abort_auto = true;
              tool_eval_interrupted = true;
            }
          }

          if (tool_eval_interrupted) break;

          auto_continue = !abort_auto;
          intra_loop_strikes = 0;
          break;
      }

      bool partial_tag = false;
      const vector<string> hidden_tags = {tokenStart, tokenEnd, toolStart, toolEnd};
      for (const auto& tag : hidden_tags) {
        size_t pos;
        while ((pos = unprinted_text.find(tag)) != string::npos) unprinted_text.erase(pos, tag.length());
        for (size_t i = 1; i < tag.length(); i++) {
          if (unprinted_text.length() >= i && unprinted_text.substr(unprinted_text.length() - i) == tag.substr(0, i)) {
            partial_tag = true; break;
          }
        }
      }

      bool in_tool_call_stream = (generated_text.find(toolStart) != string::npos);

      // --- EOG HANDLER & UNCLOSED TAG INTERCEPTOR ---
      if (llama_vocab_is_eog(vocab, next_token) || n_past >= (int)cparams.n_ctx) {
        if (!unprinted_text.empty() && !partial_tag && !in_tool_call_stream) {
            printf("%s", unprinted_text.c_str()); fflush(stdout);
        }

        bool is_eog = llama_vocab_is_eog(vocab, next_token);
        size_t ts = generated_text.find(toolStart);
        bool unclosed = false;

        if (is_eog && ts != string::npos && generated_text.find(toolEnd, ts) == string::npos) {
            size_t global_ts = full_response.rfind(toolStart);
            if (global_ts == string::npos || global_ts == 0 || (full_response[global_ts - 1] != '\\' && full_response[global_ts - 1] != '`')) {
                unclosed = true;
            }
        }

        if (unclosed) {
             string close_str = toolEnd;
             vector<llama_token> close_tokens = tokenize(close_str);

             batch.n_tokens = 0;
             for (size_t i = 0; i < close_tokens.size(); i++) {
                 common_batch_add(batch, close_tokens[i], n_past++, {0}, true);
                 if (!handle_llama_decode_error(ctx, batch)) {
                   break;
                 }
                 batch.n_tokens = 0;
             }

             generated_text += close_str;
             full_response += close_str;
             unprinted_text += close_str;

             tool_end = generated_text.find(toolEnd, ts);
             goto execute_tool;
        }

        if (!generated_text.empty() && !unclosed) {
            log_entry("ASSISTANT", generated_text);
        }

        batch.n_tokens = 0;
        common_batch_add(batch, next_token, n_past++, {0}, true);
        if (!handle_llama_decode_error(ctx, batch)) {
          break;
        }
        auto_continue = false;
        break;
      }

      if (!partial_tag && !in_tool_call_stream && !unprinted_text.empty()) {
        printf("%s", unprinted_text.c_str()); fflush(stdout); unprinted_text = "";
      }

      t_count++;
      batch.n_tokens = 0;
      common_batch_add(batch, next_token, n_past++, {0}, true);
      if (!handle_llama_decode_error(ctx, batch)) {
        break;
      }
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();
    if (t_count > 0) printf("\n\033[34m[Speed: %.2f t/s | Elapsed: %.2fs]\033[0m\n", (t_count / elapsed), elapsed);
  }

  log_entry("SYSTEM", "Shutting down LLM Controller Session");

  llama_sampler_free(smpl);
  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
