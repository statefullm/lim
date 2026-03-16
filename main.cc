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

// --- Model Detection and Chat Template Selection ---
enum class ModelType {
    UNKNOWN,
    CHATML,      // Standard ChatML format (Nemotron, Mistral, etc.)
    LLAMA3       // Llama 3 format
};

// Function to get the appropriate chat template for a model type
std::string get_chat_template(ModelType model_type) {
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

        std::string text(token_text);

        if (text.find(Tokens::TURN_START) != std::string::npos) has_im_start = true;
        if (text.find(Tokens::TURN_END) != std::string::npos) has_im_end = true;
        if (text.find("<think>") != std::string::npos) has_reasoning_start = true;
        if (text.find("</think>") != std::string::npos) has_reasoning_end = true;
    }

    if (has_reasoning_start && has_reasoning_end) return ModelType::CHATML;
    if (has_im_start && has_im_end) return ModelType::CHATML;

    return ModelType::CHATML;
}

bool handle_llama_decode_error(llama_context *ctx, llama_batch batch, const char* error_msg = "KV Cache Exhausted. Type 'clear' to reset.", bool should_break = true) {
    int ret = llama_decode(ctx, batch);
    if (ret < -1) {
        printf("\n\033[31m[%s]\033[0m\n", error_msg);
        fflush(stdout);
        return false;
    } else if (ret == -1) {
        printf("\n\033[31m[Invalid input batch: %s]\033[0m\n", error_msg);
        fflush(stdout);
        return false;
    } else if (ret == 1 || ret == 2) {
        printf("\n\033[31m[%s]\033[0m\n", ret == 1 ? error_msg : "Aborted");
        fflush(stdout);
        if (should_break) return false;
        return true;
    }
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
bool is_debug = false;
std::ofstream chat_log;

void dummy_log_callback(enum ggml_log_level level, const char * text, void * user_data) {}

void custom_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    if (first_prompt_displayed) return;
    fprintf(stderr, "%s", text);
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
void replace_all_tags(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

// --- Tool Execution Logic ---
string execute_tool_call(const string& tool_call, set<string>& clean_files, string& last_grep_req) {
  string result = "";
  string tool_name = "";
  size_t ns = tool_call.find(Tokens::FUNC_START);
  if (ns != string::npos) {
      ns += string(Tokens::FUNC_START).length();
      size_t ne = tool_call.find('>', ns);
      if (ne != string::npos) {
          tool_name = tool_call.substr(ns, ne - ns);
      }
  }

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

            if (file.at("error").empty()) clean_files.insert(file.at("path"));
          }
        }
      }
    } else {
      result = "Error: No paths provided to read_files";
    }
  } else if (tool_name == "search_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    string text = extract_string_arg_bounded(tool_call, "text");
    replace_all_tags(text, Tokens::PARAM_END_ESC, Tokens::PARAM_END); // Unescape
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
    replace_all_tags(content, Tokens::PARAM_END_ESC, Tokens::PARAM_END); // Unescape
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
    replace_all_tags(old_str, Tokens::PARAM_END_ESC, Tokens::PARAM_END); // Unescape
    replace_all_tags(new_str, Tokens::PARAM_END_ESC, Tokens::PARAM_END); // Unescape
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
    vector<string> patterns = {Tokens::FUNC_START, Tokens::FUNC_END, Tokens::TURN_START, Tokens::TURN_END};
    for (const auto& pattern : patterns) {
      size_t pos = 0;
      while ((pos = text.find(pattern, pos)) != std::string::npos) {
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
    fprintf(stderr, "Error: This program must be run as user 'ai'\n");
    return 1;
  }

  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != nullptr) {
    ofstream cwd_file("/home/ai/.cwd");
    if (cwd_file.is_open()) {
      cwd_file << cwd << endl;
      cwd_file.close();
    }
  }

  umask(0002);
  std::atexit(NetworkTools::cleanup_searxng);
  signal(SIGINT, sigint_handler);

  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
    return 1;
  }

  const char* debug_env = getenv("LLM_DEBUG");
  if (debug_env != nullptr && (strcmp(debug_env, "1") == 0 || strcasecmp(debug_env, "true") == 0)) {
    is_debug = true;
  }

  if (!is_debug) {
    llama_log_set(dummy_log_callback, nullptr);
  } else {
    llama_log_set(custom_log_callback, nullptr);
  }

  mkdir("log", 0755);
  int log_index = 1;
  string log_file_name;
  while (true) {
      log_file_name = "log/" + to_string(log_index);
      ifstream check_file(log_file_name.c_str());
      if (!check_file.good()) break;
      log_index++;
  }
  chat_log.open(log_file_name, ios::app);

  auto log_entry = [&](const string& role, const string& text) {
      if (chat_log.is_open()) {
          string clean_text = text;
          const vector<string> tags_to_remove = {Tokens::FUNC_START, Tokens::FUNC_END, Tokens::TURN_START, Tokens::TURN_END};
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
  HOME=getenv("HOME");

  ifstream prompt_file(HOME+"/prompt");
  if (prompt_file.is_open()) {
    stringstream buffer;
    buffer << prompt_file.rdbuf();
    system_prompt = buffer.str();
    prompt_file.close();
  }

  auto tokenize = [&](string text) { return common_tokenize(ctx, text, false, true); };

  std::string formatted_system_prompt = string(Tokens::TURN_START) + "system\n" + system_prompt + Tokens::TURN_END + "\n";
  vector<llama_token> system_tokens = common_tokenize(ctx, formatted_system_prompt, true, true);

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

  /*
  // Sampling parameters: open-webui defaults
  float temp = 0.0f;
  float top_p = 0.9f;
  int32_t top_k = 40;
  float min_p = 0.0f;
  float penalty_present = 0.0f;
  float penalty_repeat = 1.1f;
  */

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
        log_entry("SYSTEM", "Context Cleared");
        printf("\033[32m[Context Cleared Successfully]\033[0m\n");
        continue;
    }

    if (user_input == "reset") {
        clean_files.clear();
        last_grep_req = "";
        loop_guard.clear_history();
        intra_loop_strikes = 0;
        llama_sampler_reset(smpl);
        log_entry("SYSTEM", "Loop Counter and File Cache Reset");
        printf("\033[32m[Loop Counter and File Cache Reset Successfully]\033[0m\n");
        continue;
    }

    if (user_input.empty() && !auto_continue) continue;

    if (!user_input.empty()) {
      if (!auto_continue) log_entry("USER", user_input);

      std::string user_message = string(Tokens::TURN_START) + "user\n" + user_input + Tokens::TURN_END + "\n" + Tokens::TURN_START + "assistant\n";

/* The "Dummy Thought" Injection
      std::string user_message = string(Tokens::TURN_START) + "user\n" +
                           user_input + Tokens::TURN_END + "\n" +
                           Tokens::TURN_START + "assistant\n<think>\nThe user  wants a direct answer. I will output the requested data immediately without preamble.\n</think>\n";
*/
      vector<llama_token> tokens = tokenize(user_message);

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

    generated_text.reserve(32768);
    unprinted_text.reserve(1024);
    full_response.reserve(32768);

    // Core state tracking for the token loop
    bool in_tool_call_stream = false;
    bool partial_tag = false;
    size_t tool_start = string::npos;
    size_t tool_end = string::npos;
    bool trigger_tool_execution = false;
    size_t func_search_pos = 0; // PERF OPTIMIZATION: Cache for O(1) searches

    // --- INNER TOKEN GENERATION LOOP ---
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

      // Stop generating if the model outputs an End-Of-Turn token
      if (llama_vocab_is_eog(vocab, next_token)) {
          size_t active_ts = generated_text.rfind(Tokens::FUNC_START);
          size_t active_te = generated_text.rfind(Tokens::FUNC_END);

          // If the model tries to end its turn while a tool is still open...
          if (active_ts != string::npos && (active_te == string::npos || active_ts > active_te)) {
              printf("\n\033[33m[System: Premature End-Of-Turn detected. Auto-recovering tags...]\033[0m\n");
              fflush(stdout);

              // Clean up a dangling "</" or "<" if the model gave up halfway
              size_t trailing_slash = generated_text.rfind("</");
              if (trailing_slash != string::npos && trailing_slash > active_ts) {
                  size_t drop_len = generated_text.length() - trailing_slash;
                  generated_text.erase(trailing_slash);
                  full_response.erase(full_response.length() - drop_len);
              }

              // Force close
              string forced_close = "\n" + string(Tokens::PARAM_END) + "\n" + string(Tokens::FUNC_END) + "\n";
              generated_text += forced_close;
              full_response += forced_close;

              tool_start = active_ts;
              tool_end = generated_text.find(Tokens::FUNC_END, active_ts);
              trigger_tool_execution = true;
              break; // Safely exit the generation loop to execute the tool
          }

          auto_continue = false;
          break; // Normal EOG, break the loop
      }

      string token_str = common_token_to_piece(ctx, next_token).c_str();
      generated_text += token_str;
      unprinted_text += token_str;
      full_response += token_str;

      // --- PERF OPTIMIZATION: O(1) TRACKING OFFSETS ---
      if (tool_start == string::npos) {
          tool_start = generated_text.find(Tokens::FUNC_START, func_search_pos);
          if (tool_start == string::npos) {
              // Cache the search position so we never scan the old story text again
              func_search_pos = generated_text.length() > 20 ? generated_text.length() - 20 : 0;
          }
      }
      if (tool_start != string::npos && tool_end == string::npos) {
          tool_end = generated_text.find(Tokens::FUNC_END, tool_start);
      }

      in_tool_call_stream = (tool_start != string::npos && tool_end == string::npos);

      // --- PERF OPTIMIZATION: O(1) PARTIAL TAG DETECTION ---
      partial_tag = false;
      size_t open_bracket = unprinted_text.rfind('<'); // Only scan the tiny unprinted buffer!
      if (open_bracket != string::npos) {
          string suffix = unprinted_text.substr(open_bracket);
          if (string(Tokens::FUNC_START).find(suffix) == 0 || string(Tokens::FUNC_END).find(suffix) == 0) {
              partial_tag = true;
          }
      }

      // --- INTRA-TURN LOOP DETECTION (Babbling Prevention) ---
      if (t_count > 0 && t_count % 10 == 0 && generated_text.length() > 100) {
          size_t n = generated_text.length();
          bool intra_loop = false;
          size_t max_len = min((size_t)3000, n / 2);
          char last_char = generated_text.back();
          // PERF OPTIMIZATION: Step by 10 to cut string comparisons by 90%
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

          // PERF OPTIMIZATION: Deleted O(N) reasoning search that slowed down text generation
          bool in_reasoning_block = false;

          if (!intra_loop && n > 300 && !in_tool_call_stream && !in_reasoning_block) {
              size_t suffix_len = 250;
              string suffix = generated_text.substr(n - suffix_len);
              size_t prev_pos = generated_text.rfind(suffix, n - suffix_len - 1);
              if (prev_pos != string::npos) intra_loop = true;
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
              size_t b_ts = generated_text.find(Tokens::FUNC_START);
              if (b_ts != string::npos && generated_text.find(Tokens::FUNC_END, b_ts) == string::npos) in_tool = true;

              if (!unprinted_text.empty() && !in_tool) {
                  printf("%s", unprinted_text.c_str());
                  fflush(stdout);
              }
              unprinted_text = "";
              log_entry("ASSISTANT (Interrupted Reasoning Loop)", generated_text);

              string active_intervention_msg = get_next_loop_message();
              string msg = string(Tokens::TURN_END) + "\n" + Tokens::TURN_START + "user\n" + active_intervention_msg + "\n" + Tokens::TURN_END + "\n" + Tokens::TURN_START + "assistant\n";
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
              break; // Exit generation loop and let outer loop handle the injection
          }
      }

      // --- TARGETED SYNTAX TRAP (Stutter Fix) ---
      // Catches Qwen's infinite `</</` loops and safely hands off to execution
      if (unprinted_text.find("</</") != string::npos) {
          printf("\n\033[33m[System: Infinite slash loop detected. Auto-recovering...]\033[0m\n");
          fflush(stdout);
          unprinted_text = "";

          if (tool_start != string::npos && tool_end == string::npos) {
              // Strip the malformed `</</` stutter so it isn't parsed
              size_t bad_pos = generated_text.rfind("</</");
              if (bad_pos != string::npos && bad_pos > tool_start) {
                  size_t drop_len = generated_text.length() - bad_pos;
                  generated_text.erase(bad_pos);
                  full_response.erase(full_response.length() - drop_len);
              }

              // Force-close the XML cleanly
              string forced_close = "\n" + string(Tokens::PARAM_END) + "\n" + string(Tokens::FUNC_END) + "\n";
              generated_text += forced_close;
              full_response += forced_close;

              tool_end = generated_text.find(Tokens::FUNC_END, tool_start);
              trigger_tool_execution = true;
              break; // Safely exit the generation loop to execute the tool
          } else {
              auto_continue = false;
              break;
          }
      }

      // --- NORMAL TOOL COMPLETION ---
      if (tool_end != string::npos) {
          trigger_tool_execution = true;
          break; // The tool tag is closed. Break the loop to execute it!
      }

      // --- PERF OPTIMIZATION: SAFETY VALVE (Terminal Printing Buffering) ---
      if (!partial_tag && !in_tool_call_stream && !unprinted_text.empty()) {
        // Buffer output: Only print to the terminal every 10 tokens or on a newline
        if (t_count % 10 == 0 || unprinted_text.back() == '\n') {
            printf("%s", unprinted_text.c_str());
            fflush(stdout);
            unprinted_text = "";
        }
      } else if (partial_tag && unprinted_text.length() > 50) {
        printf("%s", unprinted_text.c_str());
        fflush(stdout);
        unprinted_text = "";
      }

      t_count++;
      batch.n_tokens = 0;
      common_batch_add(batch, next_token, n_past++, {0}, true);
      if (!handle_llama_decode_error(ctx, batch)) {
        break;
      }
    } // END INNER TOKEN LOOP

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();
    if (t_count > 0) printf("\n\033[34m[Speed: %.2f t/s | Elapsed: %.2fs]\033[0m\n", t_count / elapsed, elapsed);

    if (stop_generation) {
      stop_generation = 0;
      auto_continue = false;
    }

    // --- TOOL EXECUTION BLOCK ---
    // If the generation loop exited and signaled a tool is ready, we process it here.
    if (trigger_tool_execution && tool_start != string::npos && tool_end != string::npos) {
        string tool_call = generated_text.substr(tool_start, tool_end - tool_start + string(Tokens::FUNC_END).length());
        string preamble = "";
        if (tool_start > 0) preamble = generated_text.substr(0, tool_start);

        const vector<string> strip_tags = {Tokens::TURN_START, Tokens::TURN_END};
        for (const auto& tag : strip_tags) {
            size_t p;
            while ((p = tool_call.find(tag)) != string::npos) tool_call.erase(p, tag.length());
        }

        size_t t_start_in_unprinted = unprinted_text.find(Tokens::FUNC_START);
        if (t_start_in_unprinted != string::npos && t_start_in_unprinted > 0) {
            printf("%s", unprinted_text.substr(0, t_start_in_unprinted).c_str());
        }

        string tool_name_for_display = "";
        size_t name_start = tool_call.find(Tokens::FUNC_START);
        if (name_start != string::npos) {
            name_start += string(Tokens::FUNC_START).length();
            size_t name_end = tool_call.find('>', name_start);
            if (name_end != string::npos) {
                tool_name_for_display = tool_call.substr(name_start, name_end - name_start);
            }
        }

        string clean_log = preamble;
        if (!clean_log.empty() && clean_log.back() != '\n') clean_log += "\n";
        clean_log += tool_call; // Log the RAW XML
        if (chat_log.is_open()) {
            chat_log << clean_log << "\n\n";
            chat_log.flush();
        }

        printf("%s\n", tool_call.c_str()); // Print RAW XML to the terminal
        fflush(stdout);
        unprinted_text = "";

        bool is_real_tool = false;
        vector<string> valid_tools = {
            "read_files", "search_file", "exec_shell", "edit_file", "write_file", "chmod", "web_search"
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

        size_t global_ts = full_response.rfind(Tokens::FUNC_START);
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
          tool_result = "System Error: Invalid tool format or unsupported tool. You MUST use the strict XML schema. Supported tools: read_files, write_file, edit_file, chmod, exec_shell, search_file, web_search. Please try again.";
          display_result = tool_result;
        } else {
          bool is_mutating_tool = (tool_name == "edit_file" || tool_name == "write_file");

          was_loop = loop_guard.check_for_loop(preamble, tool_call);
          int current_strikes = loop_guard.get_loop_strikes();

          tool_result = execute_tool_call(tool_call, clean_files, last_grep_req);

          if (stop_generation) {
            printf("\n\033[31m[Tool Interrupted by User]\033[0m\n");
            fflush(stdout);
            stop_generation = 0;
            abort_auto = true;
          }

          if (!abort_auto) {
              bool tool_failed = (tool_result.find("System Error:") != string::npos || tool_result.find("Error:") != string::npos);

              if (was_loop) {
                  active_intervention_msg = get_next_loop_message();
                  bool has_match_count = (tool_result.find("replacement(s)") != string::npos);

                  if (tool_result.find("exact grep") == string::npos) {
                      if (!has_match_count) {
                          tool_result = active_intervention_msg;
                      } else {
                          tool_result += "\n" + active_intervention_msg;
                      }
                  }

                  display_result = tool_result;

                  int max_attempts = loopMessages.size();
                  int attempt_num = current_strikes - 2;

                  if (attempt_num <= max_attempts) {
                      printf("\n\033[35m[System: Loop Detected. Automating intervention (Attempt %d/%d).]\033[0m\n", attempt_num, max_attempts);
                      fflush(stdout);
                      abort_auto = false;
                      inject_auto_user_msg = true;
                  } else {
                      printf("\n\033[1;31m[System: Intervention failed after %d attempts. Agent is stuck. Ejecting to prompt.]\033[0m\n", max_attempts);
                      fflush(stdout);
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
              printf("\033[92m[Tool Result]\033[0m\n");
              string result_to_print = display_result;
              size_t p = 0;
              while ((p = result_to_print.find('\n')) != string::npos) {
                  printf("  %.*s\n", (int)p, result_to_print.c_str());
                  result_to_print.erase(0, p + 1);
              }
              if (!result_to_print.empty()) printf("  %s\n", result_to_print.c_str());
            } else {
              if (has_error || has_match_count) {
                printf("\033[92m[Tool Result]\033[0m\n");
                if (tool_name == "edit_file") {
                  string first_line = display_result;
                  size_t p = first_line.find('\n');
                  if (p != string::npos) first_line.erase(p);
                  printf("  %s\n", first_line.c_str());
                } else {
                  string result_to_print = display_result;
                  size_t p = 0;
                  if ((p = result_to_print.find('\n')) != string::npos) {
                      printf("  %.*s\n", (int)p, result_to_print.c_str());
                  } else {
                    printf("  %s\n", result_to_print.c_str());
                  }
                }
              }
            }
            fflush(stdout);

            log_entry("TOOL RESULT", display_result);
            generated_text = ""; unprinted_text = "";

            std::string tool_result_section = string(Tokens::TURN_START) + "user\n[Tool Result]\n" + sanitize(tool_result) + Tokens::TURN_END + "\n";
            string tool_msg = tool_result_section + Tokens::TURN_START + "assistant\n";

            if (inject_auto_user_msg) {
                tool_msg += active_intervention_msg + string(Tokens::TURN_END) + "\n" + Tokens::TURN_START + "assistant\n";
            }

            vector<llama_token> t_tokens = tokenize(tool_msg);
            if (n_past + t_tokens.size() >= cparams.n_ctx) {
                printf("\n\033[31m[Context limit exhausted. Type 'clear' to reset.]\033[0m\n");
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
