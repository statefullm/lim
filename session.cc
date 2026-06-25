#include "session.h"
#include "tokens.h"
#include "output.h"
#include "server.h"
#include "model.h"
#include "loop_detector.h"
#include "signals.h"
#include "parsers.h"
#include "filesystem.h"
#include "network.h"
#include "tools.h"
#include "token_generator.h"
#include "tool_executor.h"
#include "session_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cstring>

// From main.cc
extern std::string g_model_path;
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <signal.h>
#include <cctype>
#include <set>
#include <functional>
#include <iomanip>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/select.h>

// --- Readline Headers ---
#include <readline/readline.h>
#include <readline/history.h>

// Internal readline variable: suppress _rl_callback_newline() after accept.
extern "C" { extern void (*rl_linefunc)(char *); }

// --- Custom readline function: insert literal newline (Ctrl+J) ---
static int rl_insert_newline(int /*count*/, int /*key*/) {
    rl_insert_text("\n");
    return 0;
}

using namespace std;
using namespace Tokens;

// Forward declarations for functions defined in main.cc
extern void diag(const string& msg, const char* color);
extern bool is_debug;
extern ofstream chat_log;
extern ofstream token_log;
extern bool honest_speed;

// HOME is declared as extern std::string HOME in network.h

// --- Helper to trim leading/trailing whitespace ---
static string trim(const string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// --- Load aliases from ~/.lllm_aliases ---
static map<string, string> load_aliases() {
    // Built-in commands that cannot be overridden by aliases.
    static const set<string> builtin_commands = {
        "quit", "exit", "clear", "reset", "reincarnate", "continue", "save", "help"
    };

    map<string, string> aliases;
    string path = string(HOME) + "/.lllm_aliases";
    ifstream in(path);
    if (!in.is_open()) return aliases;
    string line;
    while (getline(in, line)) {
        // Skip comments and blank lines
        string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        size_t eq = trimmed.find('=');
        if (eq == string::npos) continue;
        string key = trim(trimmed.substr(0, eq));
        string value = trim(trimmed.substr(eq + 1));
        if (!key.empty() && key[0] == '/') {
            // Strip leading '/' to get the command name
            string cmd = key.substr(1);
            if (builtin_commands.count(cmd)) {
                cerr << "Warning: alias '" << key << "' shadows a built-in command, ignored." << endl;
            } else {
                aliases[key] = value;
            }
        } else if (!key.empty()) {
            cerr << "Warning: alias key '" << key << "' ignored: update "+HOME+"/.lllm_aliases to use '/key=value' syntax." << endl;
        }
    }
    return aliases;
}

// --- Safe Multiline History Handlers ---
static void load_history_safe(const char* filename) {
    ifstream in(filename);
    string line;
    while (getline(in, line)) {
        for (char& c : line) { if (c == '\x1E') c = '\n'; }
        add_history(line.c_str());
    }
}

static void save_history_safe(const char* filename, const string& input) {
    ofstream out(filename, ios::app);
    string enc = input;
    for (char& c : enc) { if (c == '\n') c = '\x1E'; }
    out << enc << "\n";
}

// --- Helper to build context-limit diagnostic string ---
static string context_limit_diag(int n_past, int last_n_past, size_t needed, int n_ctx) {
    if (n_past == last_n_past) return "";
    ostringstream oss;
    oss << " (n_past=" << n_past;
    if (needed > 0) {
        oss << " + " << needed << ", last_n_past=" << last_n_past << ")";
    } else {
        oss << ", last_n_past=" << last_n_past << ", n_ctx=" << n_ctx << ")";
    }
    return oss.str();
}

// Local diag_speed implementation for session.cc
static void diag_speed_impl(const string& msg) {
    if (should_output_to_stdout()) {
        cout << "\033[35m[" << msg << "]\033[0m\n";
    }
    if (chat_log.is_open()) {
        chat_log << "[" << msg << "]" << "\n\n";
        chat_log.flush();
    }
    // Send Speed/Context to browser status bar (compact: no labels)
    if (should_output_to_browser()) {
        string speed_ctx = msg;
        stream_speed(speed_ctx);
    }
}

// ============================================================================
// ChatSession class: orchestrates the main chat turn loop
// ============================================================================

class ChatSession {
public:
    ChatSession(
        llama_context* ctx,
        const llama_vocab* vocab,
        llama_sampler* smpl,
        llama_batch& batch,
        int& n_past,
        const llama_context_params& cparams,
        const vector<llama_token>& system_tokens,
        bool use_dummy_thought,
        SessionState& state
    ) : ctx_(ctx), vocab_(vocab), smpl_(smpl), batch_(batch),
       n_past_(n_past), cparams_(cparams), system_tokens_(system_tokens),
       use_dummy_thought_(use_dummy_thought), state_(state),
       prev_stdout_ended_with_newline_(false),
       g_auto_continue_depth_(0)
    {
        const char* cur_tty = ttyname(STDIN_FILENO);
        prev_tty_ = cur_tty ? string(cur_tty) : "";
    }

    bool run();

private:
    enum class Command { NONE, QUIT, CLEAR, RESET, REINCARNATE, CONTINUE, SAVE, HELP };

    // Parsed command and optional save prefix
    Command last_cmd_ = Command::NONE;
    string save_prefix_;

    // --- Helper methods (extracted from lambdas) ---
    vector<llama_token> tokenize(string text) {
        return common_tokenize(ctx_, text, false, true);
    }

    void log_entry(const string& role, const string& text) {
        if (chat_log.is_open()) {
            string clean_text = text;
            vector<string> tags_to_remove = {FUNC_START, FUNC_END};
            // Strip model-specific turn markers from the log
            if (!g_model_tokens.user_turn_start.text.empty()) tags_to_remove.push_back(g_model_tokens.user_turn_start.text);
            if (!g_model_tokens.assistant_turn_start.text.empty()) tags_to_remove.push_back(g_model_tokens.assistant_turn_start.text);
            if (!g_model_tokens.system_turn_start.text.empty()) tags_to_remove.push_back(g_model_tokens.system_turn_start.text);
            if (!g_model_tokens.turn_end.text.empty()) tags_to_remove.push_back(g_model_tokens.turn_end.text);
            strip_tags(clean_text, tags_to_remove);
            while (!clean_text.empty() && isspace(clean_text.back())) clean_text.pop_back();
            chat_log << "=== " << role << " ===\n" << clean_text << "\n\n";
            chat_log.flush();
        }
    }

    bool feed_tokens_impl(const vector<llama_token>& toks) {
        batch_.n_tokens = 0;
        for (size_t i = 0; i < (int)toks.size(); i++) {
            if (stop_generation) return false;
            common_batch_add(batch_, toks[i], n_past_++, {0}, (i == (int)toks.size() - 1));
            if (batch_.n_tokens == (int)cparams_.n_batch && i != (int)toks.size() - 1) {
                if (!handle_llama_decode_error(ctx_, batch_)) { sync_n_past(ctx_, n_past_); return false; }
                batch_.n_tokens = 0;
            }
        }
        if (batch_.n_tokens > 0) {
            if (!handle_llama_decode_error(ctx_, batch_, "KV Cache Exhausted. Type '/clear' to reset.", false)) {
                sync_n_past(ctx_, n_past_);
                return false;
            }
            sync_n_past(ctx_, n_past_);
        }
        // Track all tokens fed into context for save/restore
        state_.all_context_tokens.insert(state_.all_context_tokens.end(), toks.begin(), toks.end());
        return true;
    }

    void clear_context() {
        llama_memory_clear(llama_get_memory(ctx_), true);
        n_past_ = 0;
        // Reset context token tracker to just system tokens
        state_.all_context_tokens = system_tokens_;
        feed_tokens_impl(system_tokens_);

        // Reset the current directory to the initial value so no memory of the last session persists
        {
            chdir(INITIAL_CWD.c_str());
            ofstream cwd_file(HOME + "/.cwd");
            if (cwd_file.is_open()) {
                cwd_file << INITIAL_CWD << endl;
                cwd_file.close();
            }
        }
    }

    void reset_llm_state() {
        state_.file_cache.clear();
        state_.loop_guard.clear_history();
        state_.invalid_tool_strikes = 0;
        llama_sampler_reset(smpl_);
    }

    void reset_session_state() {
        reset_llm_state();
        NetworkTools().reset_search();
        NetworkTools::reset_context_usage();
        g_browser_warning_suppressed = false;
        state_.partial_tool_text.clear();
    }

    // --- Main loop methods ---
    string get_user_input();
    Command handle_command(const string& input);
    bool feed_user_message(const string& input);
    TokenGenerator::Result generate_response();
    bool process_tool_call();
    bool handle_reincarnate_completion();

    // --- Member variables ---
    llama_context* ctx_;
    const llama_vocab* vocab_;
    llama_sampler* smpl_;
    llama_batch& batch_;
    int& n_past_;
    const llama_context_params& cparams_;
    const vector<llama_token>& system_tokens_;
    bool use_dummy_thought_;
    SessionState& state_;

    map<string, string> aliases_;
    string prev_tty_;
    bool prev_stdout_ended_with_newline_;
    int g_auto_continue_depth_;

    // Generation result shared between generate_response and process_tool_call
    TokenGenerator::Result gen_result_;
    string generated_text_;
    int t_count_;
    double elapsed_;
    bool was_mid_tool_call_;
    bool allow_continue_resume_;
    int max_auto_continue_;
};

// --- get_user_input: readline callback interface with Ctrl+J newline support ---
string ChatSession::get_user_input() {
    string user_input = "";

    if (!state_.auto_continue) {
        // Print Speed from previous generation right before >>> (skip first turn)
        if (state_.first_turn_done && state_.last_t_count > 0) {
            double context_percent = (state_.last_n_past / (double)cparams_.n_ctx) * 100.0;
            ostringstream oss;
            oss << fixed << setprecision(1);

            // Pick denominator based on honest_speed global:
            //   false (default): benchmark-style — use decode_time only
            //   true: honest wall-clock speed
            double denom = state_.last_elapsed;  // default: wall clock ("honest")
            if (!honest_speed && state_.last_decode_time > 0.0) {
                denom = state_.last_decode_time;
            }
            double speed = state_.last_t_count / denom;
            string ctx_str = std::to_string(state_.last_n_past) + " (" + std::to_string((int)context_percent) + "%)";
            string speed_str = std::to_string(round_int(speed)) + " t/s";
            diag_speed_impl(speed_str + " | " + ctx_str);
        }

        // If stdout didn't end with a newline, ensure we start on a fresh line
        // before printing the speed diagnostic.
        if (should_output_to_stdout() && !prev_stdout_ended_with_newline_) {
            cout << "\n";
            cout.flush();
        }

        const char* main_p = "\001\033[1;96m\002>>> \001\033[96m\002";

        // Bind Ctrl+J to insert a literal newline instead of accepting the line.
        // In callback mode, \r (Enter/Return) remains bound to accept-line (submit).
        // rl_bind_key('\n') works in xterm but not VS Code (pty translates \n -> \r).
        // For VS Code, the extension sends \x1c (File Separator) for Ctrl+J.
        rl_bind_key('\n', rl_insert_newline);
        rl_bind_key('\x1c', rl_insert_newline);  // File separator = Ctrl+J in VS Code

        bool input_complete = false;
        string captured_line;

        // Callback is invoked by readline when a complete line is available.
        // rl_done is unreliable here because _rl_callback_newline() resets it
        // to 0 before rl_callback_read_char() returns, so we use static
        // variables shared with the callback.
        static string g_captured_line;
        static bool g_input_complete = false;
        g_captured_line.clear();
        g_input_complete = false;

        auto storing_callback = [](char* line) {
            g_captured_line = line ? line : "";
            g_input_complete = true;
            // Suppress _rl_callback_newline() so readline doesn't redraw the
            // prompt after accepting the line.
            rl_linefunc = nullptr;
        };

        // Set screen size BEFORE installing the handler so readline knows
        // the terminal width from the start.  This avoids needing
        // rl_forced_update_display() afterward (which would duplicate the prompt).
        // SIGWINCH resizes during the session are handled by readline internally.
        {
            struct winsize ws;
            if (ioctl(0, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
                rl_set_screen_size(ws.ws_row, ws.ws_col);
            }
        }

        rl_callback_handler_install(main_p, storing_callback);

        // Event loop: poll for input with select(), check for interrupts
        while (!input_complete) {
            if (g_was_interrupted) {
                break;
            }

            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(0, &fds);  // stdin

            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 100000;  // 100ms timeout to check interrupts

            int ret = select(1, &fds, nullptr, nullptr, &tv);
            if (ret < 0) {
                if (errno == EINTR) continue;
                break;
            }

            if (ret > 0 && FD_ISSET(0, &fds)) {
                rl_callback_read_char();
            }

            // Check our static flag set by the callback instead of rl_done,
            // which gets reset to 0 by _rl_callback_newline() inside
            // rl_callback_read_char() before we can observe it.
            if (g_input_complete) {
                input_complete = true;
            }
        }

        rl_callback_handler_remove();
        rl_unbind_key('\n');  // Restore default \n binding

        // _rl_callback_newline() was suppressed (rl_linefunc = nullptr), so
        // readline left the cursor at the end of the accepted input.

        captured_line = g_captured_line;
        if (!captured_line.empty()) {
            user_input = captured_line;

            if (!user_input.empty()) {
                save_history_safe(".lllm_history", user_input);
                add_history(user_input.c_str());
            }
        } else {
            // EOF (Ctrl+D on empty line) -- treat as interrupt/break
            g_was_interrupted = 0;
        }
    }

    // Alias expansion: if user_input matches an alias key, replace with its value (single-level only).
    {
        auto alias_it = aliases_.find(user_input);
        if (alias_it != aliases_.end()) {
            user_input = alias_it->second;
        }
    }

    return user_input;
}

// --- handle_command: detect which command the input represents ---
// Commands must be prefixed with '/'.  "/save [path]" is handled specially:
//   /save              -> SAVE with empty path (saves to log/<N>.save)
//   /save cats         -> SAVE with path "cats" (saves to cats.save)
//   /save /tmp/check   -> SAVE with path "/tmp/check" (saves to /tmp/check.save)

ChatSession::Command ChatSession::handle_command(const string& input) {
    if (input.empty() || input[0] != '/') return Command::NONE;

    // Strip the leading '/'
    string rest = input.substr(1);

    // Helper function to check command with optional argument support
    static auto check_command = [](const string& rest, const char* name, int len, Command cmd) -> Command {
        if (rest.size() == len || (rest.size() > len && isspace(rest[len]))) {
            if (rest.substr(0, len) == name) {
                return cmd;
            }
        }
        return Command::NONE;
    };

    // Define command patterns with optional argument support
    // STR_CMD(name, cmd) derives len from the literal at compile time -- string appears once.
#define STR_CMD(name, cmd) { name, (int)(sizeof(name) - 1), cmd }

    static const struct CmdPattern {
        const char* name;
        int len;
        Command cmd;
    } patterns[] = {
        STR_CMD("quit", Command::QUIT),
        STR_CMD("exit", Command::QUIT),
        STR_CMD("clear", Command::CLEAR),
        STR_CMD("reincarnate", Command::REINCARNATE),
        STR_CMD("save", Command::SAVE),
        STR_CMD("reset", Command::RESET),
        STR_CMD("continue", Command::CONTINUE),
        STR_CMD("help", Command::HELP)
    };

#undef STR_CMD

    // Check patterns with optional argument support
    for (const auto& p : patterns) {
        Command result = check_command(rest, p.name, p.len, p.cmd);
        if (result != Command::NONE) {
            save_prefix_ = trim(rest.substr(p.len));
            return result;
        }
    }

    return Command::NONE;
}

// --- feed_user_message: construct and feed user message tokens ---
bool ChatSession::feed_user_message(const string& input) {
    // If user provides regular input (not "continue"), clear any pending tool interrupt state.
    if (!state_.auto_continue) state_.tool_interrupt_pending = false;

    if (!state_.auto_continue) {
        log_entry("USER", input);
        if (should_output_to_browser() && pipe_fd >= 0) {
            string user_html = "\n\n<div style=\"color: #79c0ff;\"><pre><code>" + html_escape(input) + "</code></pre></div>\n\n";
            stream_html(user_html);
        }
    }

    // Build user turn + assistant prefill using model-type-aware token vectors.
    string turn_close_str = state_.prev_was_interrupted ? (g_model_tokens.turn_end.text + "\n") : "";
    state_.prev_was_interrupted = false;

    vector<llama_token> tokens;
    if (!turn_close_str.empty()) {
        auto close_tok = common_tokenize(ctx_, turn_close_str, false, true);
        tokens.insert(tokens.end(), close_tok.begin(), close_tok.end());
    }
    auto user_ass = build_user_assistant_turn(ctx_, input);
    tokens.insert(tokens.end(), user_ass.begin(), user_ass.end());

    // If using dummy thought, append the thinking block as content tokens.
    if (use_dummy_thought_) {
        string think_block = string(THINK_START) + "\nThe user wants a direct answer. I will output the requested data immediately without preamble.\n" + THINK_END + "\n";
        auto think_tok = common_tokenize(ctx_, think_block, false, true);
        tokens.insert(tokens.end(), think_tok.begin(), think_tok.end());
    }

    if (n_past_ + (int)tokens.size() >= (int)cparams_.n_ctx) {
        string ctx_diag = context_limit_diag(n_past_, state_.last_n_past, (size_t)tokens.size(), (int)cparams_.n_ctx);
        diag("Context Limit Reached! Cannot process input" + ctx_diag + ". Type '/clear' to reset.", "\033[31m");
        return false;
    }

    if (!feed_tokens_impl(tokens)) {
        if (stop_generation) {
            diag("Input Evaluation Interrupted", "\033[31m");
            stop_generation = 0;
        }
        return false;
    }

    // Log user input tokens to token_log when debug is enabled
    log_tokens("FEED USER_INPUT", tokens, ctx_);
    return true;
}

// --- generate_response: invoke TokenGenerator and update state ---
TokenGenerator::Result ChatSession::generate_response() {
    // Reset terminal color to default before LLM text starts printing.
    // Readline draws the >>> prompt in cyan; without this reset, all LLM output
    // would appear in cyan.
    if (should_output_to_stdout()) {
        cout << "\033[0m";
        cout.flush();
    }

    g_auto_continue_depth_ = state_.auto_continue ? g_auto_continue_depth_ : 0;
    allow_continue_resume_ = false;

    // Compute turn timeout from environment
    static constexpr double DEFAULT_TURN_TIMEOUT_SEC = 300.0;
    const char* timeout_env = getenv("LLLM_TURN_TIMEOUT");
    double turn_timeout_sec = DEFAULT_TURN_TIMEOUT_SEC;
    if (timeout_env != nullptr && strlen(timeout_env) > 0) {
        char* endp = nullptr;
        double val = strtod(timeout_env, &endp);
        if (*endp == '\0') turn_timeout_sec = val;
    }
    if (turn_timeout_sec < 5.0) turn_timeout_sec = DEFAULT_TURN_TIMEOUT_SEC;

    static constexpr int DEFAULT_MAX_AUTO_CONTINUE = 500;
    const char* max_auto_env = getenv("LLLM_MAX_AUTO_CONTINUE");
    max_auto_continue_ = (max_auto_env != nullptr && strlen(max_auto_env) > 0) ? atoi(max_auto_env) : DEFAULT_MAX_AUTO_CONTINUE;
    if (max_auto_continue_ < 5) max_auto_continue_ = DEFAULT_MAX_AUTO_CONTINUE;

    // When resuming mid-tool-call via state.tool_interrupt_pending, initialize tracking
    // variables to reflect that we are already inside a tool call.
    was_mid_tool_call_ = state_.tool_interrupt_pending;
    state_.tool_interrupt_pending = false;

    auto start = chrono::high_resolution_clock::now();

    // --- TOKEN GENERATION via TokenGenerator class ---
    TokenGenerator tg(ctx_, vocab_, smpl_, batch_, n_past_, cparams_,
                      turn_timeout_sec, was_mid_tool_call_, state_.last_n_past,
                      &state_.all_context_tokens);
    gen_result_ = tg.generate();

    generated_text_ = gen_result_.text;
    t_count_ = gen_result_.token_count;

    // Handle mid-tool-call state saving (regardless of exit reason)
    if (gen_result_.tool_start != string::npos && gen_result_.tool_end == string::npos) {
        state_.tool_interrupt_pending = true;
        if (was_mid_tool_call_) {
            state_.partial_tool_text = state_.partial_tool_text + generated_text_;
        } else {
            state_.partial_tool_text = generated_text_.substr(gen_result_.tool_start);
        }
    }

    // Update session state based on exit reason
    if (gen_result_.was_interrupted) {
        state_.prev_was_interrupted = true;
        state_.auto_continue = false;
        state_.reincarnate_mode = false;
    } else if (gen_result_.early_exit) {
        state_.auto_continue = false;
        state_.reincarnate_mode = false;
    } else if (!gen_result_.has_tool_call) {
        // Normal EOG
        state_.auto_continue = false;
    }

    auto end = chrono::high_resolution_clock::now();
    elapsed_ = chrono::duration<double>(end - start).count();

    bool stdout_ended_with_newline = prev_stdout_ended_with_newline_;
    if (!generated_text_.empty() && generated_text_.back() == '\n') {
        stdout_ended_with_newline = true;
    }
    if (!stdout_ended_with_newline && should_output_to_stdout()) {
        console("\n");
        consoleFlush();
        stdout_ended_with_newline = true;
    }

    state_.last_t_count = t_count_;
    state_.last_elapsed = elapsed_;
    state_.last_decode_time = gen_result_.decode_time;
    state_.last_n_past = n_past_;
    state_.first_turn_done = true;

    prev_stdout_ended_with_newline_ = stdout_ended_with_newline;

    if (stop_generation) {
        stop_generation = 0;
        state_.auto_continue = false;
    }

    return gen_result_;
}

// --- process_tool_call: execute tool via ToolExecutor if needed ---
bool ChatSession::process_tool_call() {
    bool trigger_tool_execution = gen_result_.has_tool_call;
    size_t tool_start = gen_result_.tool_start;
    size_t tool_end = gen_result_.tool_end;

    if (trigger_tool_execution && tool_start != string::npos && tool_end != string::npos) {
        string full_generated = generated_text_;

        auto tool_result = ToolExecutor::execute(
            state_, generated_text_, full_generated,
            tool_start, tool_end, was_mid_tool_call_,
            [this](string text) { return tokenize(text); },
            [this](const vector<llama_token>& toks) { return feed_tokens_impl(toks); },
            ctx_, n_past_, cparams_,
            g_auto_continue_depth_, max_auto_continue_,
            allow_continue_resume_
        );

        if (tool_result.should_auto_continue) {
            return true; // Signal to continue outer loop
        }
    }

    return false;
}

// --- handle_reincarnate_completion: post-generation reincarnate logic ---
bool ChatSession::handle_reincarnate_completion() {
    if (!state_.reincarnate_mode || state_.auto_continue) return false;

    state_.reincarnate_mode = false;

    string userprompt_path = string(HOME) + "/userprompt";

    ifstream userprompt_file(userprompt_path);
    if (!userprompt_file.is_open()) {
        diag("Reincarnate failed: LLM did not write " + userprompt_path + ". Session will not be reincarnated.", "\033[31m");
        log_entry("SYSTEM", "Reincarnate failed: userprompt was not written by LLM");
        return true; // continue outer loop
    }

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
        return true; // continue outer loop
    }

    diag("Clearing context and starting reincarnated session...", "\033[35m");
    clear_context();

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

    vector<llama_token> new_session_tokens = build_user_assistant_turn(ctx_, follow_prompt);

    if (n_past_ + (int)new_session_tokens.size() >= (int)cparams_.n_ctx) {
        string ctx_diag = context_limit_diag(n_past_, state_.last_n_past, (size_t)new_session_tokens.size(), (int)cparams_.n_ctx);
        diag("Context Limit Reached! Cannot process reincarnated prompt" + ctx_diag + ".", "\033[31m");
        return true; // continue outer loop
    }

    if (!feed_tokens_impl(new_session_tokens)) {
        if (stop_generation) stop_generation = 0;
        diag("Failed to feed reincarnated session tokens. Type '/clear' to reset.", "\033[31m");
        return true; // continue outer loop
    }

    // Log reincarnated session tokens to token_log when debug is enabled
    log_tokens("FEED USER_INPUT", new_session_tokens, ctx_);

    state_.auto_continue = true;
    reset_session_state();
    log_entry("SYSTEM", "Context Cleared and Reincarnated with New Prompt");
    return true; // continue outer loop
}

// Helper: build a save path from an optional user-supplied prefix.
// If prefix is empty, returns the default autosave path.
// Appends .save if not already present.
static string make_save_path(const string& prefix, const string& default_path) {
    if (prefix.empty()) return default_path;
    string path = prefix;
    if (path.size() < 5 || path.substr(path.size() - 5) != ".save") {
        path += ".save";
    }
    return path;
}

// Helper: compact save -- write only the token sequence (not the raw KV cache).
// On restore, tokens are re-decoded through the model to regenerate the KV cache.
// For a 75K-token session this produces ~300 KB instead of ~2 GB.
static bool save_session_with_header(const vector<llama_token>& tokens, const string& path,
                                     bool write_v1 = false, llama_context* ctx = nullptr) {
    if (tokens.empty()) return false;
    bool ok = write_token_save(path, tokens);

    // Also write V1 cache for instant future restores (only on explicit /save)
    if (ok && write_v1) {
        // Resolve to absolute path for consistent cache key
        char abs_buf[4096];
        string abs_path = path;
        if (realpath(path.c_str(), abs_buf)) abs_path = abs_buf;

        FILE* pipe = popen("git rev-parse HEAD 2>/dev/null", "r");
        string git_sha;
        if (pipe) {
            char buf[48];
            if (fgets(buf, sizeof(buf), pipe)) {
                git_sha = buf;
                while (!git_sha.empty() && (git_sha.back() == '\n' || git_sha.back() == '\r'))
                    git_sha.pop_back();
            }
            pclose(pipe);
        }
        if (!git_sha.empty()) {
            if (is_debug) diag("Save to cache.", "\033[35m");
            write_v1_cache(abs_path, g_model_path, git_sha, ctx);
        } else {
            // No git repo -- still cache using just path + model hash
            if (is_debug) diag("Save to cache.", "\033[35m");
            write_v1_cache(abs_path, g_model_path, "", ctx);
        }
    }
    return ok;
}

// --- run: the main chat turn loop ---
bool ChatSession::run() {
    const char* history_file = ".lllm_history";
    load_history_safe(history_file);

    // Load user-defined aliases from ~/.lllm_aliases
    aliases_ = load_aliases();

    // --- MAIN CHAT TURN LOOP ---
    while (true) {
        stop_generation = 0;
        g_was_interrupted = 0;

        // 1. Get user input
        string user_input = get_user_input();

        // 2. Parse and dispatch commands (all require '/' prefix)
        last_cmd_ = handle_command(user_input);

        if (last_cmd_ == Command::QUIT) {
            // Auto-save before exiting so the user's current work is preserved.
            string autosave_path = make_save_path(save_prefix_, "log/" + to_string(state_.log_index) + ".save");
            bool ok = save_session_with_header(state_.all_context_tokens, autosave_path);
            if (!ok) {
                diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
            } else {
                diag("Auto-saved to " + autosave_path + " (" + to_string(state_.all_context_tokens.size()) + " tokens)", "\033[35m");
            }
            return false;
        }

        if (last_cmd_ == Command::CLEAR) {
            // Auto-save before clearing so nothing is truly lost.
            // Uses a distinct name (e.g., log/5-clear.save) so it doesn't conflict
            // with the regular save file that /quit or /exit will overwrite.
            {
                string autosave_path = make_save_path(save_prefix_, "log/" + to_string(state_.log_index) + "-clear.save");
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path);
                if (!ok) {
                    diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
                } else {
                    diag("Auto-saved to " + autosave_path + " (" + to_string(state_.all_context_tokens.size()) + " tokens)", "\033[35m");
                }
            }

            clear_context();
            state_.auto_continue = false;
            state_.prev_was_interrupted = false;
            reset_session_state();
            state_.last_t_count = 0;
            state_.last_elapsed = 0.0;
            state_.last_n_past = n_past_;
            log_entry("SYSTEM", "Context Cleared");

            // Update browser: clear the viewer and immediately set the new
            // context diagnostic in a single pipe write so they arrive together.
            if (should_output_to_browser()) {
                double context_percent = (n_past_ / (double)cparams_.n_ctx) * 100.0;
                string ctx_str = std::to_string(n_past_) + " (" + std::to_string((int)context_percent) + "%)";
                const char soh = 0x01;
                pipe_write(&soh, 1);
                pipe_write(&SEG_SPEED, 1);
                string speed_msg = "Cleared | " + ctx_str;
                pipe_write(speed_msg.c_str(), speed_msg.length());
            }

            diag("Context Cleared Successfully", "\033[32m");
            continue;
        }

        if (last_cmd_ == Command::RESET) {
            reset_llm_state();
            log_entry("SYSTEM", "Loop Counter and File Cache Reset");
            diag("Loop Counter and File Cache Reset Successfully", "\033[32m");
            continue;
        }

        if (last_cmd_ == Command::REINCARNATE) {
            // Auto-save before reincarnating so nothing is truly lost.
            // Uses the same -clear.save name since reincarnate calls clear_context internally.
            {
                string autosave_path = make_save_path(save_prefix_, "log/" + to_string(state_.log_index) + "-clear.save");
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path);
                if (!ok) {
                    diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
                } else {
                    diag("Auto-saved to " + autosave_path + " (" + to_string(state_.all_context_tokens.size()) + " tokens)", "\033[35m");
                }
            }

            string reincarnate_path = string(HOME) + "/reincarnate";
            ifstream reincarnate_file(reincarnate_path);
            if (!reincarnate_file.is_open()) {
                diag("Reincarnate failed: Cannot open " + reincarnate_path, "\033[31m");
                continue;
            }
            stringstream reincarnate_buffer;
            reincarnate_buffer << "Use the write_file tool to write a new prompt to "
                          << HOME << "/userprompt. Read the following instructions and compose an appropriate prompt, then write it and return without further comment. "
                          << reincarnate_file.rdbuf();
            string reincarnate_text = reincarnate_buffer.str();
            reincarnate_file.close();

            string userprompt_check_path = string(HOME) + "/userprompt";
            {
                ofstream userprompt_clear(userprompt_check_path, ios::trunc);
                if (!userprompt_clear.is_open()) {
                    diag("Warning: Cannot truncate " + userprompt_check_path + " before reincarnate.", "\033[33m");
                }
            }

            diag("Sending reincarnate request to LLM...", "\033[35m");
            log_entry("USER", "[reincarnate] " + reincarnate_text);

            vector<llama_token> reincarnate_tokens;
            if (state_.prev_was_interrupted) {
                // Close the interrupted assistant turn first
                auto close_tok = common_tokenize(ctx_, g_model_tokens.turn_end.text + "\n", false, true);
                reincarnate_tokens.insert(reincarnate_tokens.end(), close_tok.begin(), close_tok.end());
            }
            auto user_ass_reinc = build_user_assistant_turn(ctx_, reincarnate_text);
            reincarnate_tokens.insert(reincarnate_tokens.end(), user_ass_reinc.begin(), user_ass_reinc.end());
            state_.prev_was_interrupted = false;

            if (n_past_ + (int)reincarnate_tokens.size() >= (int)cparams_.n_ctx) {
                string ctx_diag = context_limit_diag(n_past_, state_.last_n_past, (size_t)reincarnate_tokens.size(), (int)cparams_.n_ctx);
                diag("Context Limit Reached! Cannot process reincarnate" + ctx_diag + ". Type '/clear' to reset.", "\033[31m");
                continue;
            }

            if (!feed_tokens_impl(reincarnate_tokens)) {
                if (stop_generation) stop_generation = 0;
                continue;
            }

            // Log reincarnate tokens to token_log when debug is enabled
            log_tokens("FEED USER_INPUT", reincarnate_tokens, ctx_);

            state_.auto_continue = true;
            state_.reincarnate_mode = true;
            reset_session_state();
            continue;
        }

        if (last_cmd_ == Command::SAVE) {
            string save_path = make_save_path(save_prefix_, "log/" + to_string(state_.log_index) + ".save");

            diag("Saving session to " + save_path + "...", "\033[35m");

            bool ok = save_session_with_header(state_.all_context_tokens, save_path, true /* write_v1 */, ctx_);
            if (!ok) {
                diag("Save failed: could not write " + save_path, "\033[31m");
            } else {
                diag("Session saved to " + save_path + " (" + to_string(state_.all_context_tokens.size()) + " tokens)", "\033[32m");
                log_entry("SYSTEM", "Session saved to " + save_path);
            }
            continue;
        }

        if (last_cmd_ == Command::HELP) {
            diag("Available Commands:", "\033[1;36m");
            diag("  /quit or /exit [path]  Save to [path].save and exit (default: log/<N>.save)", "\033[37m");
            diag("  /clear [path]          Clear context (auto-saves first to log/<N>-clear.save, or [path].save)", "\033[37m");
            diag("  /reset                 Reset loop detector and file cache", "\033[37m");
            diag("  /reincarnate [path]    Compose new prompt in ~/userprompt, then restart (auto-saves first)", "\033[37m");
            diag("  /continue              Resume generation after interruption", "\033[37m");
            diag("  /save [path]           Save session state (default: log/<N>.save)", "\033[37m");
            diag("  /help               Show this help message", "\033[37m");
            diag("Multi-line input: Ctrl+J to insert newline, Enter to submit", "\033[1;36m");
            continue;
        }

        if (last_cmd_ == Command::CONTINUE) {
            if (state_.tool_interrupt_pending) {
                state_.prev_was_interrupted = false;
                diag("Resuming after tool interruption...", "\033[1;33m");
                state_.auto_continue = true;
                state_.auto_continue_depth_val = 0;
                user_input = "";
            } else if (state_.prev_was_interrupted) {
                state_.prev_was_interrupted = false;
                diag("Resuming generation...", "\033[1;33m");
                state_.auto_continue = true;
                state_.auto_continue_depth_val = 0;
                user_input = "";
            } else {
                // Not interrupted -- nothing to resume. Return silently to prompt.
                continue;
            }
        }

        // 3. Skip empty input when not auto-continuing
        if (user_input.empty() && !state_.auto_continue) continue;

        // 8. Browser/TTY setup
        bool browser_connected = check_browser_connected();

        const char* cur_tty = ttyname(STDIN_FILENO);
        if (cur_tty && !prev_tty_.empty() && prev_tty_ != cur_tty) {
            system("reset");
        }
        if (cur_tty) prev_tty_ = cur_tty;

        if (browser_connected && g_browser_warning_suppressed) {
            g_browser_warning_suppressed = false;
        }

        if (should_output_to_browser()) {
            if (!g_browser_warning_suppressed && !browser_connected) {
                browser_connected = prompt_for_browser_connection();
                if (!browser_connected) {
                    disable_browser_output();
                }
            }
        }

        // 9. Feed user message (if non-empty)
        if (!user_input.empty()) {
            if (!feed_user_message(user_input)) continue;
        }

        // 10. Generate response
        auto gen_result = generate_response();

        // 11. Process tool call
        if (process_tool_call()) continue;

        // 12. Log assistant output
        if (!state_.auto_continue && !gen_result.text.empty()) log_entry("ASSISTANT", gen_result.text);

        // 13. Handle reincarnate completion
        if (handle_reincarnate_completion()) continue;
    }

    return true;
}

// ============================================================================
// run_chat_session: thin wrapper that creates a ChatSession and runs it
// ============================================================================

bool run_chat_session(
    llama_context* ctx,
    const llama_vocab* vocab,
    llama_sampler* smpl,
    llama_batch& batch,
    int& n_past,
    const llama_context_params& cparams,
    const vector<llama_token>& system_tokens,
    bool use_dummy_thought,
    SessionState& state
) {
    ChatSession session(ctx, vocab, smpl, batch, n_past, cparams,
                        system_tokens, use_dummy_thought, state);
    return session.run();
}
