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

// --- Load aliases from ~/.lim_aliases ---
static map<string, string> load_aliases() {
    // Built-in commands that cannot be overridden by aliases.
    static const set<string> builtin_commands = {
        "quit", "exit", "clear", "undo", "reset", "reincarnate", "continue", "save", "help"
    };

    map<string, string> aliases;
    string path = string(HOME) + "/.lim_aliases";
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
            cerr << "Warning: alias key '" << key << "' ignored: update "+HOME+"/.lim_aliases to use '/key=value' syntax." << endl;
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
    enum class Command { NONE, QUIT, CLEAR, RESET, REINCARNATE, CONTINUE, SAVE, HELP, UNDO };

    // Parsed command and optional arguments
    Command last_cmd_ = Command::NONE;
    string save_prefix_;
    int undo_count_ = 1;
    // Track if the previous turn was a manual save, so /quit can skip redundant auto-save
    bool prev_was_save_ = false;
    // Track if we already logged assistant output this turn (via process_tool_call),
    // so step 12 doesn't duplicate it.
    bool assistant_logged_this_turn_ = false;
    // Last non-empty user input, used as checkpoint label for tool-call turns
    string last_user_input_;

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
        // Reset context token tracker to empty; feed_tokens_impl will rebuild it.
        state_.all_context_tokens.clear();
        state_.prompt_checkpoints.clear();
        feed_tokens_impl(system_tokens_);

        // Reset sampler state (penalty history, RNG) for a fresh start
        llama_sampler_reset(smpl_);

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
        state_.loop_guard.clear_history();
        state_.invalid_tool_strikes = 0;
    }

    void reset_session_state() {
        reset_llm_state();
        NetworkTools().reset_search();
        NetworkTools::reset_context_usage();
        g_browser_warning_suppressed = false;
        state_.partial_tool_text.clear();
        state_.tool_interrupt_pending = false;
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
        // If stdout didn't end with a newline, ensure we start on a fresh line
        // before printing the speed diagnostic.
        if (should_output_to_stdout() && !prev_stdout_ended_with_newline_) {
            cout << "\n";
            cout.flush();
        }

        // Print Speed from previous generation right before >>> (skip first turn)
        if (state_.first_turn_done && state_.last_t_count > 0) {
            double context_percent = (state_.last_n_past / (double)cparams_.n_ctx) * 100.0;
            ostringstream oss;
            oss << fixed << setprecision(1);

            // Pick denominator based on honest_speed global:
            //   false (default): sample+sync window (first to last token),
            //                     matching llama-cli's "Generation: X t/s"
            //   true: full wall-clock elapsed time including pre/post overhead
            double denom = state_.last_elapsed;  // default: wall clock ("honest")
            if (!honest_speed && state_.last_decode_time > 0.0) {
                denom = state_.last_decode_time;
            }
            double speed = state_.last_t_count / denom;
            string ctx_str = std::to_string(state_.last_n_past) + " (" + std::to_string((int)context_percent) + "%)";
            string speed_str = std::to_string(round_int(speed)) + " t/s";
            diag_speed_impl(speed_str + " | " + ctx_str);
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
                save_history_safe(".lim_history", user_input);
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
// Commands must be prefixed with '/'.  Only /save accepts an optional argument:
//   /save              -> SAVE with empty path (saves to log/<N>.save)
//   /save cats         -> SAVE with path "cats" (saves to cats.save)
//   /save /tmp/check   -> SAVE with path "/tmp/check" (saves to /tmp/check.save)

ChatSession::Command ChatSession::handle_command(const string& input) {
    if (input.empty() || input[0] != '/') return Command::NONE;

    // Strip the leading '/'
    string rest = input.substr(1);

    // Helper: exact match -- command must be followed by nothing or whitespace.
    static auto check_command = [](const string& rest, const char* name, int len, Command cmd) -> Command {
        if (rest.size() == len || (rest.size() > len && isspace(rest[len]))) {
            if (rest.substr(0, len) == name) {
                return cmd;
            }
        }
        return Command::NONE;
    };

        static const struct CmdPattern {
        const char* name;
        int len;
        Command cmd;
    } patterns[] = {
        STR("quit", Command::QUIT),
        STR("exit", Command::QUIT),
        STR("clear", Command::CLEAR),
        STR("reincarnate", Command::REINCARNATE),
        STR("reset", Command::RESET),
        STR("continue", Command::CONTINUE),
        STR("help", Command::HELP)
    };

    // Check exact-match commands first (no arguments allowed)
    for (const auto& p : patterns) {
        Command result = check_command(rest, p.name, p.len, p.cmd);
        if (result != Command::NONE) {
            save_prefix_.clear();
            return result;
        }
    }

    // /save is special: it accepts an optional path argument.
    static constexpr const char* SAVE_NAME = "save";
    static constexpr int SAVE_LEN = 4;
    if (rest.size() == SAVE_LEN || (rest.size() > SAVE_LEN && isspace(rest[SAVE_LEN]))) {
        if (rest.substr(0, SAVE_LEN) == SAVE_NAME) {
            save_prefix_ = trim(rest.substr(SAVE_LEN));
            return Command::SAVE;
        }
    }

    // /undo is special: it accepts an optional integer argument.
    static constexpr const char* UNDO_NAME = "undo";
    static constexpr int UNDO_LEN = 4;
    if (rest.size() == UNDO_LEN || (rest.size() > UNDO_LEN && isspace(rest[UNDO_LEN]))) {
        if (rest.substr(0, UNDO_LEN) == UNDO_NAME) {
            string arg = trim(rest.substr(UNDO_LEN));
            undo_count_ = 1;
            if (!arg.empty()) {
                try {
                    int val = stoi(arg);
                    if (val > 0) undo_count_ = val;
                } catch (...) {
                    // If parsing fails, default to 1.
                }
            }
            return Command::UNDO;
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
    string turn_close_str = state_.prev_was_interrupted ? g_model_tokens.turn_end.text : "";
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
    const char* timeout_env = getenv("LIM_TURN_TIMEOUT");
    double turn_timeout_sec = DEFAULT_TURN_TIMEOUT_SEC;
    if (timeout_env != nullptr && strlen(timeout_env) > 0) {
        char* endp = nullptr;
        double val = strtod(timeout_env, &endp);
        if (*endp == '\0') turn_timeout_sec = val;
    }
    if (turn_timeout_sec < 5.0) turn_timeout_sec = DEFAULT_TURN_TIMEOUT_SEC;

    static constexpr int DEFAULT_MAX_AUTO_CONTINUE = 500;
    const char* max_auto_env = getenv("LIM_MAX_AUTO_CONTINUE");
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

    // Signal the viewer that generation is complete so it can render
    // remaining raw tails (no arbitrary timeout needed).
    pipe_write(&SEG_TURN_END, 1);

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

    // TokenGenerator::generate() already flushes remaining unprinted text to
    // stdout with a trailing newline. If stdout was enabled during this turn,
    // we know it ends with \n. If not, preserve the previous state.
    if (should_output_to_stdout()) {
        prev_stdout_ended_with_newline_ = true;
    }

    state_.last_t_count = t_count_;
    state_.last_elapsed = elapsed_;
    state_.last_decode_time = gen_result_.decode_time;
    state_.last_n_past = n_past_;
    state_.first_turn_done = true;

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
        // Log the assistant's preamble text (text before the tool call) so it
        // appears in the chat log just like it does in the browser.
        if (tool_start > 0) {
            string preamble = generated_text_.substr(0, tool_start);
            log_entry("ASSISTANT", preamble);
        }

        // Log the tool call itself so the chat log shows what was invoked.
        // Write directly (not via log_entry) to preserve FUNC_START/FUNC_END tags.
        {
            string tool_call = generated_text_.substr(tool_start, tool_end - tool_start + string(FUNC_END).length());
            if (chat_log.is_open()) {
                chat_log << "=== TOOL_CALL ===\n" << tool_call << "\n\n";
                chat_log.flush();
            }
        }

        // Mark that we've logged assistant output this turn so step 12 doesn't duplicate.
        assistant_logged_this_turn_ = true;

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
    if (path.size() < std::strlen(SAVE_EXT) || path.compare(path.size() - std::strlen(SAVE_EXT), std::strlen(SAVE_EXT), SAVE_EXT) != 0) {
        path += SAVE_EXT;
    }
    return path;
}

// Helper: format a save diagnostic with proper pluralization.
static string save_diag(size_t n_checkpoints, size_t n_tokens) {
    return to_string(n_checkpoints) + " checkpoint" + (n_checkpoints != 1 ? "s" : "")
         + ", " + to_string(n_tokens) + " token" + (n_tokens != 1 ? "s" : "");
}

// Helper: compact save -- write only the token sequence (not the raw KV cache).
// On restore, tokens are re-decoded through the model to regenerate the KV cache.
// For a 75K-token session this produces ~300 KB instead of ~2 GB.
// When checkpoints are available, writes V3 format with prompt-return positions.
static bool save_session_with_header(const vector<llama_token>& tokens, const string& path,
                                     bool write_v1 = false, llama_context* ctx = nullptr,
                                     const vector<PromptCheckpoint>* checkpoints = nullptr) {
    if (tokens.empty()) return false;

    // Read old tokens from the existing save file *before* overwriting it,
    // so write_v1_cache can compute and delete the stale cache entry.
    vector<llama_token> old_tokens;
    string old_hash;
    if (write_v1) {
        if (read_token_save(path, old_tokens)) {
            old_hash = cache_hash_for_save(old_tokens, g_model_path);
        }
    }

    // Write V3 format.
    bool ok = write_token_save_v3(path, tokens, *checkpoints);

    // Also write V1 cache for instant future restores (only on explicit /save)
    if (ok && write_v1) {
        char abs_buf[4096];
        string abs_path = path;
        if (realpath(path.c_str(), abs_buf)) abs_path = abs_buf;

        if (is_debug) diag("Save to cache.", "\033[35m");
        write_v1_cache(abs_path, tokens, g_model_path, ctx, old_hash);
    }
    return ok;
}

// --- run: the main chat turn loop ---
bool ChatSession::run() {
    const char* history_file = ".lim_history";
    load_history_safe(history_file);

    // Push saved prompt checkpoints onto readline history so up-arrow
    // navigates through the session as it appeared when /save was called.
    using_history();
    for (const auto& cp : state_.prompt_checkpoints) {
        if (!cp.prompt.empty()) {
            add_history(cp.prompt.c_str());
        }
    }

    // Load user-defined aliases from ~/.lim_aliases
    aliases_ = load_aliases();

    // --- MAIN CHAT TURN LOOP ---
    while (true) {
        stop_generation = 0;
        g_was_interrupted = 0;
        prev_was_save_ = false;
        assistant_logged_this_turn_ = false;

        // 1. Get user input
        string user_input = get_user_input();

        // 2. Parse and dispatch commands (all require '/' prefix)
        last_cmd_ = handle_command(user_input);

        if (last_cmd_ == Command::QUIT) {
            // If there's actual conversation to preserve, auto-save before exiting.
            // Skip if the user just manually saved -- nothing has changed since then.
            if (!state_.prompt_checkpoints.empty() && !prev_was_save_) {
                string autosave_path = "log/" + to_string(state_.log_index) + ".save";
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path, false, nullptr, &state_.prompt_checkpoints);
                if (!ok) {
                    diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
                } else {
                    diag("Auto-saved to " + autosave_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[35m");
                }
            }
            return false;
        }

        if (last_cmd_ == Command::CLEAR) {
            // Auto-save before clearing so nothing is truly lost.
            // Uses a distinct name (e.g., log/5-clear.save) so it doesn't conflict
            // with the regular save file that /quit or /exit will overwrite.
            {
                string autosave_path = "log/" + to_string(state_.log_index) + "-clear.save";
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path, false, nullptr, &state_.prompt_checkpoints);
                if (!ok) {
                    diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
                } else {
                    diag("Auto-saved to " + autosave_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[35m");
                }
            }

            clear_context();
            state_.file_cache.clear();
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

        if (last_cmd_ == Command::UNDO) {
            // Auto-save before undoing so nothing is truly lost.
            {
                string autosave_path = "log/" + to_string(state_.log_index) + "-clear.save";
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path, false, nullptr, &state_.prompt_checkpoints);
                if (!ok) {
                    diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
                } else {
                    diag("Auto-saved to " + autosave_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[35m");
                }
            }

            // Handle zero/negative: NOP, return silently to prompt.
            if (undo_count_ <= 0) continue;



            // If no checkpoints remain after undo, or N exceeds available, fall back to /clear.
            if (state_.prompt_checkpoints.empty() || undo_count_ >= (int)state_.prompt_checkpoints.size()) {
                clear_context();
                state_.file_cache.clear();
                state_.auto_continue = false;
                state_.prev_was_interrupted = false;
                reset_session_state();
                state_.last_t_count = 0;
                state_.last_elapsed = 0.0;
                state_.last_n_past = n_past_;
                log_entry("SYSTEM", "Context Cleared (undo fallback)");

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

            // Find the first checkpoint to remove (target_idx), then restore
            // to the state recorded by the checkpoint just before it.
            int target_idx = (int)state_.prompt_checkpoints.size() - undo_count_;
            PromptCheckpoint& target = state_.prompt_checkpoints[target_idx - 1];

            diag("Undoing " + to_string(undo_count_) + " prompt" + (undo_count_ > 1 ? "s" : "") + " (restoring to: \"" + target.prompt.substr(0, min((int)target.prompt.size(), 60)) + "\")", "\033[35m");

            // Truncate the token tracker to what we're keeping.
            state_.all_context_tokens.resize(target.n_past);

            // Undo via seq_rm.  For pure attention models this works instantly.
            // For hybrid models (Qwen3.5/3.6) we first restore the saved recurrent
            // checkpoint, then seq_rm succeeds because plane 0 already holds the
            // correct R/S state - instant undo with no re-decode.
            // Falls back to clear+re-decode only if no checkpoint is available.
            {
                llama_memory_t mem = llama_get_memory(ctx_);

                // Translate the prompt_checkpoints index into a live stack index.
                // After a fast restore, historical checkpoints from the save file
                // have no corresponding entries in the recurrent checkpoint stack;
                // only checkpoints saved during this session are present.
                int stack_idx = (target_idx - 1) - state_.checkpoint_stack_offset;

                if (stack_idx >= 0) {
                    // Restore recurrent state from the target checkpoint.
                    // No-op for pure attention models (no recurrent state).
                    llama_memory_rs_checkpoint_restore(mem, 0, (uint32_t)stack_idx);
                }

                bool ok = llama_memory_seq_rm(mem, 0, target.n_past, -1);
                if (ok) {
                    n_past_ = target.n_past;
                    // Prune stale recurrent checkpoints beyond the restored index.
                    if (stack_idx >= 0) {
                        llama_memory_rs_checkpoint_prune(mem, 0, (uint32_t)stack_idx);
                    }
                } else {
                    diag("Regenerating KV cache for " + to_string(target.n_past) + " tokens...", "\033[35m");
                    llama_memory_clear(mem, true);
                    n_past_ = 0;

                    auto start = chrono::high_resolution_clock::now();
                    if (!feed_tokens_impl(state_.all_context_tokens)) {
                        diag("Failed to re-feed tokens after undo. Type '/clear' to reset.", "\033[31m");
                        continue;
                    }
                    state_.all_context_tokens.resize(target.n_past);

                    auto end = chrono::high_resolution_clock::now();
                    double elapsed = chrono::duration<double>(end - start).count();
                    int secs = (int)elapsed;
                    double speed = target.n_past / (elapsed > 0 ? elapsed : 1.0);
                    diag("KV cache regenerated: " + to_string(target.n_past) + " tokens at " + to_string(round_int(speed)) + " t/s (" + to_string(secs) + "s)", "\033[35m");
                }
            }

            // Erase all checkpoints at and beyond the undo boundary.
            // Adjust the stack offset: after erasing, count how many remaining
            // prompt_checkpoints are still historical (before the live stack).
            int erased_count = 0;
            for (int i = target_idx; i < (int)state_.prompt_checkpoints.size(); i++) {
                if (i < state_.checkpoint_stack_offset) erased_count++;
            }
            state_.prompt_checkpoints.erase(state_.prompt_checkpoints.begin() + target_idx, state_.prompt_checkpoints.end());
            state_.checkpoint_stack_offset = std::max(0, state_.checkpoint_stack_offset - erased_count);

            // Reset generation state so the session is ready for a fresh user prompt.
            state_.auto_continue = false;
            state_.prev_was_interrupted = false;
            reset_session_state();
            state_.last_t_count = 0;
            state_.last_elapsed = 0.0;
            state_.last_n_past = n_past_;

            log_entry("SYSTEM", "Undid " + to_string(undo_count_) + " prompt" + (undo_count_ > 1 ? "s" : ""));

            if (should_output_to_browser()) {
                double context_percent = (n_past_ / (double)cparams_.n_ctx) * 100.0;
                string ctx_str = std::to_string(n_past_) + " (" + std::to_string((int)context_percent) + "%)";
                const char soh = 0x01;
                pipe_write(&soh, 1);
                pipe_write(&SEG_SPEED, 1);
                string speed_msg = "Undid " + to_string(undo_count_) + " | " + ctx_str;
                pipe_write(speed_msg.c_str(), speed_msg.length());
            }

            diag("Undo successful", "\033[32m");
            continue;
        }

        if (last_cmd_ == Command::RESET) {
            reset_llm_state();
            log_entry("SYSTEM", "Loop Counter Reset");
            diag("Loop Counter Reset Successfully", "\033[32m");
            continue;
        }

        if (last_cmd_ == Command::REINCARNATE) {
            // Auto-save before reincarnating so nothing is truly lost.
            // Uses the same -clear.save name since reincarnate calls clear_context internally.
            {
                string autosave_path = "log/" + to_string(state_.log_index) + "-clear.save";
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path, false, nullptr, &state_.prompt_checkpoints);
                if (!ok) {
                    diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
                } else {
                    diag("Auto-saved to " + autosave_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[35m");
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
                auto close_tok = common_tokenize(ctx_, g_model_tokens.turn_end.text, false, true);
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

            // Named saves (with a meaningful argument) are cached in fast format
            // for instant restore. Unnamed saves are compact-only since they're
            // ephemeral and overwritten each session.
            bool write_v1 = !save_prefix_.empty();

            diag("Saving session to " + save_path + "...", "\033[35m");

            bool ok = save_session_with_header(state_.all_context_tokens, save_path, write_v1, ctx_, &state_.prompt_checkpoints);
            if (!ok) {
                diag("Save failed: could not write " + save_path, "\033[31m");
            } else {
                diag("Session saved to " + save_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[32m");
                log_entry("SYSTEM", "Session saved to " + save_path);
                prev_was_save_ = true;
            }
            continue;
        }

        if (last_cmd_ == Command::HELP) {
            diag("Available Commands:", "\033[1;36m");
            diag("  /quit or /exit         Save to log/<N>.save and exit", "\033[37m");
            diag("  /clear                 Clear context (auto-saves first to log/<N>-clear.save)", "\033[37m");
            diag("  /undo [N]              Undo last N prompts (default: 1; auto-saves first)", "\033[37m");
            diag("  /reset                 Reset loop detector and file cache", "\033[37m");
            diag("  /reincarnate           Compose new prompt in ~/userprompt, then restart (auto-saves first)", "\033[37m");
            diag("  /continue              Resume generation after interruption", "\033[37m");
            diag("  /save [path]           Save session state (default: log/<N>.save)", "\033[37m");
            diag("  /help                  Show this help message", "\033[37m");
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
            } else if (state_.first_turn_done) {
                // Not interrupted, but user wants to keep the model going.
                // After a normal EOG the batch is empty (EOG token wasn't added),
                // so we need to feed an assistant prefill for the LLM to sample from.
                diag("Continuing generation...", "\033[1;33m");
                state_.auto_continue = true;
                state_.auto_continue_depth_val = 0;
                // Feed a minimal assistant prefill so the batch isn't empty.
                vector<llama_token> ass_prefill = common_tokenize(ctx_, g_model_tokens.assistant_turn_start.text, false, true);
                if (!ass_prefill.empty() && n_past_ + (int)ass_prefill.size() < (int)cparams_.n_ctx) {
                    feed_tokens_impl(ass_prefill);
                }
                user_input = "";
            } else {
                // No turns have happened yet -- nothing to resume. Return silently to prompt.
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
            last_user_input_ = user_input;
            if (!feed_user_message(user_input)) continue;
        }

        // 10. Generate response
        auto gen_result = generate_response();

        // 11. Process tool call
        if (process_tool_call()) {
            continue;
        }

        // 12. Log assistant output (skip if already logged via process_tool_call)
        if (!state_.auto_continue && !assistant_logged_this_turn_ && !gen_result.text.empty()) log_entry("ASSISTANT", gen_result.text);

        // 13. Handle reincarnate completion
        if (handle_reincarnate_completion()) continue;

        // Record checkpoint at every prompt return for partial restore.
        // Only save once per user turn -- clear last_user_input_ after use so that
        // subsequent tool-call iterations within the same turn don't create duplicates.
        if (!last_user_input_.empty()) {
            state_.prompt_checkpoints.push_back({n_past_, last_user_input_});
            // Save recurrent state for instant undo on hybrid models.
            llama_memory_rs_checkpoint_save(llama_get_memory(ctx_), 0);
            last_user_input_.clear();
        }
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
