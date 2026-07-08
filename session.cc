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

// From main.cc
extern std::string g_model_path;
#include <cstdlib>
#include <chrono>
#include <signal.h>
#include <cctype>
#include <set>
#include <functional>
#include <iomanip>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <termios.h>

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
extern int chatbot_mode;
extern std::ofstream tps_log;

// HOME is declared as extern std::string HOME in network.h

// --- Helper to trim leading/trailing whitespace ---
static string trim(const string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// --- Command table: single source of truth for dispatch, alias blocking, and help ---
enum class Cmd : int { NONE, QUIT, CLEAR, RESET, REINCARNATE, CONTINUE, SAVE, RESTORE, DELETE, HELP, UNDO };

enum class ArgType { NONE, PATH };

static const struct CmdInfo {
    const char* name;
    Cmd cmd;
    ArgType arg;
    const char* description;
} g_commands[] = {
    { "quit",         Cmd::QUIT,        ArgType::NONE,   "Save to log/<N>.save and exit" },
    { "exit",         Cmd::QUIT,        ArgType::NONE,   nullptr },              // alias, not shown in help
    { "clear",        Cmd::CLEAR,       ArgType::NONE,   "Clear context (auto-saves first to log/<N>-clear.save)" },
    { "undo",         Cmd::UNDO,        ArgType::NONE,   "Interactive undo: select a checkpoint to restore to" },
    { "continue",     Cmd::CONTINUE,    ArgType::NONE,   "Resume generation after interruption" },
    { "reset",        Cmd::RESET,       ArgType::NONE,   "Reset loop detector and file cache" },
    { "reincarnate",  Cmd::REINCARNATE,ArgType::NONE,   "Compose new prompt in ~/.config/lim/userprompt, then restart (auto-saves first)" },
    { "save",         Cmd::SAVE,        ArgType::PATH,   "Save session state to <path>.save (default: log/<N>.save)" },
    { "load",         Cmd::RESTORE,     ArgType::PATH,   "Load session from <path>.save (must be used after /clear)" },
    { "delete",       Cmd::DELETE,      ArgType::PATH,   "Delete <path>.save and its fast restore cache" },
    { "help",         Cmd::HELP,        ArgType::NONE,   "Show this help message" },
};

// --- Load aliases from ~/.lim_aliases ---
static map<string, string> load_aliases() {
    // Built-in commands that cannot be overridden by aliases.
    static set<string> builtin_commands;
    [[maybe_unused]] static bool init_builtin = ([](){
        for (const auto& c : g_commands) {
            if (c.name) builtin_commands.insert(c.name);
        }
        return true;
    })();

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

static void diag_speed_impl(const string& msg) {
    if (should_output_to_stdout()) {
        cout << "\033[35m[" << msg << "]\033[0m\n";
        consoleMarkNewline(true);
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
       g_auto_continue_depth_(0)
    {
        const char* cur_tty = ttyname(STDIN_FILENO);
        prev_tty_ = cur_tty ? string(cur_tty) : "";
    }

    bool run();

private:
    using Command = Cmd;

    // Parsed command and optional arguments
    Command last_cmd_ = Command::NONE;
    string save_prefix_;
    string restore_path_;
    string delete_path_;
    // Track if the previous turn was a manual save, so /quit can skip redundant auto-save
    bool prev_was_save_ = false;
    // Track if we already logged assistant output this turn (via process_tool_call),
    // so the main loop doesn't duplicate it.
    bool assistant_logged_this_turn_ = false;
    // Last non-empty user input, used as checkpoint label for tool-call turns
    string last_user_input_;
    // Readline history length right after loading .lim_history at startup.
    // Marks the end of A (persistent) entries in readline history.
    int persistent_history_len_ = 0;
    // File size of .lim_history after loading A at startup.  Used to truncate
    // back to this point at /quit before rewriting surviving C entries.
    long history_file_size_at_startup_ = 0;
    // Number of user inputs actually added to history since last restore/undo.
    // Only incremented when add_history() actually adds an entry (not skipped).
    int c_count_since_restore_ = 0;

    // --- Helper methods (extracted from lambdas) ---
    vector<llama_token> tokenize(string text) {
        return common_tokenize(ctx_, text, false, true);
    }

    void repopulate_history() {
        using_history();
        // Remove stale B entries before pushing fresh checkpoint prompts.
        int stale = history_length - persistent_history_len_;
        if (stale > 0) pop_history(stale);
        for (const auto& cp : state_.prompt_checkpoints) {
            if (!cp.prompt.empty()) {
                add_history(cp.prompt.c_str());
            }
        }
        c_count_since_restore_ = 0;
    }

    // Pop the last N entries from readline history.
    void pop_history(int n) {
        while (n > 0) {
            int len = history_length;
            if (len <= 0) break;
            remove_history(len - 1);
            n--;
        }
    }

    // Restore saved B and C entries to readline history.
    void restore_saved_history(const vector<string>& b, const vector<string>& c) {
        for (const auto& s : b) add_history(s.c_str());
        for (auto it = c.rbegin(); it != c.rend(); ++it) add_history(it->c_str());
        c_count_since_restore_ = 0;
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

    // If stdin is not a terminal (piped input), read lines directly.
    // Empty lines are skipped; EOF returns "/quit" to exit cleanly.
    if (!isatty(STDIN_FILENO)) {
        if (state_.first_turn_done && state_.last_t_count > 0) {
            diag_speed(state_.last_n_past, cparams_.n_ctx, state_.last_t_count,
                       state_.last_elapsed, state_.last_decode_time);
        }
        string line;
        while (getline(cin, line)) {
            if (!line.empty()) return line;
        }
        return "/quit";  // EOF
    }

    if (!state_.auto_continue) {
        // Print Speed from previous generation right before >>> (skip first turn).
        // Deferred here so we have all the information we need.
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

            // Ensure the diagnostic appears on its own line.
            consoleEnsureNewline();
            diag_speed_impl(speed_str + " | " + ctx_str);

            // Write to TPS log (once per turn)
            if (denom > 0) {
                tps_log << state_.last_n_past << " " << std::fixed << std::setprecision(3) << speed << "\n";
            }
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

        // Save readline's raw termios so we can restore it after Ctrl+Z / fg.
        // The shell restores cooked mode during suspend; on resume we need to
        // put the terminal back into readline's expected raw-mode state.
        struct termios saved_raw_tios{};
        tcgetattr(STDIN_FILENO, &saved_raw_tios);

        // Event loop: poll for input with select(), check for interrupts
        while (!input_complete) {
            if (g_was_interrupted) {
                break;
            }

            // After Ctrl+Z / fg, SIGCONT sets g_was_resumed. The shell has
            // restored cooked terminal mode, so restore readline's raw settings.
            if (g_was_resumed) {
                g_was_resumed = 0;
                tcsetattr(STDIN_FILENO, TCSANOW, &saved_raw_tios);
                rl_forced_update_display();
            }

            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(0, &fds);  // stdin

            // Use pselect with an empty signal mask so that SIGCONT (from fg)
            // and SIGINT (Ctrl+C) wake us immediately.  Blocks indefinitely --
            // no timeout needed since signals provide the wakeup.
            sigset_t empty_mask;
            sigemptyset(&empty_mask);

            int ret = pselect(1, &fds, nullptr, nullptr, nullptr, &empty_mask);
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
                int before = history_length;
                add_history(user_input.c_str());
                if (history_length > before) c_count_since_restore_++;
            }
        } else {
            // EOF (Ctrl+D on empty line) -- treat as interrupt/break
            g_was_interrupted = 0;
        }
    }

    // Strip leading whitespace so commands like " /quit" work.
    {
        size_t start = user_input.find_first_not_of(" \t");
        if (start != string::npos) {
            user_input = user_input.substr(start);
        } else if (!user_input.empty()) {
            // Input is entirely whitespace -- treat as empty.
            user_input.clear();
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
// Commands must be prefixed with '/'.  /save, /load, and /undo accept optional arguments.

ChatSession::Command ChatSession::handle_command(const string& input) {
    if (input.empty() || input[0] != '/') return Command::NONE;

    // Strip the leading '/'
    string rest = input.substr(1);

    for (const auto& c : g_commands) {
        int len = (int)strlen(c.name);
        bool matched = false;
        if (rest.size() == (size_t)len) {
            matched = (rest == c.name);
        } else if (rest.size() > (size_t)len && isspace(rest[len])) {
            matched = (rest.substr(0, len) == c.name);
        }
        if (!matched) continue;

        // Parse optional argument.
        string arg = trim(rest.substr(len));
        switch (c.arg) {
            case ArgType::PATH:
                save_prefix_.clear();
                restore_path_.clear();
                delete_path_.clear();
                if (c.cmd == Cmd::SAVE)    save_prefix_   = arg;
                if (c.cmd == Cmd::RESTORE) restore_path_  = arg;
                if (c.cmd == Cmd::DELETE)  delete_path_   = arg;
                return static_cast<Command>(c.cmd);

            case ArgType::NONE:
                save_prefix_.clear();
                restore_path_.clear();
                return static_cast<Command>(c.cmd);
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
        // \033[0m is an escape sequence, not a newline.
        // Don't change g_stdout_ended_with_newline - escape codes don't affect cursor position.
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
                      &state_.all_context_tokens, state_.last_feed_time);
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
    // stdout with a trailing newline, so the speed diagnostic in get_user_input()
    // will naturally appear on its own line.

    state_.last_t_count = t_count_;
    state_.last_elapsed = elapsed_ + state_.last_feed_time;
    state_.last_decode_time = gen_result_.decode_time;
    state_.last_n_past = n_past_;
    state_.first_turn_done = true;

    // Flush TPS log so data is durable after each turn
    tps_log.flush();

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

    string userprompt_path = LIM_CONFIG_DIR + "/userprompt";

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

    string follow_prompt = "Follow the prompt in " + LIM_CONFIG_DIR + "/userprompt";
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

// Prepend LIM_SAVE_DIR to relative paths (those not starting with /).
// Absolute paths are returned unchanged.
static string apply_save_dir(const string& path) {
    if (!path.empty() && path[0] == '/') return path;
    return LIM_SAVE_DIR + "/" + path;
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
                                     const vector<PromptCheckpoint>* checkpoints = nullptr,
                                     int session_num = -1) {
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
    bool ok = write_token_save_v3(path, tokens, *checkpoints, session_num);

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

    // --- Readline History Layout (A / B / C) ---
    // A = persistent history loaded from .lim_history at startup.
    // B = prompt checkpoints pushed onto readline history so up-arrow
    //     navigates through the session as it appeared when /save was called.
    // C = user inputs added since last restore/undo (survive across /clear).
    persistent_history_len_ = history_length;  // length of A
    c_count_since_restore_ = 0;

    // Record file size so we can truncate back to A at /quit time.
    {
        struct stat st;
        if (stat(history_file, &st) == 0) {
            history_file_size_at_startup_ = st.st_size;
        }
    }

    // Push checkpoint prompts (B) onto readline history so up-arrow
    // navigates through the session as it appeared when /save was called.
    repopulate_history();

    // Load user-defined aliases from ~/.lim_aliases
    aliases_ = load_aliases();

    // --- MAIN CHAT TURN LOOP ---
    while (true) {
        stop_generation = 0;
        g_was_interrupted = 0;
        assistant_logged_this_turn_ = false;

        // 1. Get user input
        string user_input = get_user_input();

        // 2. Parse and dispatch commands (all require '/' prefix)
        last_cmd_ = handle_command(user_input);

        if (last_cmd_ == Command::QUIT) {
            // Truncate .lim_history back to the end of A, then append
            // only the surviving C entries.  This removes undone-away inputs
            // while preserving crash safety (inputs were written immediately).
            {
                FILE* f = fopen(history_file, "r+");
                if (f) {
                    ftruncate(fileno(f), history_file_size_at_startup_);
                    fclose(f);
                }
                for (int i = c_count_since_restore_ - 1; i >= 0; i--) {
                    HIST_ENTRY* he = history_get(history_length - i);
                    if (he) save_history_safe(history_file, he->line);
                }
            }

            // If there's actual conversation to preserve, auto-save before exiting.
            // Skip if the user just manually saved -- nothing has changed since then.
            if (!state_.prompt_checkpoints.empty() && !prev_was_save_) {
                string autosave_path = LIM_LOG_DIR + "/" + to_string(state_.log_index) + ".save";
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path, false, nullptr, &state_.prompt_checkpoints, state_.log_index);
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
                string autosave_path = LIM_LOG_DIR + "/" + to_string(state_.log_index) + "-clear.save";
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path, false, nullptr, &state_.prompt_checkpoints, state_.log_index);
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

            // Remove B (checkpoint prompts) while preserving A (persistent)
            // and C (user inputs since last restore).
            // NOTE: history_get() uses 1-based indexing (last entry at history_length).
            {
                vector<string> saved_c;
                for (int i = 0; i < c_count_since_restore_; i++) {
                    HIST_ENTRY* he = history_get(history_length - i);
                    if (he) saved_c.push_back(he->line);
                }
                pop_history(history_length - persistent_history_len_);
                for (auto it = saved_c.rbegin(); it != saved_c.rend(); ++it) {
                    add_history(it->c_str());
                }
            }

            continue;
        }

        if (last_cmd_ == Command::UNDO) {
            // Nothing to undo.
            if (state_.prompt_checkpoints.empty()) {
                diag("No checkpoints available to undo to.", "\033[33m");
                continue;
            }

            // Auto-save before undoing so nothing is truly lost.
            {
                string autosave_path = LIM_LOG_DIR + "/" + to_string(state_.log_index) + "-clear.save";
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path, false, nullptr, &state_.prompt_checkpoints, state_.log_index);
                if (!ok) {
                    diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
                } else {
                    diag("Auto-saved to " + autosave_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[35m");
                }
            }

            // Interactive checkpoint selection, modeled on the Restore> prompt.
            size_t num_cps = state_.prompt_checkpoints.size();
            diag("Save contains " + to_string(num_cps) + " checkpoint" + (num_cps != 1 ? "s" : "") + ".", "\033[35m");
            diag("Up/down arrows to navigate, Enter to confirm, Ctrl+C to cancel.", "\033[37m");

            // Save B (checkpoint prompts) and C (user inputs) separately
            // using the known boundary between them.
            // NOTE: history_get() uses 1-based indexing.
            int b_count = history_length - c_count_since_restore_ - persistent_history_len_;
            vector<string> saved_b;
            for (int i = 0; i < b_count; i++) {
                HIST_ENTRY* he = history_get(persistent_history_len_ + 1 + i);
                if (he) saved_b.push_back(he->line);
            }
            vector<string> saved_c;
            for (int i = 0; i < c_count_since_restore_; i++) {
                HIST_ENTRY* he = history_get(history_length - i);
                if (he) saved_c.push_back(he->line);
            }

            // Pop B+C from history, leaving A (persistent) intact.
            pop_history(b_count + c_count_since_restore_);

            // Add Undo> entries oldest-to-newest so pressing Up from empty line
            // shows the most recent checkpoint first (same order as Restore>).
            int undo_entries_added = 0;
            for (const auto& cp : state_.prompt_checkpoints) {
                string label = cp.prompt.empty() ? "(empty)" : cp.prompt;
                string entry = label + " (" + to_string(cp.n_past) + " tokens)";
                add_history(entry.c_str());
                undo_entries_added++;
            }

            char* line = readline("Undo> ");
            if (stop_generation || !line) {
                // Ctrl+C or Ctrl+D -- cancel undo, restore B+C.
                if (line) free(line);
                stop_generation = 0;
                diag("Undo cancelled.", "\033[33m");
                pop_history(undo_entries_added);
                restore_saved_history(saved_b, saved_c);
                continue;
            }

            string input = line;
            free(line);

            // Match the user's selection against checkpoint display strings.
            int selected_idx = -1;
            for (int i = 0; i < (int)state_.prompt_checkpoints.size(); i++) {
                const auto& cp = state_.prompt_checkpoints[i];
                string label = cp.prompt.empty() ? "(empty)" : cp.prompt;
                string expected = label + " (" + to_string(cp.n_past) + " tokens)";
                if (input == expected) {
                    selected_idx = i;
                    break;
                }
            }

            // If nothing matched, cancel and restore B+C.
            if (selected_idx < 0) {
                diag("Undo cancelled: no matching checkpoint.", "\033[33m");
                pop_history(undo_entries_added);
                restore_saved_history(saved_b, saved_c);
                continue;
            }

            // Restore to the selected checkpoint.
            PromptCheckpoint& target = state_.prompt_checkpoints[selected_idx];

            // If the user selected the last checkpoint, it's a no-op -- restore B+C.
            if (selected_idx == (int)state_.prompt_checkpoints.size() - 1) {
                pop_history(undo_entries_added);
                restore_saved_history(saved_b, saved_c);
                continue;
            }

            // Pop Undo> entries and rebuild history from saved B+C.
            pop_history(undo_entries_added);

            // Restore B (checkpoint prompts) up to selected_idx
            // (capped at original b_count).
            int b_to_restore = std::min(selected_idx + 1, b_count);
            for (int i = 0; i < b_to_restore; i++) add_history(saved_b[i].c_str());

            // Restore C entries that were not undone away.
            // saved_c is stored newest-first.  The number of C entries to keep
            // equals the checkpoints within C range that are <= selected_idx.
            int c_to_restore = std::max(0, selected_idx + 1 - b_count);
            // These are the oldest C entries, at the end of saved_c.
            // Add them in chronological order (oldest first).
            for (int i = (int)saved_c.size() - 1; i >= (int)saved_c.size() - c_to_restore; i--) {
                add_history(saved_c[i].c_str());
            }
            c_count_since_restore_ = c_to_restore;

            diag("Restoring to: \"" + target.prompt.substr(0, min((int)target.prompt.size(), 60)) + "\" (" + to_string(target.n_past) + " tokens)", "\033[35m");

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
                int stack_idx = selected_idx - state_.checkpoint_stack_offset;

                if (stack_idx >= 0) {
                    // Restore recurrent state from the target checkpoint.
                    // No-op for pure attention models (no recurrent state).
                    llama_memory_rs_checkpoint_restore(mem, 0, (uint32_t)stack_idx);
                } else if (stack_idx == -1 && state_.checkpoint_stack_offset > 0) {
                    // Undoing back to the restore boundary: use the boundary checkpoint.
                    llama_memory_rs_checkpoint_restore(mem, 0, 0);
                }

                bool ok = llama_memory_seq_rm(mem, 0, target.n_past, -1);
                if (ok) {
                    n_past_ = target.n_past;
                    // Prune stale recurrent checkpoints beyond the restored index.
                    if (stack_idx >= 0) {
                        llama_memory_rs_checkpoint_prune(mem, 0, (uint32_t)stack_idx);
                    } else if (stack_idx == -1) {
                        // Keep only the boundary checkpoint.
                        llama_memory_rs_checkpoint_prune(mem, 0, 0);
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

            // Compute how many turns we went back (for the browser message).
            int turns_back = (int)state_.prompt_checkpoints.size() - selected_idx - 1;

            // Erase all checkpoints at and beyond the undo boundary.
            int erased_count = 0;
            for (int i = selected_idx + 1; i < (int)state_.prompt_checkpoints.size(); i++) {
                if (i < state_.checkpoint_stack_offset) erased_count++;
            }
            state_.prompt_checkpoints.erase(state_.prompt_checkpoints.begin() + selected_idx + 1, state_.prompt_checkpoints.end());
            state_.checkpoint_stack_offset = std::max(0, state_.checkpoint_stack_offset - erased_count);

            // Reset generation state so the session is ready for a fresh user prompt.
            state_.auto_continue = false;
            state_.prev_was_interrupted = false;
            reset_session_state();
            state_.last_t_count = 0;
            state_.last_elapsed = 0.0;
            state_.last_n_past = n_past_;

            log_entry("SYSTEM", "Restored to checkpoint: \"" + target.prompt.substr(0, min((int)target.prompt.size(), 60)) + "\"");

            if (should_output_to_browser()) {
                double context_percent = (n_past_ / (double)cparams_.n_ctx) * 100.0;
                string ctx_str = std::to_string(n_past_) + " (" + std::to_string((int)context_percent) + "%)";
                pipe_write(&SEG_SPEED, 1);
                string speed_msg = "Undid " + to_string(turns_back) + " | " + ctx_str;
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
                string autosave_path = LIM_LOG_DIR + "/" + to_string(state_.log_index) + "-clear.save";
                bool ok = save_session_with_header(state_.all_context_tokens, autosave_path, false, nullptr, &state_.prompt_checkpoints, state_.log_index);
                if (!ok) {
                    diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
                } else {
                    diag("Auto-saved to " + autosave_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[35m");
                }
            }

            string reincarnate_path = LIM_CONFIG_DIR + "/reincarnate";
            ifstream reincarnate_file(reincarnate_path);
            if (!reincarnate_file.is_open()) {
                diag("Reincarnate failed: Cannot open " + reincarnate_path, "\033[31m");
                continue;
            }
            stringstream reincarnate_buffer;
            reincarnate_buffer << "Use the write_file tool to write a new prompt to "
                          << LIM_CONFIG_DIR << "/userprompt. Read the following instructions and compose an appropriate prompt, then write it and return without further comment. "
                          << reincarnate_file.rdbuf();
            string reincarnate_text = reincarnate_buffer.str();
            reincarnate_file.close();

            string userprompt_check_path = LIM_CONFIG_DIR + "/userprompt";
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
            string save_path = apply_save_dir(make_save_path(save_prefix_, LIM_LOG_DIR + "/" + to_string(state_.log_index) + ".save"));

            // Named saves (with a meaningful argument) are cached in fast format
            // for instant restore. Unnamed saves are compact-only since they're
            // ephemeral and overwritten each session.
            bool write_v1 = !save_prefix_.empty();

            diag("Saving session to " + save_path + "...", "\033[35m");

            bool ok = save_session_with_header(state_.all_context_tokens, save_path, write_v1, ctx_, &state_.prompt_checkpoints, state_.log_index);
            if (!ok) {
                diag("Save failed: could not write " + save_path, "\033[31m");
            } else {
                diag("Session saved to " + save_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[32m");
                log_entry("SYSTEM", "Session saved to " + save_path);
                prev_was_save_ = true;
            }
            continue;
        }

        if (last_cmd_ == Command::RESTORE) {
            // Build restore path: require a non-empty argument (no default path).
            string rpath = restore_path_;
            if (rpath.empty()) {
                diag("/load requires a path argument. Usage: /load <save_file>", "\033[31m");
                continue;
            }

            // Append .save if not already present (matches /save and CLI behavior).
            if (rpath.size() < std::strlen(SAVE_EXT) || rpath.compare(rpath.size() - std::strlen(SAVE_EXT), std::strlen(SAVE_EXT), SAVE_EXT) != 0) {
                rpath += SAVE_EXT;
            }

            // Prepend LIM_SAVE_DIR to relative paths.
            rpath = apply_save_dir(rpath);

            // Validate the save file exists.
            struct stat st_restore;
            if (stat(rpath.c_str(), &st_restore) != 0 || !S_ISREG(st_restore.st_mode)) {
                diag("Restore failed: save file not found: " + rpath, "\033[31m");
                continue;
            }

            // Verify context is in a clean state (only system prompt present).
            int expected_n_past = (int)system_tokens_.size();
            if ((int)state_.all_context_tokens.size() != expected_n_past || !state_.prompt_checkpoints.empty()) {
                diag("Restore failed: context has changed since last /clear. Type '/clear' first, then retry.", "\033[31m");
                continue;
            }

            // Read the save file tokens.
            vector<llama_token> restored_tokens;
            if (!read_token_save(rpath, restored_tokens)) {
                diag("Restore failed: invalid save file format: " + rpath, "\033[31m");
                continue;
            }

            // Resolve to absolute path for cache key consistency.
            char abs_buf[4096];
            string restore_path_abs;
            if (realpath(rpath.c_str(), abs_buf)) {
                restore_path_abs = abs_buf;
            } else {
                restore_path_abs = rpath;
            }

            // Try instant restore from V1 cache first.
            bool cache_hit = try_load_v1_cache(restore_path_abs, restored_tokens, g_model_path, ctx_);
            int saved_session = read_save_session(rpath);
            if (cache_hit) {
                diag_restore(rpath, (int)restored_tokens.size());
                n_past_ = (int)llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) + 1;

                // Update session state.
                state_.all_context_tokens = restored_tokens;
                state_.prompt_checkpoints = read_checkpoint_offsets(rpath);
                state_.checkpoint_stack_offset = (int)state_.prompt_checkpoints.size();

                diag_session_restored(saved_session, restored_tokens.size(), (int)cparams_.n_ctx);
                log_entry("SYSTEM", "Restored session from " + rpath);
            } else {
                // Slow restore: re-decode through the model.
                diag_restore(rpath, (int)restored_tokens.size());

                // Clear the KV cache before re-decoding.
                llama_memory_clear(llama_get_memory(ctx_), true);
                n_past_ = 0;
                state_.all_context_tokens.clear();

                vector<PromptCheckpoint> restored_checkpoints = read_checkpoint_offsets(rpath);
                size_t cp_restore_idx = 0;

                auto restore_start = chrono::high_resolution_clock::now();
                bool restore_failed = false;
                for (int i = 0; i < (int)restored_tokens.size() && !restore_failed; i += (int)cparams_.n_batch) {
                    int chunk = std::min((int)cparams_.n_batch, (int)restored_tokens.size() - i);
                    batch_.n_tokens = 0;
                    for (int j = 0; j < chunk; j++) {
                        common_batch_add(batch_, restored_tokens[i + j], n_past_, {0}, (i + j == (int)restored_tokens.size() - 1));
                        n_past_++;
                    }
                    if (!handle_llama_decode_error(ctx_, batch_, "KV Cache Exhausted during restore. Type '/clear' to reset.", false)) {
                        sync_n_past(ctx_, n_past_);
                        diag("Restore failed: could not decode tokens", "\033[31m");
                        restore_failed = true;
                    }
                    // Save recurrent checkpoints at prompt boundaries.
                    while (cp_restore_idx < restored_checkpoints.size() &&
                           restored_checkpoints[cp_restore_idx].n_past <= n_past_) {
                        llama_memory_rs_checkpoint_save(llama_get_memory(ctx_), 0);
                        cp_restore_idx++;
                    }
                }
                sync_n_past(ctx_, n_past_);

                if (restore_failed) continue;

                auto restore_end = chrono::high_resolution_clock::now();
                double restore_elapsed = chrono::duration<double>(restore_end - restore_start).count();
                double restore_speed = (restore_elapsed > 0) ? restored_tokens.size() / restore_elapsed : 0;
                diag("KV cache regenerated: " + to_string(restored_tokens.size()) + " tokens at " +
                     std::to_string((int)restore_speed) + " t/s (" +
                     std::to_string((int)restore_elapsed) + "s)", "\033[35m");

                // Update session state.
                state_.all_context_tokens = restored_tokens;
                state_.prompt_checkpoints = restored_checkpoints;
                state_.checkpoint_stack_offset = 0; // all checkpoints are live

                diag_session_restored(saved_session, restored_tokens.size(), (int)cparams_.n_ctx);
                log_entry("SYSTEM", "Restored session from " + rpath);
            }


            // Check git HEAD against saved session
            {
                string saved_sha;
                FILE* fp_git = fopen(rpath.c_str(), "rb");
                if (fp_git) {
                    char hdr_buf[256];
                    if (fgets(hdr_buf, sizeof(hdr_buf), fp_git)) {
                        const char* sha_ptr = strstr(hdr_buf, "git_sha=");
                        if (sha_ptr) {
                            sha_ptr += 8;
                            while (*sha_ptr && *sha_ptr != ' ') saved_sha += *sha_ptr++;
                        }
                    }
                    fclose(fp_git);
                }

                check_git_head_on_restore(rpath, saved_sha, ctx_, batch_, n_past_, state_.all_context_tokens);
            }

            // Reset sampler state for a clean generation start.
            llama_sampler_reset(smpl_);

            // Repopulate readline history from restored checkpoints so up-arrow
            // navigates through the restored session's prompts.
            repopulate_history();

            continue;
        }

        if (last_cmd_ == Command::DELETE) {
            // Build delete path: require a non-empty argument.
            string dpath = delete_path_;
            if (dpath.empty()) {
                diag("/delete requires a path argument. Usage: /delete <save_file>", "\033[31m");
                continue;
            }

            // Append .save if not already present (matches /save and /load behavior).
            if (dpath.size() < std::strlen(SAVE_EXT) || dpath.compare(dpath.size() - std::strlen(SAVE_EXT), std::strlen(SAVE_EXT), SAVE_EXT) != 0) {
                dpath += SAVE_EXT;
            }

            // Prepend LIM_SAVE_DIR to relative paths.
            dpath = apply_save_dir(dpath);

            // Validate the save file exists.
            struct stat st_del;
            if (stat(dpath.c_str(), &st_del) != 0 || !S_ISREG(st_del.st_mode)) {
                diag("Delete failed: save file not found: " + dpath, "\033[31m");
                continue;
            }

            int cache_removed = 0;
            bool ok = delete_save_and_cache(dpath, g_model_path, &cache_removed);
            if (!ok) {
                diag("Delete failed: could not remove " + dpath, "\033[31m");
            } else {
                string msg = "Deleted " + dpath;
                if (cache_removed > 0) {
                    msg += " and " + std::to_string(cache_removed) + " cache file" + (cache_removed > 1 ? "s" : "");
                }
                diag(msg, "\033[32m");
            }
            continue;
        }

        if (last_cmd_ == Command::HELP) {
            diag("Available Commands:", "\033[1;35m");
            for (size_t i = 0; i < sizeof(g_commands) / sizeof(g_commands[0]); ++i) {
                const auto& c = g_commands[i];
                if (!c.description) continue;

                // Group consecutive aliases with the same Command value (e.g., quit/exit).
                string names = c.name;
                for (size_t j = i + 1; j < sizeof(g_commands) / sizeof(g_commands[0]); ++j) {
                    if (g_commands[j].cmd == c.cmd && !g_commands[j].description) {
                        names += " or /" + string(g_commands[j].name);
                    } else {
                        break;
                    }
                }

                // Align descriptions in a fixed-width column.
                string cmd_col = "  /" + names;

                // Append argument hint based on ArgType.
                if (c.arg == ArgType::PATH) {
                    cmd_col += " <path>";
                }

                while (cmd_col.size() < 24) cmd_col += ' ';
                diag((cmd_col + c.description).c_str(), "\033[37m");
            }
            diag("Multi-line input: Ctrl+J to insert newline, Enter to submit", "\033[1;35m");
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

        // 3. Reject unrecognized commands before they reach the LLM
        if (last_cmd_ == Command::NONE && !user_input.empty() && user_input[0] == '/') {
            diag("Unknown command: " + user_input + ". Type /help for available commands.", "\033[33m");
            continue;
        }

        // 4. Skip empty input when not auto-continuing
        if (user_input.empty() && !state_.auto_continue) continue;

        // 5. Browser/TTY setup
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

        // 6. Feed user message (if non-empty)
        if (!user_input.empty()) {
            prev_was_save_ = false;
            last_user_input_ = user_input;

            // Chatbot mode: re-decode full history each turn for comparison
            if (chatbot_mode == 1 && !state_.all_context_tokens.empty()) {
                // Mode 1: clear cache, reconstruct text, re-tokenize, re-decode everything
                // (history + new user message together, like a real chat API prefill)
                vector<llama_token> saved_history = state_.all_context_tokens;

                llama_memory_clear(llama_get_memory(ctx_), true);
                n_past_ = 0;
                state_.all_context_tokens.clear();
                state_.prompt_checkpoints.clear();
                llama_sampler_reset(smpl_);

                // Reconstruct full conversation text: history + new user message.
                // This detokenization is a LIM implementation detail (needed because LIM
                // stores tokens internally), not something a real chatbot does.
                // We exclude it from timing so the benchmark fairly emulates chatbot behavior.
                string history_text;
                for (llama_token t : saved_history) {
                    history_text += common_token_to_piece(ctx_, t);
                }

                // Build the user turn text the same way feed_user_message does,
                // so the re-tokenized prefill includes the new input.
                string turn_close_str = state_.prev_was_interrupted ? g_model_tokens.turn_end.text : "";
                state_.prev_was_interrupted = false;
                string full_text = history_text + turn_close_str;

                // Append user/assistant role markers and content via the chat template.
                full_text += build_user_assistant_turn_text(user_input);

                // Timing starts here: tokenize + decode only, matching real chatbot behavior.
                auto feed_start = chrono::high_resolution_clock::now();

                auto history = common_tokenize(ctx_, full_text, false, false);
                diag("Chatbot mode 1: re-tokenized " + to_string(history.size()) +
                     " tokens from " + to_string(saved_history.size()) +
                     " history tokens (incl. new input)", "\033[90m");
                if (!feed_tokens_impl(history)) continue;
                auto feed_end = chrono::high_resolution_clock::now();
                state_.last_feed_time = chrono::duration<double>(feed_end - feed_start).count();

                // Perform the logging/browser output that feed_user_message would do,
                // but skip its token feeding since we already included the input above.
                if (!state_.auto_continue) {
                    log_entry("USER", user_input);
                    if (should_output_to_browser() && pipe_fd >= 0) {
                        string user_html = "\n\n<div style=\"color: #79c0ff;\"><pre><code>" + html_escape(user_input) + "</code></pre></div>\n\n";
                        stream_html(user_html);
                    }
                }
                // Skip feed_user_message -- tokens already fed. Fall through to generate_response().
            } else if (chatbot_mode == 2 && !state_.all_context_tokens.empty()) {
                // Mode 2: cache-aware prefix matching (emulates llama-server behavior).
                // Build new tokens the same way mode 0 does, prepend cached tokens to
                // simulate a full request, then find the prefix match and decode only delta.
                vector<llama_token> saved_history = state_.all_context_tokens;

                string turn_close_str = state_.prev_was_interrupted ? g_model_tokens.turn_end.text : "";
                state_.prev_was_interrupted = false;

                // Timing: build new tokens + prefix match + decode delta.
                auto feed_start = chrono::high_resolution_clock::now();

                // Build just the new user turn tokens (same as feed_user_message).
                vector<llama_token> new_turn_tokens;
                if (!turn_close_str.empty()) {
                    auto close_tok = common_tokenize(ctx_, turn_close_str, false, true);
                    new_turn_tokens.insert(new_turn_tokens.end(), close_tok.begin(), close_tok.end());
                }
                auto user_ass = build_user_assistant_turn(ctx_, user_input);
                new_turn_tokens.insert(new_turn_tokens.end(), user_ass.begin(), user_ass.end());

                // Simulate full request: cached tokens + new turn tokens.
                // The prefix match is trivially the full cache since we built new tokens
                // the same way. This measures the O(N) comparison cost, not decode cost.
                vector<llama_token> full_request;
                full_request.reserve(saved_history.size() + new_turn_tokens.size());
                full_request.insert(full_request.end(), saved_history.begin(), saved_history.end());
                full_request.insert(full_request.end(), new_turn_tokens.begin(), new_turn_tokens.end());

                // Prefix match: compare against cached tokens (O(N) comparison).
                size_t match_len = 0;
                size_t min_len = std::min(saved_history.size(), full_request.size());
                for (size_t i = 0; i < min_len; i++) {
                    if (saved_history[i] == full_request[i]) {
                        match_len = i + 1;
                    } else {
                        break;
                    }
                }

                // Decode only the delta after the matched prefix.
                if ((int)match_len < (int)full_request.size()) {
                    diag("Chatbot mode 2: prefix match " + to_string(match_len) + "/" +
                         to_string(saved_history.size()) + " tokens, decoding " +
                         to_string(full_request.size() - match_len) + " new", "\033[90m");
                    vector<llama_token> delta(full_request.begin() + match_len, full_request.end());
                    if (!feed_tokens_impl(delta)) {
                        auto feed_end = chrono::high_resolution_clock::now();
                        state_.last_feed_time = chrono::duration<double>(feed_end - feed_start).count();
                        continue;
                    }
                }

                auto feed_end = chrono::high_resolution_clock::now();
                state_.last_feed_time = chrono::duration<double>(feed_end - feed_start).count();

                // Logging
                if (!state_.auto_continue) {
                    log_entry("USER", user_input);
                    if (should_output_to_browser() && pipe_fd >= 0) {
                        string user_html = "\n\n<div style=\"color: #79c0ff;\"><pre><code>" + html_escape(user_input) + "</code></pre></div>\n\n";
                        stream_html(user_html);
                    }
                }

            } else {
                state_.last_feed_time = 0.0;
            }


            // Mode 1 and mode 2 already fed the user input as part of their combined prefill.
            // Only mode 0 uses feed_user_message for the new input.
            if (chatbot_mode == 0 && !feed_user_message(user_input)) continue;
        }

        // 7. Generate response
        auto gen_result = generate_response();

        // 8. Process tool call
        if (process_tool_call()) {
            continue;
        }

        // 9. Log assistant output (skip if already logged via process_tool_call)
        if (!state_.auto_continue && !assistant_logged_this_turn_ && !gen_result.text.empty()) log_entry("ASSISTANT", gen_result.text);

        // 10. Handle reincarnate completion
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
