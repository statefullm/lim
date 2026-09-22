#include "session.h"
#include "tokens.h"
#include "mtp.h"
#include "output.h"
#include "server.h"
#include "model.h"
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
#include <ctime>
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
extern std::string g_dummy_thought_text;

// HOME is declared as extern std::string HOME in network.h

// --- Helper to trim leading/trailing whitespace ---
static string trim(const string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// --- Command table: single source of truth for dispatch, alias blocking, and help ---
enum class Cmd : int { NONE, QUIT, CLEAR, RESET, REINCARNATE, REMIND, CONTINUE, SAVE, RESTORE, DELETE, HELP, UNDO };

enum class ArgType { NONE, PATH };

static const struct CmdInfo {
    const char* name;
    Cmd cmd;
    ArgType arg;
    const char* description;
} g_commands[] = {
    { "quit",         Cmd::QUIT,        ArgType::PATH,   "Save session and exit; /quit <path> does a named save with fast cache" },
    { "exit",         Cmd::QUIT,        ArgType::PATH,   nullptr },              // alias, not shown in help
    { "clear",        Cmd::CLEAR,       ArgType::NONE,   "Clear context (auto-saves first to log/<N>-clear.save)" },
    { "undo",         Cmd::UNDO,        ArgType::NONE,   "Interactive undo: select a checkpoint to restore to" },
    { "continue",     Cmd::CONTINUE,    ArgType::NONE,   "Resume generation after interruption" },
    { "reset",        Cmd::RESET,       ArgType::NONE,   "Reset terminal, loop detector, and web search" },
    { "reincarnate",  Cmd::REINCARNATE,ArgType::NONE,   "Compose new prompt in ~/.config/lim/userprompt, then restart (auto-saves first)" },
    { "remind",       Cmd::REMIND,     ArgType::NONE,   "Re-send the full system prompt to the LLM (unescaped) so it re-anchors to its instructions" },
    { "save",         Cmd::SAVE,        ArgType::PATH,   "Save session state to <path>.save (default: log/<N>.save)" },
    { "load",         Cmd::RESTORE,     ArgType::PATH,   "Load session from <path>.save (must be used after /clear); --checkpoints skips the fast cache" },
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

// Placeholder label for the turn-end checkpoint of a turn that crossed the 90%
// context threshold (see the ctx_limit_interrupt housekeeping in run()): the
// 90% point keeps this turn's original prompt as its label, while the real
// turn end is labeled with this.  It is an /undo placeholder only -- it has no
// functional connection to the /continue command, it just looks like one.
static const char* CTX_LIMIT_TURN_END_LABEL = "/continue";

// --- Checkpoint selection prompt helpers (shared by /undo and /load) ---

// Format one checkpoint for the selection prompt:
// "<prompt truncated to 120 chars> (<n_past> tokens)".
static string checkpoint_label(const PromptCheckpoint& cp) {
    string label = cp.prompt.empty() ? "(empty)" : cp.prompt;
    if (label.size() > 120) label = label.substr(0, 120) + "...";
    return label + " (" + to_string(cp.n_past) + " tokens)";
}

// Extract the target n_past from the "(N tokens)" suffix of the user's
// selection at the selection prompt.  Since n_past is unique per checkpoint,
// this works even when prompts are truncated or duplicated.  Returns true and
// sets *n_past on success.
static bool parse_checkpoint_selection(const string& input, int* n_past) {
    size_t paren_open = input.rfind('(');
    size_t paren_close = input.rfind(')');
    if (paren_open == string::npos || paren_close <= paren_open) return false;
    try {
        *n_past = std::stoi(input.substr(paren_open + 1, paren_close - paren_open - 1));
        return true;
    } catch (...) {
        return false;
    }
}

// --- Helper to build context-limit diagnostic string ---
static string context_limit_diag(int n_past, int last_n_past, size_t needed) {
    if (n_past == last_n_past) return "";
    ostringstream oss;
    oss << " (n_past=" << n_past << " + " << needed << ", last_n_past=" << last_n_past << ")";
    return oss.str();
}

// --- One-time probe: does this tokenizer space-prefix raw text fragments? ---
// Vocabularies following the SPM convention (Llama, Qwen, etc.; the
// add_space_prefix setting inside llama.cpp) prepend a space to the first raw
// text fragment of a text and to every raw text fragment following a special
// token, so the first word token of each fragment carries a leading space
// (e.g. " system").  The public API does not expose the setting, so probe the
// tokenizer once: encoding "ab12" (which contains no space) yields a first
// token whose piece starts with a space iff the vocabulary space-prefixes.
// detokenize_tracker_to_text (the one-time reconstruction fallback of
// benchmark modes 1/2) uses this to keep the detokenize/re-tokenize round-trip
// lossless (the tokenizer would otherwise re-add the prefix on top of the
// space the detokenized pieces render).
static bool vocab_space_prefixes(const llama_vocab* vocab) {
    static bool cached = [](const llama_vocab* v) {
        vector<llama_token> toks = common_tokenize(v, "ab12", false, true);
        if (toks.empty()) return false;
        string piece = common_token_to_piece(v, toks[0], true);
        return !piece.empty() && piece.front() == ' ';
    } (vocab);
    return cached;
}

// --- Benchmark-mode canonical conversation text --------------------------
// Modes 1 (standard chatbot) and 2 (llama-server emulation) maintain
// SessionState::conversation_text -- the fully templated conversation text a
// text-based client would hold -- so each turn appends to it instead of
// detokenizing the cached KV.  Mode 0 (normal operation) never touches it.
static bool maintains_conversation_text() {
    return chatbot_mode == 1 || chatbot_mode == 2;
}

// Forward declarations for save helpers defined after the class.
static string save_diag(size_t n_checkpoints, size_t n_tokens);
static bool save_session_with_header(const vector<llama_token>& tokens, const string& path,
                                     bool write_v1, llama_context* ctx,
                                     const vector<PromptCheckpoint>* checkpoints, int session_num);

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
        const string& system_prompt_text,
        bool use_dummy_thought,
        SessionState& state
    ) : ctx_(ctx), vocab_(vocab), smpl_(smpl), batch_(batch),
       n_past_(n_past), cparams_(cparams), system_tokens_(system_tokens),
       system_prompt_text_(system_prompt_text),
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
    bool restore_checkpoints_ = false;  // /load <path> --checkpoints
    // Track if the previous turn was a manual save, so /quit can skip redundant auto-save
    bool prev_was_save_ = false;
    // Track if we already logged assistant output this turn (via process_tool_call),
    // so the main loop doesn't duplicate it.
    bool assistant_logged_this_turn_ = false;
    // Last non-empty user input, used as checkpoint label for tool-call turns
    string last_user_input_;

    // True when the current turn's prompt feed crossed the 90% threshold: the
    // turn starts at/above the line, so TokenGenerator's own crossing check
    // can never fire (it breaks on a crossing, not a start above the line).
    // The first generate_response passes it to the generator, which performs
    // the 90% mid-turn checkpoint housekeeping before generating any tokens
    // (a 0-token turn end at the prompt boundary, the checkpoint labeled with
    // the turn's prompt -- "/remind" for the /remind turn, which shares
    // feed_new_user_turn).  Consumed (cleared) by generate_response.
    bool feed_crossed_90pct_ = false;
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

    // Restore saved B and C entries to readline history (both saved oldest-first;
    // add_history appends at the newest end, so chronological order is preserved).
    // Re-count the C entries actually added so the /quit flush persists them to
    // disk after a cancelled undo (zeroing them here would drop C from .lim_history).
    void restore_saved_history(const vector<string>& b, const vector<string>& c) {
        for (const auto& s : b) add_history(s.c_str());
        c_count_since_restore_ = 0;
        for (const auto& s : c) {
            int before = history_length;
            add_history(s.c_str());
            if (history_length > before) c_count_since_restore_++;
        }
    }
    // Flush .lim_history to disk: truncate back to the persistent baseline (A),
    // then append only the surviving C entries.  Updates the tracked file size
    // so subsequent calls treat these entries as the new A.
    void flush_history(const char* history_file) {
        FILE* f = fopen(history_file, "r+");
        if (f) {
            ftruncate(fileno(f), history_file_size_at_startup_);
            fclose(f);
        }
        for (const auto& s : collect_recent_user_inputs()) {
            save_history_safe(history_file, s);
        }
        struct stat st;
        if (stat(history_file, &st) == 0) {
            history_file_size_at_startup_ = st.st_size;
        }
    }

    // Increment the session number and reopen all log files for the new
    // session (same layout/headers as startup: see open_session_logs in
    // session_utils.cc). Returns the new session number.
    int bump_session() {
        state_.log_index++;
        int idx = state_.log_index;

        chat_log.close();
        token_log.close();
        tps_log.close();

        if (!open_session_logs(idx)) {
            diag("Failed to open log files for session #" + to_string(idx), "\033[31m");
        }
        return idx;
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
            strip_tags(clean_text, tags_to_remove);            while (!clean_text.empty() && isspace(clean_text.back())) clean_text.pop_back();
            chat_log << "=== " << role << " ===\n" << clean_text << "\n\n";
            chat_log.flush();
        }
    }

    // Auto-save the current state to log/<N>-clear.save before clearing,
    // undoing, or reincarnating so nothing is truly lost.
    // Returns true on success so the caller can point the user at the file.
    bool autosave_before_clear() {
        string autosave_path = LIM_LOG_DIR + "/" + to_string(state_.log_index) + "-clear.save";
        bool ok = save_session_with_header(state_.all_context_tokens, autosave_path, false, nullptr, &state_.prompt_checkpoints, state_.log_index);
        if (!ok) {
            diag("Auto-save failed: could not write " + autosave_path, "\033[33m");
            return false;
        }
        diag("Auto-saved to " + autosave_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[35m");
        return true;
    }

    // Full post-clear reset, shared by /clear and by recovery paths that must
    // reset the context to a fresh session (e.g. an interrupted /undo
    // re-decode): clear the KV cache and re-feed the system prompt, reset the
    // session state, persist the surviving C history entries and promote them
    // into A, and bump the session number (reopens the log files, and keeps
    // this session's <N>-clear.save name from being clobbered by a later
    // /clear).  Leaves the context in exactly the state /load requires, so a
    // saved session can be restored straight from the fresh prompt.  The
    // caller performs the pre-clear auto-save.
    void clear_session(const char* history_file, const string& cleared_diag,
                       const char* color = "\033[32m") {
        clear_context();
        state_.auto_continue = false;
        state_.prev_was_interrupted = false;
        reset_session_state();
        // Reset checkpoint tracking after full context clear.
        state_.checkpoint_stack_offset = 0;
        state_.tool_correction_checkpoint_idx = -1;
        state_.rs_checkpoint_saved_this_turn = false;
        state_.last_t_count = 0;
        state_.last_elapsed = 0.0;
        state_.last_n_past = n_past_;
        state_.first_turn_done = true;
        flush_history(history_file);
        int new_log_index = bump_session();

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

        diag(cleared_diag, color);
        announce_new_session(new_log_index);
        // Remove B (checkpoint prompts) while preserving A (persistent)
        // and C (user inputs since last restore).  After flush_history above,
        // the surviving C entries are now on disk, so promote them into A.
        {
            vector<string> saved_c = collect_recent_user_inputs();  // oldest first
            pop_history(history_length - persistent_history_len_);
            for (const auto& s : saved_c) {
                add_history(s.c_str());
            }
            // Promote C into A: these entries are now persisted on disk.
            persistent_history_len_ = history_length;
            c_count_since_restore_ = 0;
        }
    }

    // Collect the C entries (user inputs since last restore/clear), oldest first.
    vector<string> collect_recent_user_inputs() {
        vector<string> saved_c;
        for (int i = c_count_since_restore_ - 1; i >= 0; i--) {
            HIST_ENTRY* he = history_get(history_length - i);
            if (he) saved_c.push_back(he->line);
        }
        return saved_c;
    }

    // Stream the user's input to the browser as a blue code block.
    void stream_user_input_html(const string& input) {
        if (should_output_to_browser() && pipe_fd >= 0) {
            string user_html = "\n\n<div style=\"color: #79c0ff;\"><pre><code>" + html_escape_for_browser(input) + "</code></pre></div>\n\n";
            stream_html(user_html);
        }
    }

    // Mid-turn iff the last tracked token is not an EOG; fresh sessions
    // (system prompt only) are never mid-turn.  Read-only.
    bool position_ends_mid_turn() {
        if (state_.all_context_tokens.size() <= system_tokens_.size()) return false;
        return !llama_vocab_is_eog(vocab_, state_.all_context_tokens.back());
    }

    // Text form of a new user turn: optional turn-end close (if the previous
    // turn was interrupted) + user turn + assistant prefill.  Takes the flag
    // explicitly so the text can be built before or after the token build
    // (build_new_user_turn_tokens consumes state_.prev_was_interrupted).
    static string new_user_turn_text(bool prev_was_interrupted, const string& input) {
        string text = prev_was_interrupted ? g_model_tokens.turn_end.text : "";
        text += build_user_assistant_turn_text(input);
        return text;
    }

    // Build tokens for a new user turn: optional turn-end close (if the
    // previous turn was interrupted) + user turn + assistant prefill.
    vector<llama_token> build_new_user_turn_tokens(const string& input) {
        string turn_close_str = (state_.prev_was_interrupted || position_ends_mid_turn()) ? g_model_tokens.turn_end.text : "";
        state_.prev_was_interrupted = false;
        vector<llama_token> tokens;
        if (!turn_close_str.empty()) {
            auto close_tok = common_tokenize(ctx_, turn_close_str, false, true);
            tokens.insert(tokens.end(), close_tok.begin(), close_tok.end());
        }
        auto user_ass = build_user_assistant_turn(ctx_, input);
        tokens.insert(tokens.end(), user_ass.begin(), user_ass.end());
        return tokens;
    }

    // --- Canonical conversation text (benchmark modes 1/2 only) -----------
    // Detokenize a cached token sequence back to conversation text, token by
    // token.  One-time reconstruction fallback for events that discard the
    // text (session start, /undo, /load, tool rollback) -- steady-state turns
    // append to the maintained text and never pay this.  For space-prefixing
    // vocabularies, the first content token of every fragment following a
    // special token (or the start of text) carries an encoding space that
    // common_token_to_piece renders as a literal; re-tokenization would
    // re-add its own prefix on top, so strip exactly one leading space from
    // those fragments to keep the round-trip lossless (mirrors is_prev_special
    // in the tokenizer).
    string detokenize_tracker_to_text(const vector<llama_token>& toks) {
        bool space_prefixes = vocab_space_prefixes(vocab_);
        string text;
        bool prev_was_special = true;  // start of text is a special boundary
        for (llama_token tok : toks) {
            bool is_special = (llama_vocab_get_attr(vocab_, tok) &
                (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) != 0;
            string piece = common_token_to_piece(vocab_, tok, true);
            if (!is_special && prev_was_special && space_prefixes &&
                !piece.empty() && piece.front() == ' ') {
                piece.erase(0, 1);
            }
            text += piece;
            prev_was_special = is_special;
        }
        return text;
    }

    // Should a one-pass re-tokenization of the canonical conversation text
    // prepend BOS?  Only when the cached stream started with BOS AND the BOS
    // detokenizes to an empty piece: if it has a visible piece, that piece is
    // already in the text and add_bos=true would double it (degenerating the
    // prefix match to 0); without add_bos the literal marker parses back to
    // the BOS token instead.  (build_system_prompt_tokens likewise leaves the
    // BOS decision to common_tokenize(add_bos=true).)
    bool should_add_bos(const vector<llama_token>& toks) {
        const llama_token bos = llama_vocab_bos(vocab_);
        if (toks.empty() || toks[0] != bos) return false;
        return common_token_to_piece(vocab_, bos, true).empty();
    }

    // New user turn text, consuming state_.prev_was_interrupted exactly like
    // build_new_user_turn_tokens (used by the mode 1/2 branches, which feed
    // the turn themselves instead of going through feed_user_message).
    string build_new_user_turn_text(const string& input) {
        bool was_interrupted = state_.prev_was_interrupted || position_ends_mid_turn();
        state_.prev_was_interrupted = false;
        return new_user_turn_text(was_interrupted, input);
    }

    // Full conversation text for a benchmark turn: the canonical text plus
    // the new turn (steady state, pure append), or a one-time detokenize
    // reconstruction of the tracker when the text was discarded (session
    // start, /undo, /load, tool rollback).
    string build_full_conversation_text(const vector<llama_token>& tracker,
                                        const string& new_turn_text) {
        if (!state_.conversation_text.empty()) {
            return state_.conversation_text + new_turn_text;
        }
        return detokenize_tracker_to_text(tracker) + new_turn_text;
    }

    // --- Rollback instrumentation (correction.md 4) -----------------------
    // Written to BOTH stderr and chat_log so the record survives a context
    // clobber. Gated on is_debug (LIM_DEBUG=1). `path` is "correction" or "undo".
    void log_rollback(const char* path, int n_past_before, long target_pos,
                      bool seq_rm_ok, int n_past_after) {
        if (!is_debug) return;
        long delta = (long)n_past_before - target_pos;
        ostringstream oss;
        oss << "[ROLLBACK path=" << path << "]\n"
            << "  n_past_before = " << n_past_before << "\n"
            << "  target_pos    = " << target_pos << "\n"
            << "  delta         = " << delta << "\n"
            << "  idx           = " << state_.tool_correction_checkpoint_idx << "\n"
            << "  stack_offset  = " << state_.checkpoint_stack_offset << "\n"
            << "  n_checkpoints = " << state_.prompt_checkpoints.size() << "\n"
            << "  all_ctx_size  = " << state_.all_context_tokens.size() << "\n"
            << "  --- after ---\n"
            << "  n_past_after  = " << n_past_after << "\n"
            << "  seq_rm_ok     = " << (seq_rm_ok ? "true" : "false") << "\n";
        string s = oss.str();
        cerr << s;
        if (chat_log.is_open()) { chat_log << s; chat_log.flush(); }
    }

    // Log a tool_correction_n_past assignment so we can see when it was last
    // set and to what value (correction.md 4). idx_was is the checkpoint index
    // at the moment of assignment.
    void log_tc_npast_set() {
        if (!is_debug) return;
        ostringstream oss;
        oss << "[TC_NPAST_SET] n_past=" << n_past_
            << " idx_was=" << state_.tool_correction_checkpoint_idx << "\n";
        string s = oss.str();
        cerr << s;
        if (chat_log.is_open()) { chat_log << s; chat_log.flush(); }
    }

    // Roll back to the tool-correction checkpoint (removes bad call + any
    // correction tokens). Returns 0 if the fast seq_rm succeeded (the slot
    // holds the R/S state for the target), 2 if the prompt-checkpoint anchor
    // suffix re-decode succeeded, -1 on failure (no anchor available or a
    // feed failed; the caller ejects to the prompt).  Never clears and
    // re-decodes the full context.
    int rollback_to_tool_checkpoint(bool eject_on_failure) {
        llama_memory_t mem = llama_get_memory(ctx_);
        int n_past_before = n_past_;
        long target_pos = state_.tool_correction_n_past;

        // --- Diagnostics: capture state before the rollback attempt (LIM_DEBUG=1) ---
        if (is_debug) {
            uint32_t rs_seq = llama_n_rs_seq(ctx_);
            llama_pos mem_max = llama_memory_seq_pos_max(mem, 0);
            std::string d = "System: [ROLLBACK-DIAG] cp_idx=" + std::to_string(state_.tool_correction_checkpoint_idx) +
                " target_n_past=" + std::to_string(state_.tool_correction_n_past) +
                " n_past=" + std::to_string(n_past_before) +
                " mem_max=" + std::to_string((long)mem_max) +
                " n_rs_seq=" + std::to_string(rs_seq) +
                " tracker=" + std::to_string((int)state_.all_context_tokens.size()) +
                " prompt_cps=" + std::to_string((int)state_.prompt_checkpoints.size()) +
                " stack_off=" + std::to_string(state_.checkpoint_stack_offset);
            if (g_mtp) {
                d += " mtp_valid=" + std::string(g_mtp->valid() ? "1" : "0");
                d += " mirror_pos=" + std::to_string(g_mtp->mirrorPos());
            }
            d += " has=" + std::string(state_.has_tool_correction_checkpoint ? "1" : "0") +
                 " saved=" + std::string(state_.rs_checkpoint_saved_this_turn ? "1" : "0");
            diag(d, "\033[36;1m");
            // Compute expected stack size from LIM-side bookkeeping and flag mismatches.
            // The tool slot holds one extra entry IFF the hook actually pushed it
            // (a lockstep FUNC_START; a non-lockstep one skips the push and the
            // stack stays one short of cp_idx -- the WARNING below catches it).
            int expected_stack = (int)state_.prompt_checkpoints.size() - state_.checkpoint_stack_offset +
                (state_.rs_checkpoint_saved_this_turn ? 1 : 0);
            if (state_.tool_correction_checkpoint_idx >= 0 &&
                state_.tool_correction_checkpoint_idx >= expected_stack) {
                diag("System: [ROLLBACK-DIAG] WARNING: cp_idx >= expected stack size " +
                     std::to_string(expected_stack) + " -- restore will no-op", "\033[31;1m");
            }
        }
        // --- End diagnostics ---

        // The rollback target must exist in the token tracker: every re-decode
        // path below slices the tracker at the target position.  A tracker
        // shorter than the target means an earlier feed failed partway (n_past_
        // was synced to the partial KV, the tracker stayed short) -- re-slicing
        // would resize-GROW the tracker with garbage default tokens and feed
        // them to the model.
        if ((int)state_.all_context_tokens.size() < (int)target_pos) {
            diag("System: Correction aborted: token tracker (" + std::to_string(state_.all_context_tokens.size()) +
                 ") is shorter than the rollback target (" + std::to_string(target_pos) +
                 "). Type '/clear' to reset.", "\033[1;31m");
            if (eject_on_failure) state_.auto_continue = false;
            return -1;
        }

        // Fast path: only when the slot holds the R/S state for the target
        // position (has_tool_correction_checkpoint: the FUNC_START was in
        // lockstep and the hook saved/overwrote the slot).  With a stale or
        // absent slot (non-lockstep FUNC_START under MTP: uncommitted verify
        // cells left the recurrent state ahead of n_past_ when the hook
        // skipped the save), a restore would apply state from the wrong
        // position (or no-op) and make seq_rm falsely succeed -- the R/S
        // state would then be inconsistent with the truncated attention KV.
        // Skip the fast path; the anchor re-decode below rebuilds the state
        // from a valid prompt checkpoint.  prune is only safe here: it must
        // never run against a short stack (resize would grow it with default
        // entries), and has_tool_correction_checkpoint guarantees the slot
        // exists.
        bool rm_ok = false;
        if (state_.has_tool_correction_checkpoint) {
            llama_memory_rs_checkpoint_restore(mem, 0, (uint32_t)state_.tool_correction_checkpoint_idx);
            llama_memory_rs_checkpoint_prune(mem, 0, (uint32_t)state_.tool_correction_checkpoint_idx);
            rm_ok = llama_memory_seq_rm(mem, 0, state_.tool_correction_n_past, -1);
        }
        if (rm_ok) {
            n_past_ = state_.tool_correction_n_past;
            state_.all_context_tokens.resize(state_.tool_correction_n_past);
            // Tracker rewound: the canonical conversation text no longer
            // describes it -- invalidate (benchmark modes 1/2 rebuild from the
            // tracker on the next turn).
            state_.conversation_text.clear();
            // MTP stays enabled: the mirror is left ahead of the
            // truncated main context and self-heals in process() -- the
            // injected clean call is decoded right after the rollback,
            // at positions behind the mirror's stale max, so its
            // position guard seq_rm's the mirror back to that position
            // and re-mirrors with fresh hidden states.  Rows before the
            // rollback point are untouched and remain valid in the
            // mirror.  Only pending_h_ is stale: it feeds the embd of
            // exactly one re-mirrored row (the first after the heal
            // point) -- negligible for acceptance, irrelevant for
            // correctness (verify re-samples every committed token from
            // the main model); the same decode refreshes it.
            log_rollback("correction", n_past_before, target_pos, true, n_past_);
            return 0;
        }
        if (state_.has_tool_correction_checkpoint) {
            // Slot valid but seq_rm failed: fall back to the anchor re-decode.
            diag("System: Correction rollback failed, re-decoding...", "\033[33m");
        } else {
            // Slot stale/absent: the fast path was skipped above; rebuild the
            // R/S state from the newest valid prompt checkpoint.
            diag("System: Tool-correction slot unavailable (non-lockstep FUNC_START); re-decoding suffix from prompt checkpoint.", "\033[33m");
        }

        // Fast fallback: anchor at the newest prompt checkpoint at or before
        // the target.  Prompt checkpoints are always saved at lockstep prompt
        // boundaries, so their R/S state is valid -- restore it, roll back
        // there, and re-decode only the short suffix.  Much cheaper than a
        // full clear + re-decode of the whole context, and the mirror
        // self-heals via the position guard during the suffix decode (MTP
        // stays enabled, like the fast path).
        int anchor_cp = -1;
        for (int i = (int)state_.prompt_checkpoints.size() - 1; i >= 0; i--) {
            if (state_.prompt_checkpoints[i].n_past <= state_.tool_correction_n_past &&
                i - state_.checkpoint_stack_offset >= 0) {
                anchor_cp = i;
                break;
            }
        }
        if (anchor_cp >= 0) {
            const int anchor_pos = state_.prompt_checkpoints[anchor_cp].n_past;
            llama_memory_rs_checkpoint_restore(mem, 0,
                (uint32_t)(anchor_cp - state_.checkpoint_stack_offset));
            if (llama_memory_seq_rm(mem, 0, (llama_pos)anchor_pos, -1)) {
                diag("Correction: re-decoding " + std::to_string((int)(target_pos - anchor_pos)) +
                     " tokens from prompt checkpoint", "\033[35m");
                n_past_ = anchor_pos;
                state_.all_context_tokens.resize((size_t)target_pos);
                vector<llama_token> suffix(
                    state_.all_context_tokens.begin() + anchor_pos,
                    state_.all_context_tokens.end());
                state_.all_context_tokens.resize((size_t)anchor_pos);
                if (feed_tokens_impl(suffix)) {
                    // Tracker rewound (anchor + re-decoded suffix): invalidate
                    // the canonical conversation text, same as the fast path.
                    state_.conversation_text.clear();
                    log_rollback("correction", n_past_before, target_pos, false, n_past_);
                    return 2;
                }
                // Suffix feed failed (interrupt or decode error): n_past_ was
                // synced to the partial KV, which holds [0, anchor) plus the
                // partially fed suffix tokens.  Re-attach exactly those
                // partial tokens to the tracker (keeping tracker == KV), then
                // fall through to the abort below -- repairing further would
                // require a full-context re-decode, which this path never
                // does.  /clear recovers.
                state_.all_context_tokens.insert(state_.all_context_tokens.end(),
                    suffix.begin(), suffix.begin() + (size_t)(n_past_ - anchor_pos));
            }
        }

        // No anchor (or the suffix re-decode above failed): eject rather than
        // clear and re-decode the full context -- a slow decode of the whole
        // context is worse than ending the turn.  The context is left as-is
        // (bad call still in place, or truncated to the anchor with a partial
        // suffix); the LLM sees it on the next turn and /clear recovers fully.
        diag("System: Correction aborted: no prompt checkpoint available for suffix re-decode. Type '/clear' to reset.", "\033[1;31m");
        log_rollback("correction", n_past_before, target_pos, false, n_past_);
        if (eject_on_failure) state_.auto_continue = false;
        return -1;
    }

    // Progress in the browser status bar (same "t/s | pos (pct)" format as
    // generation): the position counter is the token progress.  Visible
    // feedback for long feeds (slow-restore/undo re-decode); no-op when
    // browser output is off.  start/n_past_start describe the feed this
    // progress belongs to (restore feeds start at 0, mid-session feeds at
    // the feed's starting position).
    void stream_feed_progress(chrono::high_resolution_clock::time_point start, int n_past_start) {
        if (!should_output_to_browser()) return;
        double el = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
        int tps = (el > 0) ? round_int((n_past_ - n_past_start) / el) : 0;
        stream_speed(format_speed_ctx(tps, n_past_, (int)cparams_.n_ctx));
    }

    bool feed_tokens_impl(const vector<llama_token>& toks) {
        batch_.n_tokens = 0;
        auto feed_start = chrono::high_resolution_clock::now();
        int n_past_feed_start = n_past_;
        for (size_t i = 0; i < (int)toks.size(); i++) {
            if (stop_generation) return false;
            common_batch_add(batch_, toks[i], n_past_++, {0}, (i == (int)toks.size() - 1));
            if (batch_.n_tokens == (int)cparams_.n_batch && i != (int)toks.size() - 1) {
                if (!handle_llama_decode_error(ctx_, batch_)) { sync_n_past(ctx_, n_past_); return false; }
                batch_.n_tokens = 0;
                stream_feed_progress(feed_start, n_past_feed_start);
            }
        }
        if (batch_.n_tokens > 0) {
            if (!handle_llama_decode_error(ctx_, batch_, "KV Cache Exhausted. Type '/clear' to reset.", false)) {
                sync_n_past(ctx_, n_past_);
                return false;
            }
            sync_n_past(ctx_, n_past_);
            stream_feed_progress(feed_start, n_past_feed_start);
        }
        // Track all tokens fed into context for save/restore
        state_.all_context_tokens.insert(state_.all_context_tokens.end(), toks.begin(), toks.end());
        return true;
    }

    // Reload the system prompt from disk (prompt file + localprompt + cwd +
    // date/time + escape contract, when enabled) so /clear re-feeds exactly
    // what startup did.
    // Returns true on success, false if the prompt file is missing (old tokens AND
    // old text kept -- the pair must stay in lockstep for conversation_text).
    bool reload_system_prompt() {
        string system_prompt;
        if (!load_system_prompt_text(system_prompt)) return false;  // Keep old pair.
        system_tokens_ = build_system_prompt_tokens(ctx_, system_prompt);
        system_prompt_text_ = system_prompt;
        return true;
    }

    void clear_context() {
        llama_memory_clear(llama_get_memory(ctx_), true);
        // Reset the MTP mirror with the main cache: the re-fed system prompt
        // rebuilds the mirror via the mirror hook, re-arming the speculator.
        if (g_mtp) g_mtp->clear();
        n_past_ = 0;
        // Reset context token tracker to empty; feed_tokens_impl will rebuild it.
        state_.all_context_tokens.clear();
        state_.prompt_checkpoints.clear();
        state_.file_cache.clear();

        // Reset the current directory to the initial value so no memory of the last session persists.
        // Done before reload_system_prompt() so getcwd() reflects the initial directory.
        {
            chdir(INITIAL_CWD.c_str());
            ofstream cwd_file(HOME + "/.cwd");
            if (cwd_file.is_open()) {
                cwd_file << INITIAL_CWD << endl;
                cwd_file.close();
            }
        }

        // Reload system prompt from disk to pick up any edits + fresh timestamp.
        if (!reload_system_prompt()) {
            diag("Prompt file not found; reusing cached system prompt.", "\033[33m");
        }
        feed_tokens_impl(system_tokens_);

        // Benchmark modes 1/2: re-seed the canonical conversation text to
        // exactly what was just fed (an empty prompt means no system turn,
        // which is the empty/invalid text).
        if (maintains_conversation_text()) {
            state_.conversation_text = build_system_turn_text(system_prompt_text_);
        }

        // Reset sampler state (penalty history, RNG) for a fresh start
        llama_sampler_reset(smpl_);
    }

    void reset_session_state() {
        state_.correction_attempted_this_turn = false;
        NetworkTools().reset_search();
        NetworkTools::reset_context_usage();
        g_browser_warning_suppressed = false;

        state_.partial_tool_text.clear();
        state_.tool_interrupt_pending = false;
        state_.thinking_block_open = false;
        state_.interrupted_checkpoint_idx = -1;
        state_.rs_checkpoint_saved_this_turn = false;
    }

    // --- Main loop methods ---
    string get_user_input();
    Command handle_command(const string& input);
    bool feed_new_user_turn(const string& input);
    bool feed_user_message(const string& input);
    TokenGenerator::Result generate_response(bool is_correction_gen = false);
    bool process_tool_call();
    bool handle_reincarnate_completion();

    // --- Member variables ---
    llama_context* ctx_;
    const llama_vocab* vocab_;
    llama_sampler* smpl_;
    llama_batch& batch_;
    int& n_past_;
    const llama_context_params& cparams_;
    vector<llama_token> system_tokens_;
    // The text system_tokens_ was built from (build_system_prompt_tokens).
    // Kept in lockstep with the tokens so benchmark modes 1/2 can seed
    // state_.conversation_text after a clear/restore without re-reading disk
    // (and so "prompt file not found, reusing cached prompt" keeps the old
    // pair instead of mixing new text with old tokens).
    string system_prompt_text_;
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
            // Ensure the diagnostic appears on its own line.
            consoleEnsureNewline();
            diag_speed(state_.last_n_past, (int)cparams_.n_ctx, state_.last_t_count,
                       state_.last_elapsed, state_.last_decode_time, true);
        } else if (!state_.first_turn_done && should_output_to_browser()) {
            // First turn: show context position while user types their prompt.
            double context_percent = (n_past_ / (double)cparams_.n_ctx) * 100.0;
            ostringstream oss;
            oss << n_past_ << " (" << (int)context_percent << "%)";
            stream_speed(oss.str());
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

// Match 'rest' (input after '/') against the command table: the exact command
// name, or the name followed by whitespace separating it from its argument.
// Returns the matching table entry, or nullptr if no command name matches.
static const CmdInfo* match_command(const string& rest) {
    for (const auto& c : g_commands) {
        if (!c.name) continue;
        int len = (int)strlen(c.name);
        if (rest.size() == (size_t)len) {
            if (rest == c.name) return &c;
        } else if (rest.size() > (size_t)len && isspace(rest[len])) {
            if (rest.substr(0, len) == c.name) return &c;
        }
    }
    return nullptr;
}

ChatSession::Command ChatSession::handle_command(const string& input) {
    if (input.empty() || input[0] != '/') return Command::NONE;

    // Strip the leading '/'
    string rest = input.substr(1);

    const CmdInfo* c = match_command(rest);
    if (!c) return Command::NONE;

    // Parse optional argument.
    string arg = trim(rest.substr((int)strlen(c->name)));
    switch (c->arg) {
        case ArgType::PATH:
            save_prefix_.clear();
            restore_path_.clear();
            delete_path_.clear();
            restore_checkpoints_ = false;
            if (c->cmd == Cmd::SAVE)    save_prefix_   = arg;
            if (c->cmd == Cmd::RESTORE) {
                // Optional trailing flag: "/load <path> --checkpoints"
                restore_checkpoints_ = strip_checkpoints_flag(arg);
                restore_path_ = arg;
            }
            if (c->cmd == Cmd::DELETE)  delete_path_   = arg;
            if (c->cmd == Cmd::QUIT)    save_prefix_   = arg;
            return static_cast<Command>(c->cmd);

        case ArgType::NONE:
            save_prefix_.clear();
            restore_path_.clear();
            if (!arg.empty()) {
                diag("/" + string(c->name) + " does not accept arguments", "\033[31m");
                return Command::NONE;
            }
            return static_cast<Command>(c->cmd);
    }
    return Command::NONE;
}

// --- feed_new_user_turn: build and feed the tokens of a new user turn ---
// Shared by feed_user_message (a typed prompt) and /remind (the re-sent
// system prompt): optional turn-end close (the previous turn was interrupted)
// + user turn + assistant prefill + dummy thought stub when thinking is
// suppressed (LIM_THINKING=0).  Keeps the canonical conversation text
// (benchmark modes 1/2) in lockstep.  Returns false if the turn does not fit
// in the context or the feed fails -- in both cases nothing is fed.
bool ChatSession::feed_new_user_turn(const string& input) {
    // Build user turn + assistant prefill using model-type-aware token vectors.
    // Read the close decision BEFORE build_new_user_turn_tokens: it consumes
    // state_.prev_was_interrupted while tokenizing the turn-close, and the
    // text form below needs it.  position_ends_mid_turn() is read-only.
    bool turn_was_closed = state_.prev_was_interrupted || position_ends_mid_turn();
    vector<llama_token> tokens = build_new_user_turn_tokens(input);

    // Benchmark modes 1/2: text form of exactly what the feed below adds, so
    // the canonical conversation text stays in lockstep with the tracker.
    string turn_text;
    if (maintains_conversation_text() && !state_.conversation_text.empty()) {
        turn_text = new_user_turn_text(turn_was_closed, input);
    }

    // If using dummy thought, append the thinking block as content tokens.
    if (use_dummy_thought_) {
        string think_block = g_model_tokens.think_start + "\n" + g_dummy_thought_text + "\n" + g_model_tokens.think_end + "\n";
        auto think_tok = common_tokenize(ctx_, think_block, false, true);
        tokens.insert(tokens.end(), think_tok.begin(), think_tok.end());
        if (!turn_text.empty()) turn_text += think_block;
    }

    const int n_before = n_past_;
    if (n_past_ + (int)tokens.size() >= (int)cparams_.n_ctx) {
        string ctx_diag = context_limit_diag(n_past_, state_.last_n_past, tokens.size());
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

    // The feed succeeded: keep the canonical text in lockstep (only when it
    // is currently valid -- an empty text is rebuilt from the tracker by the
    // next mode 1/2 turn instead of being appended to stale state).
    if (!turn_text.empty()) state_.conversation_text += turn_text;

    // 90% line crossed by the prompt feed itself: the turn starts at/above
    // the line, where the generator's crossing check (which breaks on a
    // crossing, never on a start above the line) would skip this turn's
    // checkpoint.  Record the crossing; the first generate_response of this
    // turn performs the checkpoint housekeeping before its first token (a
    // 0-token turn end at the prompt boundary labeled "/continue"; the
    // /remind turn shares this path: boundary checkpoint "/continue", end
    // checkpoint "/remind").  The crossing is recorded ONLY on the actual
    // below->above transition: a session crosses the line at most once
    // (context only grows within a session, and /undo below the line
    // discards the checkpoints above the restore point), so there is never
    // more than one "/continue" checkpoint.  A feed that continues at/above
    // an already-crossed line does not re-record: the turn takes the
    // warn-only path and gets its single turn-end checkpoint as usual.
    if (n_before < (int)(cparams_.n_ctx * 0.9) && n_past_ >= (int)(cparams_.n_ctx * 0.9)) {
        feed_crossed_90pct_ = true;
    }

    // Log user input tokens to token_log when debug is enabled
    log_tokens("FEED USER_INPUT", tokens, ctx_);
    return true;
}

// --- feed_user_message: construct and feed user message tokens ---
bool ChatSession::feed_user_message(const string& input) {
    // If user provides regular input (not "continue"), clear any pending tool interrupt state.
    if (!state_.auto_continue) state_.tool_interrupt_pending = false;
    if (!state_.auto_continue) state_.partial_tool_text.clear();  // abandoned partial would otherwise be prepended to the next executed call
    if (!state_.auto_continue) state_.thinking_block_open = false;
    if (!state_.auto_continue) {
        log_entry("USER", input);
        stream_user_input_html(input);
    }
    return feed_new_user_turn(input);
}

// --- generate_response: invoke TokenGenerator and update state ---
TokenGenerator::Result ChatSession::generate_response(bool is_correction_gen) {
    // Reset terminal color to default before LLM text starts printing.
    // Readline draws the >>> prompt in cyan; without this reset, all LLM output
    // would appear in cyan.
    if (should_output_to_stdout()) {
        cout << "\033[0m";
        cout.flush();
        // \033[0m is an escape sequence, not a newline.
        // Don't change g_stdout_ended_with_newline - escape codes don't affect cursor position.
    }

    if (!state_.auto_continue) g_auto_continue_depth_ = 0;

    // Reset sampler between user turns so the penalties ring buffer starts
    // fresh.  Only generated tokens are tracked (user input goes through
    // feed_tokens_impl, not the sampler chain), so repetition penalties
    // apply only to what the model itself produces this turn.
    if (!state_.auto_continue) {
        llama_sampler_reset(smpl_);
    }

    // Compute turn timeout from environment
    static constexpr double DEFAULT_TURN_TIMEOUT_SEC = 3600.0;
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

    // When resuming mid-think (interrupted inside an open thinking block), the
    // generator starts in thinking mode: the opening tag is already in the
    // context, and the viewer is reminded via the SEG_THINK of the first chunk.
    bool was_mid_thinking_block = state_.thinking_block_open;
    state_.thinking_block_open = false;

    // 90% crossed during this turn's prompt feed: the generator performs the
    // mid-turn checkpoint housekeeping before its first token.
    bool feed_crossed_90pct = feed_crossed_90pct_;
    feed_crossed_90pct_ = false;

    // The turn's recurrent checkpoint slot is saved lazily by the on_tool_start
    // hook (passed to TokenGenerator below), right after the FUNC_START token is
    // fed -- the only point where "right before the tool-call body" is reachable.
    // If a generation produces no tool call, no slot activity happens; the
    // end-of-turn block pushes one if the whole turn used none.
    // Per-turn flags reset only on the first generate_response() of a turn;
    // subsequent calls (auto-continue, correction cycle) preserve the existing
    // checkpoint index so rollback still works.
    if (!is_correction_gen && state_.tool_correction_checkpoint_idx < 0) {
        state_.has_tool_correction_checkpoint = false;
        state_.correction_attempted_this_turn = false;
    }

    auto start = chrono::high_resolution_clock::now();


    // --- TOKEN GENERATION via TokenGenerator class ---
    // The on_tool_start hook fires right after the FUNC_START token is fed+decoded:
    // it (re)saves the turn's recurrent checkpoint slot at that exact position and
    // sets tool_correction_n_past to match, so a later correction rollback restores
    // recurrent state from exactly where seq_rm truncates the attention cache.
    // The correction's own regeneration (is_correction_gen) gets no hook: it must
    // not touch the slot or the target, or a second correction attempt on the same
    // call would roll back incorrectly.
    std::function<void(bool lockstep)> on_tool_start;
    if (!is_correction_gen) {
        on_tool_start = [this](bool lockstep) {
            llama_memory_t mem = llama_get_memory(ctx_);
            if (state_.tool_correction_checkpoint_idx < 0) {
                // First FUNC_START of this turn: create the slot.
                state_.tool_correction_checkpoint_idx =
                    (int)state_.prompt_checkpoints.size() - state_.checkpoint_stack_offset;
            }
            if (lockstep) {
                // R/S state is at n_past_: safe to save/overwrite the checkpoint.
                if (state_.rs_checkpoint_saved_this_turn) {
                    llama_memory_rs_checkpoint_overwrite(mem, 0,
                        (uint32_t)state_.tool_correction_checkpoint_idx);
                } else {
                    llama_memory_rs_checkpoint_save(mem, 0);
                    state_.rs_checkpoint_saved_this_turn = true;
                }
                state_.has_tool_correction_checkpoint = true;
            } else {
                // R/S state is ahead of n_past_ (uncommitted verify cells):
                // a checkpoint from an earlier position would be stale for
                // this target -- mark the fast rollback path as unavailable
                // (the correction rollback re-decodes the suffix from the
                // newest prompt checkpoint instead; never a full-context
                // re-decode).
                state_.has_tool_correction_checkpoint = false;
            }
            // Always update the rollback target: n_past_ is correct
            // regardless of whether the R/S state is in lockstep.
            state_.tool_correction_n_past = n_past_;
            log_tc_npast_set();
        };
    }
    TokenGenerator tg(ctx_, vocab_, smpl_, batch_, n_past_, cparams_,
                      turn_timeout_sec, was_mid_tool_call_, state_.last_n_past,
                      &state_.all_context_tokens, state_.last_feed_time,
                      state_.reincarnate_mode, on_tool_start,
                      was_mid_thinking_block, feed_crossed_90pct);
    gen_result_ = tg.generate();

    // Signal the viewer that generation is complete so it can render
    // remaining raw tails (no arbitrary timeout needed).
    pipe_write(&SEG_TURN_END, 1);

    generated_text_ = gen_result_.text;
    t_count_ = gen_result_.token_count;

    // Benchmark modes 1/2: the generated tokens just landed in the tracker;
    // append their streamed text (the same text a text-based chatbot or
    // llama-server's client would send back on the next turn).  Its
    // re-tokenization may drift from the sampled tokens -- that is the real
    // drift the prefix match measures.  Interrupted generations append their
    // partial text, which matches the partial tracker.
    // A non-recovered EOG termination feeds the EOG token into the tracker
    // WITHOUT appending its piece to the streamed text (the break precedes
    // the detokenize block in TokenGenerator): mirror that piece here, or the
    // next turn's re-tokenization comes up one token short at every turn
    // boundary -- a systematic "drift" the old detokenize path reproduced
    // for free.
    if (maintains_conversation_text() && !state_.conversation_text.empty()) {
        string turn_text = gen_result_.text;
        if (gen_result_.ended_on_eog) {
            turn_text += common_token_to_piece(vocab_, llama_vocab_eot(vocab_), true);
        }
        state_.conversation_text += turn_text;
    }

    // Handle mid-tool-call state saving (regardless of exit reason).
    // A silent-loop abort (stuck_in_tool_call) is excluded: the correction cycle
    // rolls back to right after FUNC_START, so no resume state is needed -- and
    // any stale partial text from a previous interrupted generation would corrupt
    // ToolExecutor's extraction of the injected call.
    if (gen_result_.stuck_in_tool_call) {
        state_.partial_tool_text.clear();
    } else if (gen_result_.tool_start != string::npos && gen_result_.tool_end == string::npos) {
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
        // Interrupted inside an open thinking block: /continue must resume in
        // thinking mode (see the consume at the top of this function).
        state_.thinking_block_open = gen_result_.was_in_thinking_block;
    } else if (gen_result_.early_exit) {
        state_.auto_continue = false;
        state_.reincarnate_mode = false;
    } else if (!gen_result_.has_tool_call) {
        // Normal EOG
        if (state_.reincarnate_first_turn) {
            // LLM hit EOG prematurely on the first turn after reincarnate.
            // Auto-continue so it can produce a full response rather than
            // dropping to the prompt and requiring the user to type /continue.
            state_.reincarnate_first_turn = false;
            // Keep auto_continue = true so generation resumes immediately.
        } else {
            state_.auto_continue = false;
        }
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

    // Per-turn MTP acceptance stats (benchmark: compare against mode 0 runs).
    if (g_mtp) {
        std::string mtp_stats;
        if (g_mtp->stats_line(mtp_stats)) {
            if (tps_log.is_open()) {
                tps_log << mtp_stats;
            }
            if (is_debug) {
                diag(std::string("MTP this turn: ") + mtp_stats, "\033[90m");
            }
            g_mtp->reset_stats();
        }
    }

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
        // The turn's recurrent checkpoint was already saved by the on_tool_start
        // hook right after FUNC_START was fed -- do NOT touch the checkpoint
        // bookkeeping here: by the time process_tool_call runs, n_past is already
        // past the call, so saving now would desync the rollback (checkpoint
        // position != tool_correction_n_past).  Likewise do NOT overwrite
        // has_tool_correction_checkpoint: the hook already set it to the slot's
        // actual state (a non-lockstep FUNC_START skips the push), and forcing
        // true here would hide that and let the correction path roll back to a
        // slot that was never pushed.

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
            g_auto_continue_depth_, max_auto_continue_
        );

        if (tool_result.should_auto_continue) {
            return true; // Signal to continue outer loop
        }

        // Tool-call correction: feed system prompt reminder, generate once,
        // parse for a valid tool call, roll back to the tool-correction
        // checkpoint (fast slot path, or the prompt-anchor suffix re-decode
        // when the slot is stale), and inject the good tool call + its output
        // cleanly.
        if (tool_result.needs_correction) {
            state_.tool_correction_mode = true;
            // Return false so we fall through to step 8b in the same iteration,
            // where the correction cycle is handled before generating again.
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
    const char* history_file = ".lim_history";
    flush_history(history_file);
    int new_log_index = bump_session();
    announce_new_session(new_log_index);

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
        string ctx_diag = context_limit_diag(n_past_, state_.last_n_past, new_session_tokens.size());
        diag("Context Limit Reached! Cannot process reincarnated prompt" + ctx_diag + ".", "\033[31m");
        return true; // continue outer loop
    }

    if (!feed_tokens_impl(new_session_tokens)) {
        if (stop_generation) stop_generation = 0;
        diag("Failed to feed reincarnated session tokens. Type '/clear' to reset.", "\033[31m");
        return true; // continue outer loop
    }

    // Keep the canonical conversation text (benchmark modes 1/2) in lockstep:
    // clear_context() above re-seeded it with the system turn; append this
    // turn's text form (new_session_tokens was built by build_user_assistant_turn).
    if (maintains_conversation_text() && !state_.conversation_text.empty()) {
        state_.conversation_text += build_user_assistant_turn_text(follow_prompt);
    }

    // Log reincarnated session tokens to token_log when debug is enabled
    log_tokens("FEED USER_INPUT", new_session_tokens, ctx_);

    // Record checkpoint for the reincarnated session prompt so saves and restores work.
    state_.prompt_checkpoints.push_back({n_past_, follow_prompt});
    // Save recurrent state for instant undo on hybrid models.
    llama_memory_rs_checkpoint_save(llama_get_memory(ctx_), 0);

    // Reset checkpoint tracking after context clear + new checkpoint.
    state_.checkpoint_stack_offset = 0;
    state_.tool_correction_checkpoint_idx = -1;
    state_.interrupted_checkpoint_idx = -1;
    state_.rs_checkpoint_saved_this_turn = false;

    state_.auto_continue = true;
    state_.reincarnate_first_turn = true;
    reset_session_state();
    log_entry("SYSTEM", "Context Cleared and Reincarnated with New Prompt");
    return true; // continue outer loop
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
        // Persist the MTP mirror KV alongside the main KV so a future fast
        // restore can bring MTP back with it (only when the mirror is currently
        // consistent with the main context; an invalidated mirror is skipped
        // and the next fast restore just falls back to disabling MTP).
        write_v1_cache(abs_path, tokens, g_model_path, ctx, old_hash,
                       (g_mtp && g_mtp->valid()) ? g_mtp->draft_ctx() : nullptr);
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

    // CLI restore: inject "/load <arg>" as the first input so the normal
    // /load handler runs -- identical to in-session /clear + /load.
    string pending_input;
    if (!state_.cli_restore_path.empty()) {
        pending_input = "/load " + state_.cli_restore_path;
        state_.cli_restore_path.clear();
    }

    // --- MAIN CHAT TURN LOOP ---
    while (true) {
        stop_generation = 0;
        g_was_interrupted = 0;
        assistant_logged_this_turn_ = false;

        // 1. Get user input (or the injected CLI restore command)
        string user_input;
        if (!pending_input.empty()) {
            user_input = pending_input;
            pending_input.clear();
        } else {
            user_input = get_user_input();
        }

        // 2. Parse and dispatch commands (all require '/' prefix)
        last_cmd_ = handle_command(user_input);

        if (last_cmd_ == Command::QUIT) {
            // Truncate .lim_history back to the end of A, then append
            // only the surviving C entries.  This removes undone-away inputs
            // while preserving crash safety (inputs were written immediately).
            flush_history(history_file);

            // If there's actual conversation to preserve, auto-save before exiting.
            // Skip if the user just manually saved -- nothing has changed since then.
            if (!state_.prompt_checkpoints.empty() && !prev_was_save_) {
                string save_path;
                bool write_v1 = false;                if (!save_prefix_.empty()) {
                    // Named save: use fast cache for instant future restores.
                    save_path = apply_save_dir(append_save_ext(save_prefix_));
                    write_v1 = true;
                } else {
                    save_path = LIM_LOG_DIR + "/" + to_string(state_.log_index) + SAVE_EXT;
                }
                bool ok = save_session_with_header(state_.all_context_tokens, save_path, write_v1, ctx_, &state_.prompt_checkpoints, state_.log_index);
                if (!ok) {
                    diag("Auto-save failed: could not write " + save_path, "\033[33m");
                } else {
                    diag("Auto-saved to " + save_path + " (" + save_diag(state_.prompt_checkpoints.size(), state_.all_context_tokens.size()) + ")", "\033[35m");
                }
            }

            return false;
        }
        if (last_cmd_ == Command::CLEAR) {
            // Auto-save before clearing so nothing is truly lost.
            // Uses a distinct name (e.g., log/5-clear.save) so it doesn't conflict
            // with the regular save file that /quit or /exit will overwrite.
            autosave_before_clear();
            clear_session(history_file, "Context Cleared Successfully");
            continue;
        }

        if (last_cmd_ == Command::UNDO) {
            // Nothing to undo.
            if (state_.prompt_checkpoints.empty()) {
                diag("No checkpoints available to undo to.", "\033[33m");
                continue;
            }
            // Auto-save before undoing so nothing is truly lost.
            bool undo_autosave_ok = autosave_before_clear();
            string pre_undo_save = LIM_LOG_DIR + "/" + to_string(state_.log_index) + "-clear.save";

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
            vector<string> saved_c = collect_recent_user_inputs();  // oldest first

            // Pop B+C from history, leaving A (persistent) intact.
            pop_history(b_count + c_count_since_restore_);

            // Add Undo> entries oldest-to-newest so pressing Up from empty line
            // shows the most recent checkpoint first (same order as Restore>).
            for (const auto& cp : state_.prompt_checkpoints) {
                add_history(checkpoint_label(cp).c_str());
            }
            int undo_entries_added = (int)state_.prompt_checkpoints.size();

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

            // Treat /quit and /exit at the Undo> prompt as a graceful cancel,
            // not as a failed checkpoint match.
            if (input == "/quit" || input == "/exit") {
                diag("Undo cancelled.", "\033[33m");
                pop_history(undo_entries_added);
                restore_saved_history(saved_b, saved_c);
                continue;
            }

            // Match the user's selection by the "(N tokens)" suffix (n_past is
            // unique per checkpoint, so this works even for truncated or
            // duplicated prompts).
            int selected_idx = -1;
            int target_n_past = -1;
            if (parse_checkpoint_selection(input, &target_n_past)) {
                for (int i = 0; i < (int)state_.prompt_checkpoints.size(); i++) {
                    if (state_.prompt_checkpoints[i].n_past == target_n_past) {
                        selected_idx = i;
                        break;
                    }
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
            // saved_c is stored oldest-first.  The number of C entries to keep
            // equals the checkpoints within C range that are <= selected_idx.
            int c_to_restore = std::max(0, selected_idx + 1 - b_count);
            // These are the oldest C entries, at the start of saved_c.
            for (int i = 0; i < c_to_restore; i++) {
                add_history(saved_c[i].c_str());
            }
            c_count_since_restore_ = c_to_restore;

            diag("Restoring to: \"" + target.prompt.substr(0, min((int)target.prompt.size(), 60)) + "\" (" + to_string(target.n_past) + " tokens)", "\033[35m");

            // Truncate the token tracker to what we're keeping.  The canonical
            // conversation text no longer describes the tracker: invalidate it
            // (benchmark modes 1/2 rebuild it from the truncated tracker on the
            // next turn) rather than trying to find the text offset of the
            // checkpoint boundary.
            state_.all_context_tokens.resize(target.n_past);
            state_.conversation_text.clear();

            // Undo via seq_rm.  For pure attention models this works instantly.
            // For hybrid models (Qwen3.5/3.6) we first restore the saved recurrent
            // checkpoint, then seq_rm succeeds because plane 0 already holds the
            // correct R/S state - instant undo with no re-decode.
            // Falls back to clear+re-decode only if no checkpoint is available.
            {
                llama_memory_t mem = llama_get_memory(ctx_);

                // Translate the prompt_checkpoints index into a live stack index.
                int stack_idx = selected_idx - state_.checkpoint_stack_offset;
                int n_past_before = n_past_;
                long target_pos = target.n_past;

                if (stack_idx >= 0) {
                    // Restore recurrent state from the target checkpoint.
                    // No-op for pure attention models (no recurrent state).
                    llama_memory_rs_checkpoint_restore(mem, 0, (uint32_t)stack_idx);
                }
                // If stack_idx < 0: no live recurrent checkpoint exists for this
                // pre-restore position.  Skip the restore; seq_rm below will fail
                // on hybrid models (no rs_restored flag), triggering the clear+re-decode fallback.

                bool ok = llama_memory_seq_rm(mem, 0, target.n_past, -1);
                if (ok) {
                    n_past_ = target.n_past;
                    // Prune stale recurrent checkpoints beyond the restored index.
                    if (stack_idx >= 0) {
                        llama_memory_rs_checkpoint_prune(mem, 0, (uint32_t)stack_idx);
                    }
                    // If stack_idx < 0: nothing to prune (no live entries for pre-restore).
                    // MTP stays enabled: the mirror is left ahead of the
                    // truncated main context and self-heals in process() --
                    // the next main decode (the user's new prompt) lands
                    // behind the mirror's stale max, so its position guard
                    // seq_rm's the mirror back to that position and re-
                    // mirrors with fresh hidden states.  Rows before the
                    // undo point are untouched by the undo and remain valid
                    // in the mirror.  Only pending_h_ is stale: it feeds the
                    // embd of exactly one re-mirrored row (the first after
                    // the heal point) -- negligible for acceptance,
                    // irrelevant for correctness (verify re-samples every
                    // committed token from the main model); the same decode
                    // refreshes it.
                    log_rollback("undo", n_past_before, target_pos, true, n_past_);
                } else {
                    diag("Regenerating KV cache for " + to_string(target.n_past) + " tokens...", "\033[35m");
                    llama_memory_clear(mem, true);
                    if (g_mtp) g_mtp->clear();  // re-decode rebuilds the mirror via the hook
                    n_past_ = 0;
                    // Recurrent checkpoints were lost with the clear; reset tracking.
                    state_.checkpoint_stack_offset = (int)state_.prompt_checkpoints.size();
                    state_.tool_correction_checkpoint_idx = -1;
                    state_.rs_checkpoint_saved_this_turn = false;

                    // Re-decode only the prefix up to the undo target.  Feeding
                    // the whole tracker (as before) would re-feed the undone
                    // content -- the model would keep "remembering" it -- and
                    // would leave the KV cache ahead of the tracker, so a later
                    // /save would be rejected by the fast-restore consistency
                    // check.  Copy the prefix first: feed_tokens_impl appends
                    // what it feeds to the tracker, so feeding the tracker
                    // itself would double it.
                    state_.all_context_tokens.resize(target.n_past);
                    vector<llama_token> undo_prefix = state_.all_context_tokens;
                    state_.all_context_tokens.clear();

                    auto start = chrono::high_resolution_clock::now();
                    if (!feed_tokens_impl(undo_prefix)) {
                        bool was_interrupted = stop_generation;
                        stop_generation = 0;  // proceed with the recovery reset
                        log_rollback("undo", n_past_before, target_pos, false, n_past_);
                        // The KV holds a partial re-decoded prefix that the
                        // (empty) token tracker doesn't account for, and the
                        // R/S checkpoint stack was lost with the memory clear:
                        // continuing would append the next prompt to a
                        // mid-system-prompt KV (garbage context), and /save
                        // would write a corrupted file.  Reset to a fresh
                        // session exactly as /clear does (clear_session) -- no
                        // manual /clear needed.  The pre-clear auto-save is
                        // skipped: the tracker is empty, and the pre-undo state
                        // is already in the auto-save written when the undo
                        // started.  Bumping the session keeps that
                        // <N>-clear.save from being clobbered by a later
                        // /clear, and leaves the context in exactly the state
                        // /load requires.
                        string msg = string("Undo re-decode ") +
                            (was_interrupted ? "interrupted" : "failed") +
                            ", context reset to a fresh session";
                        if (undo_autosave_ok) {
                            msg += "; the pre-undo state is in " + pre_undo_save +
                                   " -- /load it to restore.";
                        } else {
                            msg += "; the pre-undo auto-save failed, so it could not be preserved.";
                        }
                        clear_session(history_file, msg, "\033[31m");
                        log_entry("SYSTEM", string("Undo re-decode ") +
                            (was_interrupted ? "interrupted" : "failed") +
                            ", context reset to a fresh session");
                        continue;
                    }

                    // The re-decode ended exactly at the undo target, so the
                    // recurrent state is precisely the target's: save it as
                    // live slot 0 (the stack is empty after the clear) and
                    // mark exactly this checkpoint live.  A later /undo back
                    // to this checkpoint then takes the fast path (restore +
                    // seq_rm) instead of re-decoding the whole prefix a
                    // second time.  No-op for pure attention models (no
                    // recurrent state), matching the /load save calls.
                    llama_memory_rs_checkpoint_save(llama_get_memory(ctx_), 0);
                    state_.checkpoint_stack_offset = selected_idx;

                    auto end = chrono::high_resolution_clock::now();
                    double elapsed = chrono::duration<double>(end - start).count();
                    int secs = (int)elapsed;
                    double speed = target.n_past / (elapsed > 0 ? elapsed : 1.0);
                    diag("KV cache regenerated: " + to_string(target.n_past) + " tokens at " + to_string(round_int(speed)) + " t/s (" + to_string(secs) + "s)", "\033[35m");
                    log_rollback("undo", n_past_before, target_pos, false, n_past_);
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
            // Clear file cache: after undo, the LLM may re-read files it "already saw"
            // in the undone-away turns, but from its current conversation position it
            // has no context of having seen them.  Without clearing, read_files would
            // return false cache hits and skip content the LLM genuinely needs.
            state_.file_cache.clear();
            state_.last_t_count = 0;
            state_.last_elapsed = 0.0;
            state_.last_n_past = n_past_;
            state_.first_turn_done = true;

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
            NetworkTools().reset_search();
            system("reset");
            log_entry("SYSTEM", "Terminal and search reset");
            diag("Terminal Reset Successfully", "\033[32m");
            continue;
        }
        if (last_cmd_ == Command::REMIND) {
            // Re-send the full system prompt, unescaped, as a user turn so the
            // LLM re-anchors to its instructions.  It is sent by LIM rather
            // than read by the model: routed through a tool result, the prompt
            // would arrive with PARAM_END (and turn tokens) escaped per the
            // reserved-token contract, and the model's write-back shortcut
            // ("use the exact same text you saw") would then corrupt the tool
            // schema it is being reminded of.
            if (system_prompt_text_.empty()) {
                diag("/remind: no system prompt to re-send (the prompt file is empty).", "\033[33m");
                continue;
            }

            diag("Sending system prompt to LLM...", "\033[35m");
            log_entry("USER", "[remind] " + system_prompt_text_);
            prev_was_save_ = false;

            // A fresh user turn: clear pending tool-interrupt state, the same
            // as feed_user_message does for a new prompt.  The feed prepends
            // the turn close when mid-turn, so any provisional interrupt
            // checkpoint is finalized at the interrupt position (a legitimate
            // mid-turn undo target).
            state_.tool_interrupt_pending = false;
            state_.partial_tool_text.clear();
            state_.thinking_block_open = false;
            state_.interrupted_checkpoint_idx = -1;

            if (!feed_new_user_turn(system_prompt_text_)) {
                continue;
            }

            // Checkpointed exactly like any turn: the turn-end checkpoint is
            // labeled "/remind", and a 90% threshold crossing (in the feed or
            // mid-turn) gets its permanent mid-turn checkpoint at the crossing
            // too, so a /undo target always exists before the context fills.
            last_user_input_ = "/remind";

            // A new user-initiated turn starts a fresh auto-continue chain:
            // generate_response's reset (only when auto_continue is false) is
            // skipped below, since we set auto_continue = true so the loop
            // reaches generation without re-prompting.  Without this reset, a
            // deep tool chain from the previous turn would count against this
            // turn's chain and could trip the max-auto-continue loop eject
            // early.
            g_auto_continue_depth_ = 0;
            state_.auto_continue = true;
            continue;
        }
        if (last_cmd_ == Command::REINCARNATE) {
            // Auto-save before reincarnating so nothing is truly lost.
            // Uses the same -clear.save name since reincarnate calls clear_context internally.
            autosave_before_clear();

            string reincarnate_path = LIM_CONFIG_DIR + "/reincarnate";
            ifstream reincarnate_file(reincarnate_path);
            if (!reincarnate_file.is_open()) {
                diag("Reincarnate failed: Cannot open " + reincarnate_path, "\033[31m");
                continue;
            }
            stringstream reincarnate_buffer;
            reincarnate_buffer << "Use the write_file tool to write a new prompt to "
                          << LIM_CONFIG_DIR << "/userprompt. Read the following instructions and compose an appropriate prompt, deferring further detailed thinking to the next session. Then write it and return without further comment. Do not resume your task. "
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
            vector<llama_token> reincarnate_tokens = build_new_user_turn_tokens(reincarnate_text);

            if (n_past_ + (int)reincarnate_tokens.size() >= (int)cparams_.n_ctx) {
                string ctx_diag = context_limit_diag(n_past_, state_.last_n_past, reincarnate_tokens.size());
                diag("Context Limit Reached! Cannot process reincarnate" + ctx_diag + ". Type '/clear' to reset.", "\033[31m");
                continue;
            }

            if (!feed_tokens_impl(reincarnate_tokens)) {
                if (stop_generation) stop_generation = 0;
                continue;
            }

            // The reincarnate turn was fed but not appended to the canonical
            // conversation text: invalidate it.  The successful path clears the
            // context again (re-seeding the text); if generation is interrupted
            // before that, the invalid text is rebuilt from the tracker by the
            // next mode 1/2 turn instead of being used stale.
            state_.conversation_text.clear();

            // Log reincarnate tokens to token_log when debug is enabled
            log_tokens("FEED USER_INPUT", reincarnate_tokens, ctx_);

            state_.auto_continue = true;
            state_.reincarnate_mode = true;
            reset_session_state();
            continue;
        }
        if (last_cmd_ == Command::SAVE) {
            string default_path = LIM_LOG_DIR + "/" + to_string(state_.log_index) + SAVE_EXT;
            string save_path = apply_save_dir(save_prefix_.empty() ? default_path : append_save_ext(save_prefix_));

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
            // Append .save if not already present (matches /save and CLI behavior),
            // then prepend LIM_SAVE_DIR to relative paths.
            rpath = apply_save_dir(append_save_ext(rpath));

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

            // Fail fast if the session can't fit in the context: the restore
            // decode needs one KV position per token. Without this check the
            // decode would exhaust the KV cache mid-restore, the state tracker
            // would claim a restore that never completed, and a truncated fast
            // cache entry would be written for it.
            if ((int)restored_tokens.size() >= (int)cparams_.n_ctx) {
                diag("Restore failed: session has " + to_string(restored_tokens.size()) +
                     " tokens, which does not fit in the context size (" + to_string(cparams_.n_ctx) +
                     "). Increase LIM_CTX or restore an older checkpoint.", "\033[31m");
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

            // Try instant restore from V1 cache first (skipped with --checkpoints).
            // When MTP is active, the cache's mirror KV state (if it holds one)
            // is restored into the draft context alongside the main KV.
            bool mtp_loaded = false;
            bool cache_hit = restore_checkpoints_ ? false
                : try_load_v1_cache(restore_path_abs, restored_tokens, g_model_path, ctx_,
                                    (g_mtp && g_mtp->valid()) ? g_mtp->draft_ctx() : nullptr,
                                    &mtp_loaded);
            int saved_session = read_save_session(rpath);
            if (cache_hit) {
                if (g_mtp && mtp_loaded) {
                    // The mirror KV came from the same save as the main KV, so
                    // it matches the restored context; re-arm the speculator.
                    g_mtp->on_mirror_loaded();
                } else if (g_mtp) {
                    // The cache holds no usable mirror state (saved without MTP,
                    // or an incompatible mirror KV type): it can only be rebuilt
                    // by a /clear or a re-decode, so drafting stops until then.
                    g_mtp->invalidate("mirror stale after fast restore");
                }
                diag_restore(rpath, (int)restored_tokens.size());
                n_past_ = (int)llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) + 1;

                // Update session state.
                state_.all_context_tokens = restored_tokens;
                // Restored tracker comes from disk, not from a maintained text:
                // invalidate the canonical conversation text (benchmark modes
                // 1/2 rebuild it from the tokens on the next turn).
                state_.conversation_text.clear();
                state_.prompt_checkpoints = read_checkpoint_offsets(rpath);
                // Save a boundary checkpoint so instant undo works for the restore
                // point and subsequent new turns (same logic as CLI fast restore).
                if (!state_.prompt_checkpoints.empty()) {
                    llama_memory_rs_checkpoint_save(llama_get_memory(ctx_), 0);
                    state_.checkpoint_stack_offset = (int)state_.prompt_checkpoints.size() - 1;
                } else {
                    state_.checkpoint_stack_offset = 0;
                }
                state_.interrupted_checkpoint_idx = -1;  // list replaced from disk

                diag_session_restored(saved_session, restored_tokens.size(), (int)cparams_.n_ctx);
                log_entry("SYSTEM", "Restored session from " + rpath);
            } else {
                // Slow restore: re-decode through the model.
                diag_restore(rpath, (int)restored_tokens.size());

                vector<PromptCheckpoint> restored_checkpoints = read_checkpoint_offsets(rpath);

                // Slow restore: offer partial restore via readline history
                // navigation so the user can select a restore point.
                int restore_limit = (int)restored_tokens.size();
                int total_restore_tokens = restore_limit;
                if (!restored_checkpoints.empty()) {
                    // Snapshot the current history (A + B + C) so it can be
                    // restored after the selection prompt.
                    vector<string> saved_hist;  // oldest first
                    // history_get() is 1-based: 1 = oldest, history_length = newest,
                    // so iterate forward to collect oldest-first (add_history appends
                    // at the newest end, so restoring in this order preserves it).
                    for (int i = 1; i <= history_length; i++) {
                        HIST_ENTRY* he = history_get(i);
                        if (he) saved_hist.push_back(he->line);
                    }

                    using_history();
                    clear_history();

                    diag("Save contains " + to_string(restored_checkpoints.size()) + " prompt checkpoint" + (restored_checkpoints.size() != 1 ? "s" : "") + ".", "\033[35m");
                    diag("Up/down arrows to navigate, Enter to confirm.", "\033[37m");

                    // Add checkpoints oldest-to-newest.
                    // Pressing up from empty line shows the most recent checkpoint first.
                    for (const auto& cp : restored_checkpoints) {
                        add_history(checkpoint_label(cp).c_str());
                    }

                    char* line = readline("Restore> ");
                    bool cancelled = false;
                    // If Ctrl+C was pressed during readline, stop_generation is set.
                    if (stop_generation) {
                        cancelled = true;
                        stop_generation = 0;
                        if (line) free(line);
                    } else if (!line) {
                        cancelled = true;  // Ctrl+D (EOF)
                    } else {
                        string input = line;
                        free(line);
                        if (input == "/quit" || input == "/exit") {
                            // Graceful cancel, like the Undo> prompt: return to
                            // the user prompt without losing the session.
                            cancelled = true;
                        } else {
                            // Match by the "(N tokens)" suffix (n_past is unique per
                            // checkpoint, so this works even for truncated or
                            // duplicated prompts).
                            int target_n_past = -1;
                            if (parse_checkpoint_selection(input, &target_n_past)) {
                                for (const auto& cp : restored_checkpoints) {
                                    if (cp.n_past == target_n_past) {
                                        restore_limit = cp.n_past;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // Restore A + B + C history.
                    clear_history();
                    for (const auto& s : saved_hist) add_history(s.c_str());

                    if (cancelled) {
                        diag("Restore cancelled.", "\033[35m");
                        continue;
                    }

                    if (restore_limit < (int)restored_tokens.size()) {
                        diag("Restoring to checkpoint: " + to_string(restore_limit) + " tokens", "\033[35m");
                        // Trim to the selected range so the decode and session
                        // state cover exactly the restored prefix.
                        restored_tokens.resize(restore_limit);
                        restored_checkpoints.erase(
                            std::remove_if(restored_checkpoints.begin(), restored_checkpoints.end(),
                                           [restore_limit](const PromptCheckpoint& cp) { return cp.n_past > restore_limit; }),
                            restored_checkpoints.end());
                    }
                }

                // Clear the KV cache before re-decoding.
                llama_memory_clear(llama_get_memory(ctx_), true);
                if (g_mtp) g_mtp->clear();  // re-decode rebuilds the mirror via the hook
                n_past_ = 0;
                state_.all_context_tokens.clear();
                // The old tracker is gone (and the decode below may fail):
                // invalidate the canonical conversation text now; the success
                // path re-invalidates after re-feeding, the failure path
                // re-seeds it with the system prompt.
                state_.conversation_text.clear();

                size_t cp_restore_idx = 0;

                auto restore_start = chrono::high_resolution_clock::now();
                bool restore_failed = false;
                for (int i = 0, chunk = 0; i < (int)restored_tokens.size() && !restore_failed; i += chunk) {
                    // Size the chunk so it ends exactly on the next prompt boundary
                    // when one falls within n_batch: the recurrent checkpoint saved
                    // after the decode must correspond to the exact boundary position
                    // (a checkpoint saved at the chunk end would let instant undo
                    // "remember" up to a batch's worth of the undone turn).
                    chunk = std::min((int)cparams_.n_batch, (int)restored_tokens.size() - i);
                    if (cp_restore_idx < restored_checkpoints.size() &&
                        restored_checkpoints[cp_restore_idx].n_past > i &&
                        restored_checkpoints[cp_restore_idx].n_past < i + chunk) {
                        chunk = restored_checkpoints[cp_restore_idx].n_past - i;
                    }
                    batch_.n_tokens = 0;
                    for (int j = 0; j < chunk; j++) {
                        common_batch_add(batch_, restored_tokens[i + j], n_past_, {0}, (i + j == (int)restored_tokens.size() - 1));
                        n_past_++;
                    }
                    // should_break=true: KV exhaustion (ret 1) and aborts
                    // (ret 2) must count as failures here -- with false they
                    // would slip through and the code below would report a
                    // successful restore of a partially decoded cache.
                    if (stop_generation ||
                        !handle_llama_decode_error(ctx_, batch_, "Decode failed during restore.", true)) {
                        sync_n_past(ctx_, n_past_);
                        restore_failed = true;
                    } else {
                        // n_past_ counts from 0 over the whole restore.
                        stream_feed_progress(restore_start, 0);
                    }
                    // Save recurrent checkpoints at prompt boundaries.
                    while (cp_restore_idx < restored_checkpoints.size() &&
                           restored_checkpoints[cp_restore_idx].n_past <= n_past_) {
                        llama_memory_rs_checkpoint_save(llama_get_memory(ctx_), 0);
                        cp_restore_idx++;
                    }
                }
                sync_n_past(ctx_, n_past_);

                if (restore_failed) {
                    // The KV cache holds a partial restored prefix that the
                    // token tracker doesn't account for (all_context_tokens
                    // was cleared before the decode and is still empty).
                    // Continuing would let the model answer from context that
                    // /save and /undo can't see, and a /save would write a
                    // corrupted file. Reset to a clean fresh session instead.
                    bool was_interrupted = stop_generation;
                    stop_generation = 0;  // proceed with the recovery feed
                    llama_memory_clear(llama_get_memory(ctx_), true);
                    n_past_ = 0;
                    state_.all_context_tokens.clear();
                    state_.prompt_checkpoints.clear();
                    state_.file_cache.clear();
                    state_.auto_continue = false;
                    state_.prev_was_interrupted = false;
                    state_.checkpoint_stack_offset = 0;
                    state_.tool_correction_checkpoint_idx = -1;
                    reset_session_state();
                    if (!feed_tokens_impl(system_tokens_)) {
                        diag(string("Restore ") + (was_interrupted ? "interrupted" : "failed") +
                             " and the context reset also failed. Type '/clear' to recover.", "\033[31m");
                    } else {
                        // Fresh session re-seeded with the system prompt: re-seed
                        // the canonical conversation text the same way
                        // clear_context() does.
                        if (maintains_conversation_text()) {
                            state_.conversation_text = build_system_turn_text(system_prompt_text_);
                        }
                        llama_sampler_reset(smpl_);
                        diag(string("Restore ") + (was_interrupted ? "interrupted" : "failed") +
                             ": context reset to a fresh session", "\033[31m");
                        log_entry("SYSTEM", string("Restore ") + (was_interrupted ? "interrupted" : "failed") +
                                  ", context reset to a fresh session");
                    }
                    continue;
                }

                auto restore_end = chrono::high_resolution_clock::now();
                double restore_elapsed = chrono::duration<double>(restore_end - restore_start).count();
                double restore_speed = (restore_elapsed > 0) ? restored_tokens.size() / restore_elapsed : 0;
                diag("KV cache regenerated: " + to_string(restored_tokens.size()) + " tokens at " +
                     std::to_string((int)restore_speed) + " t/s (" +
                     std::to_string((int)restore_elapsed) + "s)", "\033[35m");

                // Update session state.
                state_.all_context_tokens = restored_tokens;
                // Restored tracker comes from disk, not from a maintained text:
                // invalidate the canonical conversation text (benchmark modes
                // 1/2 rebuild it from the tokens on the next turn).
                state_.conversation_text.clear();
                state_.prompt_checkpoints = restored_checkpoints;
                state_.checkpoint_stack_offset = 0; // all checkpoints are live
                state_.interrupted_checkpoint_idx = -1;  // list replaced from disk

                // Auto-write V1 cache for instant future restores (full restore only
                // -- a partial prefix would hash to an entry a later full restore
                // never matches; skipped with --checkpoints). The n_past_ check
                // is defense in depth: the cache must cover exactly the tracked
                // tokens or a future fast restore would desync.
                if (!restore_checkpoints_ && !restore_path_abs.empty() &&
                    (int)restored_tokens.size() == total_restore_tokens &&
                    n_past_ == (int)restored_tokens.size()) {
                    if (is_debug) {
                        diag("Save to cache.", "\033[35m");
                    }
                    // The slow-restore re-decode fed the mirror via the decode
                    // hook, so it is consistent here: persist it too.
                    write_v1_cache(restore_path_abs, restored_tokens, g_model_path, ctx_, "",
                                   (g_mtp && g_mtp->valid()) ? g_mtp->draft_ctx() : nullptr);
                }

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

                check_git_head_on_restore(rpath, saved_sha, ctx_, batch_, n_past_, state_.all_context_tokens, (int)cparams_.n_batch);
            }

            // Reset sampler state for a clean generation start.
            llama_sampler_reset(smpl_);
            // Save C (user inputs since last restore/clear) before repopulating,
            // so they survive the restore and appear after the restored prompts.
            {
                vector<string> saved_c = collect_recent_user_inputs();  // oldest first

                // Repopulate readline history from restored checkpoints so up-arrow
                // navigates through the restored session's prompts.
                repopulate_history();

                // Re-push C entries on top of the restored checkpoint prompts,
                // in chronological order. Re-count the ones actually added:
                // repopulate_history() zeroed the counter, and without this the
                // /quit flush would drop C from disk.
                for (const auto& s : saved_c) {
                    int before = history_length;
                    add_history(s.c_str());
                    if (history_length > before) c_count_since_restore_++;
                }
            }

            // Update browser status bar with restored context position
            if (should_output_to_browser()) {
                double context_percent = (n_past_ / (double)cparams_.n_ctx) * 100.0;
                string ctx_str = std::to_string(n_past_) + " (" + std::to_string((int)context_percent) + "%)";
                pipe_write(&SEG_SPEED, 1);
                string speed_msg = "Loaded | " + ctx_str;
                pipe_write(speed_msg.c_str(), speed_msg.length());
            }

            continue;
        }

        if (last_cmd_ == Command::DELETE) {
            // Build delete path: require a non-empty argument.
            string dpath = delete_path_;
            if (dpath.empty()) {
                diag("/delete requires a path argument. Usage: /delete <save_file>", "\033[31m");
                continue;
            }
            // Append .save if not already present (matches /save and /load behavior),
            // then prepend LIM_SAVE_DIR to relative paths.
            dpath = apply_save_dir(append_save_ext(dpath));

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
                // Mid-turn position (only reachable after /undo to a mid-turn
                // checkpoint): no live interrupt to resume -- no-op silently.
                // The turn will be closed by the next real user prompt.
                if (position_ends_mid_turn()) continue;
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
                    // Keep the canonical conversation text (benchmark modes 1/2)
                    // in lockstep with the fed prefill.
                    if (!state_.conversation_text.empty()) {
                        state_.conversation_text += g_model_tokens.assistant_turn_start.text;
                    }
                }
                user_input = "";
            } else {
                // No turns have happened yet -- nothing to resume. Return silently to prompt.
                continue;
            }
        }

        // 3. Reject unrecognized commands before they reach the LLM
        if (last_cmd_ == Command::NONE && !user_input.empty() && user_input[0] == '/') {
            // Only show "Unknown command" if no known command name was matched at all.
            // If handle_command returned NONE because of extra arguments on a recognized
            // command, it already printed its own warning -- don't double-report.
            if (!match_command(user_input.substr(1))) {
                diag("Unknown command: " + user_input + ". Type /help for available commands.", "\033[33m");
            }
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
            // A new prompt finalizes any provisional interrupt checkpoint: the
            // feed prepends the turn close and the checkpoint stands at the
            // interrupt position (a legitimate mid-turn undo target).
            state_.interrupted_checkpoint_idx = -1;

            // Chatbot mode: re-decode full history each turn for comparison
            if (chatbot_mode == 1 && !state_.all_context_tokens.empty()) {
                // Mode 1: standard chatbot.  The conversation is held as TEXT
                // (like a plain chat UI): each turn re-tokenizes the full
                // conversation text and re-decodes everything from scratch --
                // no KV reuse.  Per-turn cost: O(N) re-tokenize + O(N) re-decode.
                // The canonical text is maintained per turn (see
                // SessionState::conversation_text), so steady state is a pure
                // append -- the one-time detokenize round-trip only runs after
                // an event that discarded the text (build_full_conversation_text).
                // Timing: text append + re-tokenize + clear + full re-decode
                // (modes 1/2 force LIM_HONEST_SPEED=1, so it is part of the
                // measured TPS).
                auto feed_start = chrono::high_resolution_clock::now();

                // 1. New turn text (same construction as the mode 2 branch).
                string new_turn_text = build_new_user_turn_text(user_input);

                // 2. Full conversation text: canonical text + new turn, or the
                // one-time detokenize reconstruction if the text was discarded.
                string full_text =
                    build_full_conversation_text(state_.all_context_tokens, new_turn_text);

                // 3. Re-tokenize the full conversation in one pass: BOS at
                // position 0 iff the cached stream started with BOS (see
                // build_system_prompt_tokens), and parse_special=true so literal
                // turn-marker strings round-trip back into special tokens.
                vector<llama_token> full_request =
                    common_tokenize(vocab_, full_text, should_add_bos(state_.all_context_tokens), true);

                // Fail fast if the full request cannot fit in the context (like
                // the mode 2 and restore paths), before touching the KV cache.
                if ((int)full_request.size() >= (int)cparams_.n_ctx) {
                    auto feed_end = chrono::high_resolution_clock::now();
                    state_.last_feed_time = chrono::duration<double>(feed_end - feed_start).count();
                    diag("Context Limit Reached! Mode 1 full request does not fit" +
                         context_limit_diag(n_past_, state_.last_n_past, full_request.size()) +
                         ". Type '/clear' to reset.", "\033[31m");
                    continue;
                }

                // 4. A plain chatbot has no persistent cache: clear and re-decode
                // the whole request from scratch.
                llama_memory_clear(llama_get_memory(ctx_), true);
                n_past_ = 0;
                state_.all_context_tokens.clear();
                state_.prompt_checkpoints.clear();
                state_.checkpoint_stack_offset = 0;
                state_.tool_correction_checkpoint_idx = -1;
                state_.interrupted_checkpoint_idx = -1;
                state_.rs_checkpoint_saved_this_turn = false;
                llama_sampler_reset(smpl_);

                diag("Chatbot mode 1: re-decoding " + to_string(full_request.size()) +
                     " tokens (" + to_string(full_text.size()) + " chars of conversation text)", "\033[90m");
                if (!feed_tokens_impl(full_request)) continue;
                // The feed succeeded: the tracker is exactly full_request, so the
                // canonical text is valid again.
                state_.conversation_text = full_text;
                auto feed_end = chrono::high_resolution_clock::now();
                state_.last_feed_time = chrono::duration<double>(feed_end - feed_start).count();
                log_tokens("FEED MODE1_FULL", full_request, ctx_);
                // Perform the logging/browser output that feed_user_message would do,
                // but skip its token feeding since we already included the input above.
                if (!state_.auto_continue) {
                    log_entry("USER", user_input);
                    stream_user_input_html(user_input);
                }
                // Skip feed_user_message -- tokens already fed. Fall through to generate_response().
            } else if (chatbot_mode == 2 && !state_.all_context_tokens.empty()) {                // Mode 2: cache-aware prefix matching (emulates llama-server behavior).
                // Emulate llama-server: the KV cache persists across turns, but every
                // request re-tokenizes the full conversation from scratch and decodes
                // only the delta after the longest token prefix still in the cache.
                // Per-turn cost (steady state): O(N) text append + O(N) re-tokenize +
                // O(N) prefix compare + O(delta) decode -- the detokenize tax the
                // server never pays is gone; the one-time detokenize reconstruction
                // (inside build_full_conversation_text) only runs after an event
                // that discarded the text.
                vector<llama_token> saved_history = state_.all_context_tokens;

                // Timing: text append + re-tokenize + prefix match + decode delta.
                // All of it lands in last_feed_time (modes 1/2 force
                // LIM_HONEST_SPEED=1, so it is part of the measured TPS).
                auto feed_start = chrono::high_resolution_clock::now();

                // 1. New turn text (same construction as the mode 1 branch).
                string new_turn_text = build_new_user_turn_text(user_input);

                // 2. Full conversation text, exactly as the server would receive it:
                // the canonical conversation text (maintained per turn; see
                // SessionState::conversation_text) plus the new turn.  Steady state
                // is a pure append -- the one-time detokenize reconstruction only
                // runs when the text was discarded (session start, /undo, /load,
                // tool rollback).
                string full_text = build_full_conversation_text(saved_history, new_turn_text);

                // 3. Re-tokenize the full conversation in one pass: BOS at position 0
                // iff the cached stream started with BOS (see
                // build_system_prompt_tokens / should_add_bos), and parse_special=true
                // so literal turn-marker strings round-trip back into special tokens.
                vector<llama_token> full_request =
                    common_tokenize(vocab_, full_text, should_add_bos(saved_history), true);

                // Fail fast if the full request cannot fit in the context (like the
                // restore path), before touching the KV cache.
                if ((int)full_request.size() >= (int)cparams_.n_ctx) {
                    auto feed_end = chrono::high_resolution_clock::now();
                    state_.last_feed_time = chrono::duration<double>(feed_end - feed_start).count();
                    diag("Context Limit Reached! Mode 2 full request does not fit" +
                         context_limit_diag(n_past_, state_.last_n_past, full_request.size()) +
                         ". Type '/clear' to reset.", "\033[31m");
                    continue;
                }

                // 4. Real prefix match: longest i with saved_history[i] ==
                // full_request[i].  Now that re-tokenization is real, the match can be
                // shorter than the cache (drift), and the request can be shorter than
                // the cache (round-trip truncation).
                size_t match_len = 0;
                size_t min_len = std::min(saved_history.size(), full_request.size());
                while (match_len < min_len && saved_history[match_len] == full_request[match_len]) {
                    match_len++;
                }

                // Position the suffix feed is decoded from.  Normally match_len; on
                // drift the KV tail is removed and the feed starts where the cache was
                // truncated (the drift point, or an earlier prompt checkpoint on
                // hybrid models, or 0 after a full clear).
                size_t cut = match_len;
                if (match_len < saved_history.size()) {
                    // Stale KV tail: positions match_len..end hold cache entries for
                    // tokens the re-tokenized request no longer has.  Remove them so the
                    // suffix decodes at the right positions and the token tracker and KV
                    // cache stay consistent.  llama-server does the same: truncate the
                    // cache where the match ends and re-parse the suffix.  The re-parsed
                    // suffix is canonical, so the next turn matches fully again (the
                    // drift is a one-time cost).
                    llama_memory_t mem = llama_get_memory(ctx_);
                    int cp_idx = -1;
                    bool rm_ok = llama_memory_seq_rm(mem, 0, (llama_pos)match_len, -1);
                    if (!rm_ok) {
                        // Hybrid model: seq_rm also needs the recurrent (R/S) state at
                        // the cut point, which is only checkpointed at prompt returns.
                        // Anchor the cut at the newest prompt checkpoint at or before
                        // the drift point: restore its recurrent state (like the undo
                        // path), truncate there, and re-decode the shorter suffix.
                        for (int i = (int)state_.prompt_checkpoints.size() - 1; i >= 0; i--) {
                            if (state_.prompt_checkpoints[i].n_past <= (int)match_len &&
                                i - state_.checkpoint_stack_offset >= 0) {
                                cp_idx = i;
                                break;
                            }
                        }
                        if (cp_idx >= 0) {
                            int target = state_.prompt_checkpoints[cp_idx].n_past;
                            llama_memory_rs_checkpoint_restore(mem, 0,
                                (uint32_t)(cp_idx - state_.checkpoint_stack_offset));
                            if (llama_memory_seq_rm(mem, 0, (llama_pos)target, -1)) {
                                cut = (size_t)target;
                                rm_ok = true;
                            }
                        }
                    }
                    if (rm_ok) {
                        n_past_ = (int)cut;
                        if (cut != match_len) {
                            diag("Chatbot mode 2: re-tokenization drift at token " + to_string(match_len) +
                                 " of " + to_string(saved_history.size()) +
                                 "; re-decoding from prompt checkpoint " + to_string(cut), "\033[35m");
                            // Slots beyond the anchor describe the old (drifted) tail:
                            // prune them so a later undo re-decodes instead of restoring
                            // stale R/S state.  (No-op when the anchor is the newest
                            // checkpoint, the common case.)
                            int live_top = cp_idx - state_.checkpoint_stack_offset;
                            if (live_top < (int)state_.prompt_checkpoints.size() - 1 - state_.checkpoint_stack_offset) {
                                llama_memory_rs_checkpoint_prune(mem, 0, (uint32_t)live_top);
                                state_.checkpoint_stack_offset = cp_idx + 1;
                            }
                            state_.tool_correction_checkpoint_idx = -1;
                            state_.rs_checkpoint_saved_this_turn = false;
                        }
                    } else {
                        // No usable recurrent state at or before the drift point:
                        // clear and re-decode the full request from scratch (like the
                        // undo fallback).  Recurrent checkpoints are lost with the clear.
                        diag("Chatbot mode 2: re-tokenization drift at token " + to_string(match_len) +
                             " of " + to_string(saved_history.size()) +
                             "; regenerating KV cache", "\033[35m");
                        llama_memory_clear(mem, true);
                        n_past_ = 0;
                        cut = 0;
                        state_.checkpoint_stack_offset = (int)state_.prompt_checkpoints.size();
                        state_.tool_correction_checkpoint_idx = -1;
                        state_.rs_checkpoint_saved_this_turn = false;
                    }
                    // feed_tokens_impl appends to the tracker: shrink it to the cut
                    // point first so it ends up exactly full_request after the suffix feed.
                    state_.all_context_tokens.resize(cut);
                    // Drop checkpoints beyond the (possibly shorter) new sequence.
                    int full_len = (int)full_request.size();
                    state_.prompt_checkpoints.erase(
                        std::remove_if(state_.prompt_checkpoints.begin(), state_.prompt_checkpoints.end(),
                                       [full_len](const PromptCheckpoint& cp) { return cp.n_past > full_len; }),
                        state_.prompt_checkpoints.end());
                }

                // 5. Decode only the delta after the cut point (normally the matched
                // prefix; earlier on drift, where the KV cache was truncated).
                vector<llama_token> delta(full_request.begin() + cut, full_request.end());
                diag("Chatbot mode 2: prefix match " + to_string(match_len) + "/" +
                     to_string(saved_history.size()) + " tokens, decoding " +
                     to_string(delta.size()) + " new", "\033[90m");
                if (!delta.empty() && !feed_tokens_impl(delta)) {
                    auto feed_end = chrono::high_resolution_clock::now();
                    state_.last_feed_time = chrono::duration<double>(feed_end - feed_start).count();
                    continue;
                }

                // The feed succeeded: the tracker is exactly full_request (after
                // the possible drift truncation), so the canonical text is valid
                // again and the next turn appends to it.
                state_.conversation_text = full_text;

                auto feed_end = chrono::high_resolution_clock::now();
                state_.last_feed_time = chrono::duration<double>(feed_end - feed_start).count();
                if (!delta.empty()) log_tokens("FEED MODE2_DELTA", delta, ctx_);
                // Logging
                if (!state_.auto_continue) {
                    log_entry("USER", user_input);
                    stream_user_input_html(user_input);
                }

            } else {
                state_.last_feed_time = 0.0;
            }

            // Mode 1 already fed tokens above and falls through to generate_response().
            // Mode 2 fed tokens only when cache existed; otherwise needs feed_user_message.
            // Mode 0 always uses feed_user_message.
            if (chatbot_mode == 1 && !state_.all_context_tokens.empty()) {
                // Already fed history + new turn tokens above.
            } else if (chatbot_mode == 2 && !state_.all_context_tokens.empty()) {
                // Already fed delta tokens above.
            } else if (!feed_user_message(user_input)) {
                continue;
            }
        }

        // 7. Generate response
        auto gen_result = generate_response();

        // 8. Process tool call
        if (process_tool_call()) {
            continue;
        }

        // 8a. Stuck tool call (silent-loop abort): handle identically to an invalid
        // tool call -- consume this call's correction attempt, then run the same
        // correction cycle in step 8b (feed system prompt, regenerate once,
        // roll back to right after FUNC_START, inject the clean call). The garbage
        // tokens are simply part of what the rollback removes.
        // The slot may be stale when MTP speculation left the recurrent state
        // ahead of n_past_ at the FUNC_START (uncommitted verify cells: the hook
        // skipped the save) -- the rollback in step 8b handles that by re-decoding
        // the suffix from the newest prompt checkpoint (never a full-context
        // re-decode).  Ejects when this call's correction attempt is already
        // spent.
        if (gen_result.stuck_in_tool_call) {
            if (!state_.correction_attempted_this_turn) {
                diag("System: Invalid tool call. Attempting correction.", "\033[1;33m");
                state_.correction_attempted_this_turn = true;
                state_.tool_correction_mode = true;
            } else {
                diag("Aborting to prompt.", "\033[1;31m");
            }
        }

        // 8b. Tool-call correction: feed system prompt, wait for valid tool call,
        // roll back, inject into Assistant field, then let process_tool_call execute.
        if (state_.tool_correction_mode) {
            state_.tool_correction_mode = false;

            // Base prompt file only (no localprompt/cwd/date): the correction
            // message is rolled back later, so it must stay compact.
            string system_prompt = read_prompt_file();

            string correction_msg = "System Error: Invalid tool call. Did you forget a closing parameter or function tag in your tool call?\n\nFollow these instructions strictly:\n\n";
            correction_msg += system_prompt;

            vector<llama_token> correction_tokens = build_tool_result_turn(ctx_, correction_msg);
            if (n_past_ + (int)correction_tokens.size() < (int)cparams_.n_ctx) {
                if (!feed_tokens_impl(correction_tokens)) {
                    // The correction prompt could not be fully decoded: n_past_
                    // is synced to the partial KV and the token tracker stays
                    // short of it -- continuing would regenerate from a broken
                    // context and desync save/restore.  Eject to the prompt.
                    diag("System: Tool correction aborted: failed to feed correction prompt. Type '/clear' to reset.", "\033[1;31m");
                    state_.conversation_text.clear();
                    state_.auto_continue = false;
                    state_.correction_attempted_this_turn = false;
                    continue;
                }
                log_tokens("FEED TOOL_CORRECTION", correction_tokens, ctx_);

                // Generate once -- LLM produces corrected tool call.
                gen_result_ = generate_response(/*is_correction_gen=*/true);
                generated_text_ = gen_result_.text;
                t_count_ = gen_result_.token_count;
                elapsed_ = gen_result_.decode_time;

                // If valid, roll back to checkpoint, inject good call, hand off to process_tool_call.
                if (gen_result_.has_tool_call && gen_result_.tool_start != string::npos && gen_result_.tool_end != string::npos) {
                    // Extract and validate the corrected tool call before injecting it.
                    // The parser only checks for XML structure; we must verify the tool name
                    // and parameters are actually correct to avoid injecting another bad call.
                    string corr_text = gen_result_.text;
                    size_t cs = gen_result_.tool_start;
                    size_t ce = gen_result_.tool_end;
                    string corr_tool_call_raw = corr_text.substr(cs, ce - cs + string(FUNC_END).length());
                    if (!validate_tool_call(corr_tool_call_raw)) {
                        diag("System: Correction produced another invalid tool call. Ejecting to prompt.", "\033[1;31m");

                        // Eject WITHOUT rolling back: a rollback would leave the
                        // context at a weird place (right after FUNC_START, mid
                        // tool call with no body), and the rollback itself is the
                        // unstable part (a failed seq_rm falls back to the
                        // prompt-anchor suffix re-decode, or aborts the turn if
                        // no anchor exists).  The failed correction stays in
                        // the LLM's history; it recovers on the next turn.  The
                        // checkpoint slot is deliberately left untouched (still
                        // valid at tool_correction_n_past if the hook pushed
                        // it): the next prompt return overwrites it in place,
                        // and a /continue resume can still correct from it.
                        // has_tool_correction_checkpoint is left as the hook
                        // set it (a /continue re-correction without a pushed
                        // slot takes the prompt-anchor suffix re-decode path in
                        // the rollback instead of restoring a missing entry).
                        // MTP is untouched: no rollback, so the mirror stays
                        // in lockstep.
                        state_.conversation_text.clear();  // fed correction prompt is not in it; rebuild from tracker
                        state_.auto_continue = false;
                        // Returning control to the user prompt: the correction latch is
                        // only meaningful within one auto-continue chain (step 9 clears it
                        // on a normal prompt return, but this path `continue`s past it).
                        // Clearing it also lets /continue trigger a fresh correction.
                        state_.correction_attempted_this_turn = false;
                        continue;
                    }
                    diag("System: Tool correction successful, injecting clean tool call.", "\033[35m");

                    // Roll back to the tool-correction checkpoint (removes bad call + system prompt + correction).
                    // rc: 0 = fast seq_rm (the slot holds the R/S state for the
                    // target: the FUNC_START was in lockstep), 2 = prompt-checkpoint
                    // anchor + suffix re-decode (the slot is stale/absent after a
                    // non-lockstep MTP FUNC_START, or seq_rm failed: the R/S state
                    // is rebuilt from a valid prompt checkpoint; the re-decode ends
                    // at the same FUNC_START, so the slot, if pushed, still holds
                    // the R/S state for exactly this position and the end-of-turn
                    // overwrite reuses it), -1 = no anchor available -- the turn
                    // ejects to the prompt instead of paying for a full-context
                    // re-decode.
                    int rollback_rc = rollback_to_tool_checkpoint(true);
                    if (rollback_rc < 0) {
                        // The rollback could not complete without a full-context
                        // re-decode, which we never do: eject to the prompt.  The
                        // failed correction stays in the LLM's history; /clear
                        // recovers fully.
                        state_.conversation_text.clear();  // fed correction prompt is not in it; rebuild from tracker
                        state_.auto_continue = false;
                        state_.correction_attempted_this_turn = false;
                        continue;
                    }

                    // Inject the corrected tool call into the Assistant field.
                    // The rollback above left context ending in [preamble][FUNC_START],
                    // so feed ONLY the call body (drop the leading FUNC_START -- it is
                    // already in context).  Keep the full string (with FUNC_START) for
                    // ToolExecutor parsing below.
                    string corr_preamble = (cs > 0) ? corr_text.substr(0, cs) : "";
                    vector<llama_token> inj_tokens = tokenize(corr_tool_call_raw.substr(string(FUNC_START).length()));
                    if (!feed_tokens_impl(inj_tokens)) {
                        // The clean call could not be fully decoded: n_past_ is
                        // synced to the partial KV and the tracker is short --
                        // executing the tool now would corrupt the context.
                        // The rollback already ran, so the context ends right
                        // after FUNC_START; eject to the prompt.
                        diag("System: Tool correction aborted: failed to feed injected tool call. Type '/clear' to reset.", "\033[1;31m");
                        state_.conversation_text.clear();
                        state_.auto_continue = false;
                        state_.correction_attempted_this_turn = false;
                        continue;
                    }
                    log_tokens("FEED TOOL_CORRECTION_INJECT", inj_tokens, ctx_);

                    // Set up gen_result_ to reflect the injected call.
                    generated_text_ = corr_preamble + corr_tool_call_raw;
                    gen_result_.tool_start = corr_preamble.length();
                    size_t fe_pos = corr_tool_call_raw.find(string(FUNC_END));
                    gen_result_.tool_end = corr_preamble.length() + (fe_pos != string::npos ? fe_pos : corr_tool_call_raw.length());
                    was_mid_tool_call_ = false;

                    // Reset sampler: the penalty ring buffer contains stale tokens
                    // from the correction phase that were rolled back.
                    llama_sampler_reset(smpl_);

                    // Reset correction flags so that if this injected call also fails,
                    // another correction cycle can run.  Do NOT touch the checkpoint
                    // slot or tool_correction_n_past here: both still point at "right
                    // after FUNC_START" (where the rollback left context), so a second
                    // correction attempt on the same call rolls back to exactly the
                    // same position with a matching recurrent state.  The turn-end
                    // block overwrites the slot with the final state when this turn
                    // completes.  After the anchor re-decode (rc 2) the slot is still
                    // stale/absent for the target even though the live state was
                    // rebuilt -- clear the flag so a second correction takes the
                    // anchor path again instead of restoring a stale slot (rc 0
                    // leaves the slot valid, as it already was).
                    state_.has_tool_correction_checkpoint = (rollback_rc == 0);
                    // The corrected call passed validate_tool_call, which mirrors
                    // the execution checks, so it cannot come back malformed: this
                    // call's correction attempt is consumed, and a future malformed
                    // call in this chain gets a fresh attempt.
                    state_.correction_attempted_this_turn = false;

                    // Hand off to process_tool_call -- it executes normally from here.
                    if (process_tool_call()) {
                        continue;
                    }                } else {
                    diag("System: Correction failed to produce valid tool call. Ejecting to prompt.", "\033[1;31m");

                    // Eject WITHOUT rolling back: a rollback would leave the
                    // context at a weird place (right after FUNC_START, mid
                    // tool call with no body), and the rollback itself is the
                    // unstable part (a failed seq_rm falls back to the
                    // prompt-anchor suffix re-decode, or aborts the turn if
                    // no anchor exists).  The failed correction stays in
                    // the LLM's history; it recovers on the next turn.  The
                    // checkpoint slot is deliberately left untouched: this path
                    // falls through to the prompt-return block, which
                    // overwrites it in place (final state) and resets the
                    // index.  has_tool_correction_checkpoint is left as the
                    // hook set it (a /continue re-correction without a pushed
                    // slot takes the prompt-anchor suffix re-decode path in
                    // the rollback instead of restoring a missing entry).
                    // MTP is untouched: no rollback, mirror in lockstep.
                    state_.conversation_text.clear();  // fed correction prompt is not in it; rebuild from tracker
                    state_.auto_continue = false;
                    // Returning control to the user prompt: the correction latch is only
                    // meaningful within one auto-continue chain (step 9 clears it on the
                    // fall-through).
                    state_.correction_attempted_this_turn = false;
                }
            } else {
                // The correction prompt (full system prompt) itself doesn't fit in the
                // remaining context, so no correction is possible without exceeding the
                // limit. Eject to the prompt with an explanation rather than failing silently.
                diag("System: Tool correction aborted: correction prompt does not fit in remaining context (" + std::to_string(correction_tokens.size()) + " tokens needed, " + std::to_string(cparams_.n_ctx - n_past_) + " available). Type '/clear' to reset.", "\033[1;33m");
                // Returning control to the user prompt: clear the correction latch
                // (step 9 would do it on a normal prompt return, but this path
                // `continue`s past it) so the next turn gets a fresh correction attempt.
                state_.correction_attempted_this_turn = false;
            }
            continue;
        }

        // 9. Log assistant output (skip if already logged via process_tool_call)
        if (!state_.auto_continue && !assistant_logged_this_turn_ && !gen_result.text.empty()) log_entry("ASSISTANT", gen_result.text);

        // Reset per-turn failure state before returning to the user prompt.
        // Correction state is only meaningful within a single auto-continue
        // chain; once the user gets control back, it starts fresh.
        // tool_interrupt_pending is NOT cleared here: it is the /continue
        // handoff (consumed by the next generate_response as
        // was_mid_tool_call_).  Staleness is covered by the clears in
        // feed_user_message (new prompt) and reset_session_state (/clear,
        // /reincarnate, restore failure).
        state_.correction_attempted_this_turn = false;

        // 10. Handle reincarnate completion
        if (handle_reincarnate_completion()) continue;

        // Record checkpoint at every prompt return for partial restore.
        // Only save once per user turn -- clear last_user_input_ after use so that
        // subsequent tool-call iterations within the same turn don't create duplicates.
        if (!last_user_input_.empty()) {
            state_.prompt_checkpoints.push_back({n_past_, last_user_input_});

            // Overwrite the tool-correction checkpoint with the post-generation
            // undo checkpoint in place, keeping the stack at exactly one entry
            // per prompt_checkpoint.  No pop needed -- same slot is reused.
            llama_memory_t mem = llama_get_memory(ctx_);
            if (state_.tool_correction_checkpoint_idx >= 0 &&
                state_.rs_checkpoint_saved_this_turn) {
                // Tool calls occurred this turn AND the slot was actually
                // pushed (lockstep FUNC_START): overwrite it in place with
                // the final state after all tool executions.
                llama_memory_rs_checkpoint_overwrite(mem, 0,
                    (uint32_t)state_.tool_correction_checkpoint_idx);
                state_.tool_correction_checkpoint_idx = -1;
                state_.rs_checkpoint_saved_this_turn = false;
            } else if (state_.tool_correction_checkpoint_idx >= 0) {
                // Tool calls occurred but the slot was never pushed (all
                // FUNC_STARTs were non-lockstep): push a fresh checkpoint so
                // the "one entry per prompt_checkpoint" invariant holds.
                llama_memory_rs_checkpoint_save(mem, 0);
                state_.tool_correction_checkpoint_idx = -1;
                state_.rs_checkpoint_saved_this_turn = false;
            } else {
                // No tool calls this turn: push a fresh undo checkpoint.
                llama_memory_rs_checkpoint_save(mem, 0);
            }

            if (gen_result.was_interrupted && gen_result.ctx_limit_interrupt) {
                // 90% context housekeeping: the checkpoint just pushed is
                // PERMANENT -- a /undo target at the 90% position, labeled
                // with this turn's prompt -- so it is NOT marked provisional.
                // Auto-resume the turn without dropping to the prompt: the
                // batch still holds the last row's logits, so the next
                // generate_response() picks up exactly where this one paused
                // (as with Ctrl-C + /continue) and the LLM never sees the
                // break.
                if (gen_result.ctx_limit_feed_crossing) {
                    // Feed-boundary crossing: the force-end is a 0-token turn
                    // end at the prompt boundary, before the LLM produced
                    // anything.  Label the boundary checkpoint (just pushed)
                    // with the "/continue" placeholder -- /undo'ing to it
                    // resumes this turn's response from the boundary -- and
                    // record it in readline history like any user input.
                    // Keep last_user_input_ as this turn's prompt so the
                    // resumed turn's end checkpoint is labeled with it.
                    state_.prompt_checkpoints.back().prompt = CTX_LIMIT_TURN_END_LABEL;
                    save_history_safe(".lim_history", CTX_LIMIT_TURN_END_LABEL);
                    int hist_before = history_length;
                    add_history(CTX_LIMIT_TURN_END_LABEL);
                    if (history_length > hist_before) c_count_since_restore_++;
                    state_.prev_was_interrupted = false;
                    state_.auto_continue = true;
                    state_.auto_continue_depth_val = 0;
                } else {
                    // Mid-generation crossing: label the resumed turn's
                    // completion with the "/continue" placeholder (record it
                    // in readline history as a C entry too, like any user
                    // input).  The user can later /undo to either the 90%
                    // point (this turn's prompt) or the real turn end
                    // ("/continue").
                    save_history_safe(".lim_history", CTX_LIMIT_TURN_END_LABEL);
                    int hist_before = history_length;
                    add_history(CTX_LIMIT_TURN_END_LABEL);
                    if (history_length > hist_before) c_count_since_restore_++;
                    last_user_input_ = CTX_LIMIT_TURN_END_LABEL;
                    state_.prev_was_interrupted = false;
                    state_.auto_continue = true;
                    state_.auto_continue_depth_val = 0;
                }
            } else {
                // An interrupted return leaves the context mid-turn: the checkpoint
                // just pushed is provisional until the user either /continue-resumes
                // the turn (the completion overwrites it in place, below) or types a
                // new prompt (finalizes it at the interrupt position, step 6).
                if (gen_result.was_interrupted) {
                    state_.interrupted_checkpoint_idx = (int)state_.prompt_checkpoints.size() - 1;
                }

                last_user_input_.clear();
            }
        } else if (state_.interrupted_checkpoint_idx >= 0 &&
                   state_.interrupted_checkpoint_idx < (int)state_.prompt_checkpoints.size() &&
                   !gen_result.was_interrupted) {
            // /continue resumed the interrupted turn and it has now ended (EOG or
            // ejection): move the provisional interrupt checkpoint to the turn
            // end in place -- the same slot reuse as the tool-correction
            // overwrite above, so the stack keeps one entry per prompt_checkpoint.
            // The label (prompt text) is unchanged; only the position and the
            // R/S state advance.
            PromptCheckpoint& cp = state_.prompt_checkpoints[state_.interrupted_checkpoint_idx];
            cp.n_past = n_past_;
            llama_memory_t mem = llama_get_memory(ctx_);
            int stack_idx = state_.interrupted_checkpoint_idx - state_.checkpoint_stack_offset;
            if (stack_idx >= 0) {
                llama_memory_rs_checkpoint_overwrite(mem, 0, (uint32_t)stack_idx);
            }
            state_.interrupted_checkpoint_idx = -1;
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
    const string& system_prompt_text,
    bool use_dummy_thought,
    SessionState& state
) {
    ChatSession session(ctx, vocab, smpl, batch, n_past, cparams,
                        system_tokens, system_prompt_text, use_dummy_thought, state);
    return session.run();
}
