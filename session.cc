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
#include <sys/stat.h>
#include <unistd.h>

// --- Readline Headers ---
#include <readline/readline.h>
#include <readline/history.h>

using namespace std;
using namespace Tokens;

// Forward declarations for functions defined in main.cc
extern void diag(const string& msg, const char* color);
extern bool is_debug;
extern bool first_prompt_displayed;
extern ofstream chat_log;
extern ofstream token_log;

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
        if (!key.empty()) {
            aliases[key] = value;
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

// --- Helper to unescape tags passed by the LLM ---
static void replace_all_tags(string& str, const string& from, const string& to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
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


// --- Helper to escape HTML special characters ---
static string html_escape(const string& s) {
    string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '&') out += "&amp;";
        else if (c == '<') out += "&lt;";
        else if (c == '>') out += "&gt;";
        else if (c == '"') out += "&quot;";
        else out += c;
    }
    return out;
}

// --- Helper to log a batch of tokens with a label ---
static void log_tokens(const string& label, const vector<llama_token>& toks, llama_context* ctx) {
    if (!is_debug || !token_log.is_open()) return;
    for (llama_token t : toks) {
        string piece = common_token_to_piece(ctx, t);
        token_log << label << " " << t << " \"" << escape_token_piece(piece) << "\"\n";
    }
    token_log.flush();
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
    cout << "\033[0m[" << msg << "]\033[0m" << endl;
    if (chat_log.is_open()) {
        chat_log << "[" << msg << "]" << "\n\n";
        chat_log.flush();
    }
}

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
    auto tokenize = [&](string text) { return common_tokenize(ctx, text, false, true); };

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

    // Persistent state for mid-tool-call resume: saves partial tool text from KV cache.
    // (Now stored in state.partial_tool_text)

    // Feed a vector of tokens into the KV cache, batching as needed.
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
    auto reset_llm_state = [&]() {
        state.clean_files.clear();
        state.loop_guard.clear_history();
        state.invalid_tool_strikes = 0;
        llama_sampler_reset(smpl);
    };

    // Reset all session-level state variables (full reset for clear/reincarnate).
    auto reset_session_state = [&]() {
        reset_llm_state();
        NetworkTools().reset_search();
        NetworkTools::reset_context_usage();
        g_browser_warning_suppressed = false;
        state.partial_tool_text.clear();
    };

    const char* history_file = ".lllm_history";
    load_history_safe(history_file);

    // Load user-defined aliases from ~/.lllm_aliases
    map<string, string> aliases = load_aliases();

    // Persistent state across loop iterations for "continue" resume feature.

    // --- MAIN CHAT TURN LOOP ---
    while (true) {
        string user_input = "";
        stop_generation = 0;
        g_was_interrupted = 0;

        if (!state.auto_continue) {
            // Print Speed from previous generation right before >>> (skip first turn)
            if (state.first_turn_done && state.last_t_count > 0) {
                double context_percent = (state.last_n_past / (double)cparams.n_ctx) * 100.0;
                ostringstream oss;
                oss << fixed << setprecision(1);
                oss << "Speed: " << (state.last_t_count / state.last_elapsed) << " t/s | Context: " << state.last_n_past << "/" << cparams.n_ctx << " (" << context_percent << "%) | Elapsed: " << state.last_elapsed << "s";
                string speed_line = oss.str();
                diag_speed_impl(speed_line);
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
                    g_was_interrupted = 0;
                    console("\r\033[K");
                    consoleFlush();
                    break;
                }

                string line(input_c);
                free(input_c);

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

        // Alias expansion: if user_input matches an alias key, replace with its value (single-level only).
        {
            auto alias_it = aliases.find(user_input);
            if (alias_it != aliases.end()) {
                user_input = alias_it->second;
            }
        }

        if (user_input == "quit" || user_input == "exit") return false;

        if (user_input == "clear") {
            clear_context();
            state.auto_continue = false;
            state.prev_was_interrupted = false;
            reset_session_state();
            state.last_t_count = 0;
            state.last_elapsed = 0.0;
            state.last_n_past = 0;
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

            string userprompt_check_path = string(HOME) + "/userprompt";
            {
                ofstream userprompt_clear(userprompt_check_path, ios::trunc);
                if (!userprompt_clear.is_open()) {
                    diag("Warning: Cannot truncate " + userprompt_check_path + " before reincarnate.", "\033[33m");
                }
            }

            diag("Sending reincarnate request to LLM...", "\033[35m");
            log_entry("USER", "[reincarnate] " + reincarnate_text);

            string reincarnate_message;
            if (state.prev_was_interrupted) {
                reincarnate_message = string(TURN_END) + "\n" + string(TURN_START) + "user\n" + reincarnate_text + TURN_END + "\n" + TURN_START + "assistant\n";
            } else {
                reincarnate_message = string(TURN_START) + "user\n" + reincarnate_text + TURN_END + "\n" + TURN_START + "assistant\n";
            }
            state.prev_was_interrupted = false;
            vector<llama_token> reincarnate_tokens = tokenize(reincarnate_message);

            if (n_past + (int)reincarnate_tokens.size() >= (int)cparams.n_ctx) {
                string ctx_diag = context_limit_diag(n_past, state.last_n_past, (size_t)reincarnate_tokens.size(), (int)cparams.n_ctx);
                diag("Context Limit Reached! Cannot process reincarnate" + ctx_diag + ". Type 'clear' to reset.", "\033[31m");
                continue;
            }

            if (!feed_tokens(reincarnate_tokens)) {
                if (stop_generation) stop_generation = 0;
                continue;
            }

            // Log reincarnate tokens to token_log when debug is enabled
            log_tokens("FEED USER_INPUT", reincarnate_tokens, ctx);

            state.auto_continue = true;
            state.reincarnate_mode = true;
            reset_session_state();
            continue;
        }

        if (user_input.empty() && !state.auto_continue) continue;

        // Handle "continue" as a reserved word to resume generation after an interruption.
        if (user_input == "continue") {
            if (state.tool_interrupt_pending) {
                // Interrupted mid-tool-call -- assistant turn is already open with partial
                // tool XML in the KV cache. Resume without feeding any additional tokens
                // so the LLM continues generating from exactly where it left off.
                state.prev_was_interrupted = false;
                diag("Resuming after tool interruption...", "\033[1;33m");

                state.auto_continue = true;
                state.auto_continue_depth_val = 0;
                user_input = ""; // Clear so it's not processed as a regular message
            } else if (state.prev_was_interrupted) {
                // Interrupted during normal text generation -- resume seamlessly without
                // feeding any additional tokens so the LLM doesn't even know it was interrupted.
                state.prev_was_interrupted = false;
                diag("Resuming generation...", "\033[1;33m");

                state.auto_continue = true;
                state.auto_continue_depth_val = 0;
                user_input = ""; // Clear so it's not processed as a regular message
            } else {
                // No active interruption -- just treat "continue" as a regular user message.
                // Fall through to normal processing below.
            }
        }

        bool browser_connected = check_browser_connected();

        static string prev_tty = ttyname(STDIN_FILENO) ? string(ttyname(STDIN_FILENO)) : "";
        const char* cur_tty = ttyname(STDIN_FILENO);
        if (cur_tty && !prev_tty.empty() && prev_tty != cur_tty) {
            system("reset");
        }
        if (cur_tty) prev_tty = cur_tty;

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

        if (!user_input.empty()) {
            // If user provides regular input (not "continue"), clear any pending tool interrupt state.
            if (!state.auto_continue) state.tool_interrupt_pending = false;

            if (!state.auto_continue) {
                log_entry("USER", user_input);
                if (should_output_to_browser() && pipe_fd >= 0) {
                    string user_html = "\n\n<div style=\"color: #79c0ff;\"><pre><code>" + html_escape(user_input) + "</code></pre></div>\n\n";
                    stream_html(user_html);
                }
            }

            string user_message;
            string turn_close = state.prev_was_interrupted ? (string(TURN_END) + "\n") : "";
            state.prev_was_interrupted = false;

            if (use_dummy_thought) {
                user_message = turn_close + string(TURN_START) + "user\n" +
                               user_input + TURN_END + "\n" +
                               TURN_START + "assistant\n"+THINK_START+"\nThe user wants a direct answer. I will output the requested data immediately without preamble.\n"+THINK_END+"\n";
            } else {
                user_message = turn_close + string(TURN_START) + "user\n" + user_input + TURN_END + "\n" + TURN_START + "assistant\n";
            }
            vector<llama_token> tokens = tokenize(user_message);

            if (n_past + (int)tokens.size() >= (int)cparams.n_ctx) {
                string ctx_diag = context_limit_diag(n_past, state.last_n_past, (size_t)tokens.size(), (int)cparams.n_ctx);
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

            // Log user input tokens to token_log when debug is enabled
            log_tokens("FEED USER_INPUT", tokens, ctx);
        }

        if (!state.auto_continue) {
            console("\n\033[1;90m--- Generating (Ctrl+C to interrupt) ---\033[0m\n");
            consoleFlush();
            stream("\n\n<div class='generation-start'>-- Generating --</div>\n\n");
        }

        static int g_auto_continue_depth = 0;
        if (!state.auto_continue) g_auto_continue_depth = 0;
        bool allow_continue_resume = false;

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
        int max_auto_continue = (max_auto_env != nullptr && strlen(max_auto_env) > 0) ? atoi(max_auto_env) : DEFAULT_MAX_AUTO_CONTINUE;
        if (max_auto_continue < 5) max_auto_continue = DEFAULT_MAX_AUTO_CONTINUE;

        static bool prev_stdout_ended_with_newline = false;

        // When resuming mid-tool-call via state.tool_interrupt_pending, initialize tracking
        // variables to reflect that we are already inside a tool call.
        bool was_mid_tool_call = state.tool_interrupt_pending;
        state.tool_interrupt_pending = false;

        auto start = chrono::high_resolution_clock::now();

        // --- TOKEN GENERATION via TokenGenerator class ---
        TokenGenerator tg(ctx, vocab, smpl, batch, n_past, cparams,
                          turn_timeout_sec, was_mid_tool_call, state.last_n_past);
        auto gen_result = tg.generate();

        string generated_text = gen_result.text;
        int t_count = gen_result.token_count;
        bool trigger_tool_execution = gen_result.has_tool_call;
        size_t tool_start = gen_result.tool_start;
        size_t tool_end = gen_result.tool_end;

        // Handle mid-tool-call state saving (regardless of exit reason)
        if (gen_result.tool_start != string::npos && gen_result.tool_end == string::npos) {
            state.tool_interrupt_pending = true;
            if (was_mid_tool_call) {
                state.partial_tool_text = state.partial_tool_text + generated_text;
            } else {
                state.partial_tool_text = generated_text.substr(gen_result.tool_start);
            }
        }

        // Update session state based on exit reason
        if (gen_result.was_interrupted) {
            state.prev_was_interrupted = true;
            state.auto_continue = false;
            state.reincarnate_mode = false;
        } else if (gen_result.early_exit) {
            state.auto_continue = false;
            state.reincarnate_mode = false;
        } else if (!gen_result.has_tool_call) {
            // Normal EOG
            state.auto_continue = false;
        }

        auto end = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(end - start).count();

        bool stdout_ended_with_newline = prev_stdout_ended_with_newline;
        if (stdout_ended_with_newline || !generated_text.empty()) {
            stdout_ended_with_newline = true;
        }
        if (!stdout_ended_with_newline) {
            cout << "\n";
        }

        state.last_t_count = t_count;
        state.last_elapsed = elapsed;
        state.last_n_past = n_past;
        state.first_turn_done = true;

        prev_stdout_ended_with_newline = true;

        if (stop_generation) {
            stop_generation = 0;
            state.auto_continue = false;
        }

        // --- TOOL EXECUTION BLOCK ---
        if (trigger_tool_execution && tool_start != string::npos && tool_end != string::npos) {
            // When resuming from a mid-tool-call interrupt, prepend the saved partial text
            // so the extracted tool_call contains the complete XML including FUNC_START.
            string full_generated = generated_text;
            if (!state.partial_tool_text.empty()) {
                full_generated = state.partial_tool_text + full_generated;
                state.partial_tool_text.clear();
                // tool_end was found within generated_text, but full_generated now has
                // state.partial_tool_text prepended. Adjust tool_end to be relative to full_generated.
                // Re-search for FUNC_END in full_generated starting from tool_start to get the correct offset.
                // On resume (tool_start==0), search from the actual FUNC_START position inside
                // state.partial_tool_text to avoid double-counting it (depth would go 1->2 and never return).
                size_t resume_search_from = was_mid_tool_call
                    ? full_generated.find(FUNC_START)
                    : tool_start;
                tool_end = find_tool_end_robust(full_generated, resume_search_from);
                if (tool_end != string::npos) {
                    size_t exact_pos = full_generated.find(FUNC_END, resume_search_from);
                    if (exact_pos == string::npos) {
                        repair_malformed_tool_end(full_generated, tool_end);
                        tool_end = full_generated.find(FUNC_END, resume_search_from);
                    }
                }
            }

            string tool_call = full_generated.substr(tool_start, tool_end - tool_start + string(FUNC_END).length());
            string preamble = "";
            if (tool_start > 0) preamble = generated_text.substr(0, tool_start);

            const vector<string> strip_tags_vec = {TURN_START, TURN_END};
            strip_tags(tool_call, strip_tags_vec);

            ToolResult tool_out;
            bool was_loop = false;
            bool tool_blocked_by_loop = false;
            bool abort_auto = false;
            bool inject_auto_user_msg = false;
            string active_intervention_msg = "";

            // Pre-execution loop guard check.
            bool pre_loop = state.loop_guard.would_repeat(tool_call);
            if (pre_loop) {
                was_loop = state.loop_guard.record_and_check(tool_call);
                tool_blocked_by_loop = true;
                int current_strikes = state.loop_guard.get_loop_strikes();

                active_intervention_msg = get_next_loop_message();
                tool_out.content = "System Error: Loop Detected -- you already called this exact tool recently. " + active_intervention_msg + " If searching code, use search_file instead of exec_shell.";
                tool_out.display = tool_out.content;

                diag("System: Pre-execution loop blocked (Strike " + std::to_string(current_strikes) + ").", "\033[35m");

                int max_attempts = loopMessages.size();
                if (current_strikes <= max_attempts) {
                    diag("System: Automating intervention (Attempt " + std::to_string(current_strikes) + "/" + std::to_string(max_attempts) + ").", "\033[35m");
                    inject_auto_user_msg = true;
                } else {
                    diag("System: Circuit breaker -- intervention failed after " + std::to_string(max_attempts) + " strikes. Ejecting to prompt.", "\033[1;31m");
                    abort_auto = true;
                }
            } else {
                // Execute the tool.
                tool_out = execute_tool_call(tool_call, state.clean_files);

                // Handle validation errors reported by the struct.
                if (!tool_out.recognized || !tool_out.params_valid) {
                    state.invalid_tool_strikes++;

                    if (is_debug && should_show_tools() && should_output_to_browser()) {
                        string raw_display = tool_call;
                        if (raw_display.length() > 500) raw_display = raw_display.substr(0, 500) + "...";
                        string safe = html_escape(raw_display);
                        string label = !tool_out.recognized ? "Invalid Tool Call" : "Malformed Tool Call";
                        string error_html = "\n\n<div class='tool-error'>" + label + " (Strike " + std::to_string(state.invalid_tool_strikes) + "):<pre><code>" + safe + "</code></pre></div>\n\n";
                        stream_tool_result(error_html);

                        if (state.invalid_tool_strikes >= 5) {
                            diag("System: " + std::to_string(state.invalid_tool_strikes) + " consecutive invalid tool calls. Intervention failed, ejecting to prompt.", "\033[1;31m");
                            abort_auto = true;
                        } else if (state.invalid_tool_strikes >= 2) {
                            diag("System: " + std::to_string(state.invalid_tool_strikes) + " consecutive invalid tool calls. Injecting intervention.", "\033[1;31m");
                            inject_auto_user_msg = true;
                            active_intervention_msg = SYSTEM_PROMPT_REMINDER;
                        }
                    }
                } else {
                    state.invalid_tool_strikes = 0;

                    was_loop = state.loop_guard.record_and_check(tool_call);

                    if (stop_generation) {
                        diag("Tool Interrupted by User", "\033[31m");
                        stop_generation = 0;
                        allow_continue_resume = true;
                        state.reincarnate_mode = false;
                    }

                    if (!abort_auto && was_loop) {
                        active_intervention_msg = get_next_loop_message();
                        tool_out.content = "System Warning: You just repeated a tool call. " + active_intervention_msg + " If searching code, use search_file instead of exec_shell.";
                        tool_out.display = tool_out.content;

                        diag("System: Post-execution loop warning (Strike 2).", "\033[35m");
                        inject_auto_user_msg = true;
                    } else if (!abort_auto) {
                        if (tool_out.is_mutating && !tool_out.is_error) state.loop_guard.clear_history();
                        if (tool_out.is_mutating && tool_out.is_expected_error) state.loop_guard.clear_history();
                    }
                }
            }

            if (!abort_auto) {
                if (is_debug) {
                    console("\n\033[92m[Tool Result]\033[0m\n");
                    string result_to_print = tool_out.display;
                    size_t p = 0;
                    while ((p = result_to_print.find('\n')) != string::npos) {
                        console("  ", (int)p, result_to_print.c_str(),"\n");
                        result_to_print.erase(0, p + 1);
                    }
                    if (!result_to_print.empty()) console("  ", result_to_print.c_str(),"\n");
                }

                string display_for_browser = tool_out.display;

                string safe_result = html_escape(display_for_browser);

                string result_html = "\n\n<div class='tool-result'>Tool Result:<pre><code>" + safe_result + "</code></pre></div>\n\n";
                stream_tool_result(result_html);
                consoleFlush();
                prev_stdout_ended_with_newline = true;

                chat_log << "\n";
                generated_text = "";
                string tool_result_section = string(TURN_START) + "user\n[Tool Result]\n" + sanitize(tool_out.content) + TURN_END + "\n";
                string tool_msg = tool_result_section;

                if (tool_blocked_by_loop || inject_auto_user_msg) {
                    string clean_user_turn = string(TURN_START) + "user\n[Tool Result]\n" + sanitize(tool_out.content);
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
                    double pct = (double)n_past / cparams.n_ctx * 100.0;
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.1f%%", pct);
                    diag("Tool result too large to fit in context (" + std::to_string(t_tokens.size()) + " tokens needed, " + std::to_string(cparams.n_ctx - n_past) + " available). Context usage: " + string(buf) + ".", "\033[1;33m");
                    state.auto_continue = false;
                } else if (!feed_tokens(t_tokens)) {
                    abort_auto = true;
                } else {
                    // Log tool result tokens to token_log when debug is enabled
                    log_tokens("FEED TOOL_RESULT", t_tokens, ctx);

                    g_auto_continue_depth++;
                    if (g_auto_continue_depth > max_auto_continue) {
                        diag("System: Max auto-continue depth reached (" + std::to_string(g_auto_continue_depth) + "/" + std::to_string(max_auto_continue) + "). LLM may be stuck in a loop. Ejecting to prompt.", "\033[1;31m");
                        state.auto_continue = false;
                    } else if (allow_continue_resume) {
                        // Interrupted during tool call -- feed result but drop to prompt.
                        // User can type "continue" to resume generation.
                        diag("Tool execution interrupted. Type 'continue' to let the LLM proceed, or provide input.", "\033[1;33m");
                        allow_continue_resume = false;
                        state.tool_interrupt_pending = true;
                        state.auto_continue = false;
                    } else {
                        state.auto_continue = true;
                        continue;
                    }
                }
            } else {
                state.auto_continue = false;
                generated_text = "";

                string abort_msg;
                if (state.invalid_tool_strikes >= 5) {
                    abort_msg = "System Error: You are generating malformed tool calls. Your XML schema is incorrect. Stop and carefully review the required format. Do NOT wrap tool calls in markdown code blocks or other formatting.";
                } else {
                    abort_msg = "System Error: Tool call blocked -- you are repeating yourself. Stop retrying and try a different approach (e.g., use search_file instead of exec_shell for code searches).";
                }
                string tool_result_section = string(TURN_START) + "user\n[Tool Result]\n" + abort_msg + TURN_END + "\n";
                string tool_msg = tool_result_section;

                vector<llama_token> t_tokens = tokenize(tool_msg);
                if (n_past + t_tokens.size() < cparams.n_ctx) {
                    feed_tokens(t_tokens);

                    // Log abort tool result tokens to token_log when debug is enabled
                    log_tokens("FEED TOOL_RESULT", t_tokens, ctx);
                }
            }
        }

        if (!state.auto_continue && !generated_text.empty()) log_entry("ASSISTANT", generated_text);

        // --- REINCARNATE POST-GENERATION HANDLING ---
        if (state.reincarnate_mode && !state.auto_continue) {
            state.reincarnate_mode = false;

            string userprompt_path = string(HOME) + "/userprompt";

            ifstream userprompt_file(userprompt_path);
            if (!userprompt_file.is_open()) {
                diag("Reincarnate failed: LLM did not write " + userprompt_path + ". Session will not be reincarnated.", "\033[31m");
                log_entry("SYSTEM", "Reincarnate failed: userprompt was not written by LLM");
                continue;
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
                continue;
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

            string new_session_message = string(TURN_START) + "user\n" + follow_prompt + TURN_END + "\n" + TURN_START + "assistant\n";
            vector<llama_token> new_session_tokens = tokenize(new_session_message);

            if (n_past + (int)new_session_tokens.size() >= (int)cparams.n_ctx) {
                string ctx_diag = context_limit_diag(n_past, state.last_n_past, (size_t)new_session_tokens.size(), (int)cparams.n_ctx);
                diag("Context Limit Reached! Cannot process reincarnated prompt" + ctx_diag + ".", "\033[31m");
                continue;
            }

            if (!feed_tokens(new_session_tokens)) {
                if (stop_generation) stop_generation = 0;
                diag("Failed to feed reincarnated session tokens. Type 'clear' to reset.", "\033[31m");
                continue;
            }

            // Log reincarnated session tokens to token_log when debug is enabled
            log_tokens("FEED USER_INPUT", new_session_tokens, ctx);

            state.auto_continue = true;
            reset_session_state();
            log_entry("SYSTEM", "Context Cleared and Reincarnated with New Prompt");
            continue;
        }
    }

    return true;
}
