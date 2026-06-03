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

// HOME is declared as extern std::string HOME in network.h

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

// Find tool call end robustly, handling malformed closing tags.
static size_t find_tool_end_robust(const string& text, size_t from_pos) {
    size_t pos = text.find(FUNC_END, from_pos);
    if (pos != string::npos) return pos;

    // Fallback: look for FUNC_END without trailing >, followed by garbage
    string fe_str(FUNC_END);
    string partial = fe_str.substr(0, fe_str.length() - 1);
    size_t p;
    while ((p = text.find(partial, from_pos)) != string::npos) {
        if (p + partial.length() < text.length()) {
            char next_c = text[p + partial.length()];
            if (next_c == '>') return p;
            size_t garbage_end = text.find('>', p + partial.length());
            if (garbage_end != string::npos) {
                return p + partial.length();
            }
        }
        from_pos = p + partial.length();
    }
    return string::npos;
}

// Repair malformed closing tag at position returned by find_tool_end_robust.
static void repair_malformed_tool_end(string& text, size_t pos) {
    if (pos >= text.length()) return;
    char next_c = text[pos];
    if (next_c == '>') return;
    size_t garbage_end = text.find('>', pos);
    if (garbage_end == string::npos) {
        text.insert(pos, ">");
        return;
    }
    text.replace(pos, garbage_end + 1 - pos, ">");
}

// --- Helper to strip thinking and tool-call XML tags from a string ---
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
    bool& auto_continue,
    bool& reincarnate_mode,
    bool& prev_was_interrupted,
    bool& first_turn_done,
    int& last_t_count,
    double& last_elapsed,
    int& last_n_past,
    set<string>& clean_files,
    LoopDetector& loop_guard,
    int& invalid_tool_strikes
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
        clean_files.clear();
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

    const char* history_file = ".lllm_history";
    load_history_safe(history_file);

    // --- MAIN CHAT TURN LOOP ---
    while (true) {
        string user_input = "";
        stop_generation = 0;
        g_was_interrupted = 0;

        if (!auto_continue) {
            // Print Speed from previous generation right before >>> (skip first turn)
            if (first_turn_done && last_t_count > 0) {
                double context_percent = (last_n_past / (double)cparams.n_ctx) * 100.0;
                ostringstream oss;
                oss << fixed << setprecision(1);
                oss << "Speed: " << (last_t_count / last_elapsed) << " t/s | Context: " << last_n_past << "/" << cparams.n_ctx << " (" << context_percent << "%) | Elapsed: " << last_elapsed << "s";
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

        if (user_input == "quit" || user_input == "exit") return false;

        if (user_input == "clear") {
            clear_context();
            auto_continue = false;
            prev_was_interrupted = false;
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
            if (prev_was_interrupted) {
                reincarnate_message = string(TURN_END) + "\n" + string(TURN_START) + "user\n" + reincarnate_text + TURN_END + "\n" + TURN_START + "assistant\n";
            } else {
                reincarnate_message = string(TURN_START) + "user\n" + reincarnate_text + TURN_END + "\n" + TURN_START + "assistant\n";
            }
            prev_was_interrupted = false;
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
            if (!auto_continue) {
                log_entry("USER", user_input);
                if (should_output_to_browser() && pipe_fd >= 0) {
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

            string user_message;
            string turn_close = prev_was_interrupted ? (string(TURN_END) + "\n") : "";
            prev_was_interrupted = false;

            if (use_dummy_thought) {
                user_message = turn_close + string(TURN_START) + "user\n" +
                               user_input + TURN_END + "\n" +
                               TURN_START + "assistant\n"+THINK_START+"\nThe user wants a direct answer. I will output the requested data immediately without preamble.\n"+THINK_END+"\n";
            } else {
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

        if (!auto_continue) {
            console("\n\033[1;90m--- Generating (Ctrl+C to interrupt) ---\033[0m\n");
            consoleFlush();
            stream("\n\n<div class='generation-start'>-- Generating --</div>\n\n");
        }

        static int g_auto_continue_depth = 0;
        if (!auto_continue) g_auto_continue_depth = 0;
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
        bool had_eog_recovery = false;
        bool context_warned_this_turn = false;

        static constexpr double DEFAULT_TURN_TIMEOUT_SEC = 60.0;
        const char* timeout_env = getenv("LLLM_TURN_TIMEOUT");
        double turn_timeout_sec = (timeout_env != nullptr && strlen(timeout_env) > 0) ? atof(timeout_env) : DEFAULT_TURN_TIMEOUT_SEC;
        if (turn_timeout_sec < 5.0) turn_timeout_sec = DEFAULT_TURN_TIMEOUT_SEC;

        static constexpr int DEFAULT_MAX_AUTO_CONTINUE = 500;
        const char* max_auto_env = getenv("LLLM_MAX_AUTO_CONTINUE");
        int max_auto_continue = (max_auto_env != nullptr && strlen(max_auto_env) > 0) ? atoi(max_auto_env) : DEFAULT_MAX_AUTO_CONTINUE;
        if (max_auto_continue < 5) max_auto_continue = DEFAULT_MAX_AUTO_CONTINUE;

        bool in_thinking_block = false;
        size_t think_start = string::npos;
        size_t think_end = string::npos;
        string think_buffer;
        bool think_buffering = true;

        static bool prev_stdout_ended_with_newline = false;

        // --- INNER TOKEN GENERATION LOOP ---
        while (true) {
            if (stop_generation) {
                diag("Task Interrupted by User", "\033[31m");
                stream("\n\n[Task Interrupted by User]\n\n");
                stop_generation = 0;
                g_was_interrupted = 0;
                prev_was_interrupted = true;
                auto_continue = false;
                reincarnate_mode = false;
                rl_redisplay();
                break;
            }

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
                reincarnate_mode = false;
                break;
            }

            if (batch.n_tokens < 1) {
                diag("Error: no tokens in batch to sample from", "\033[31m");
                auto_continue = false;
                reincarnate_mode = false;
                break;
            }
            llama_token next_token = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);

            if (llama_vocab_is_eog(vocab, next_token)) {
                size_t active_ts = generated_text.rfind(FUNC_START);
                size_t active_te = find_tool_end_robust(generated_text, 0);

                bool inside_unclosed_tool = (active_ts != string::npos && (active_te == string::npos || active_ts > active_te));

                static int eog_event_count = 0;
                static int total_poll_iters = 0;
                static int max_poll_iters = 0;
                static int last_printed = 0;
                int poll_iter_used = 0;

                static constexpr int DEFAULT_EOG_RESAMPLE_MAX = 30;
                static constexpr int EOG_RESAMPLE_HARD_CAP    = 200;

                const char* eog_env = getenv("LLLM_EOG_RESAMPLE_MAX");
                int max_iterations = (eog_env != nullptr && strlen(eog_env) > 0) ? atoi(eog_env) : DEFAULT_EOG_RESAMPLE_MAX;
                max_iterations = std::max(1, std::min(max_iterations, EOG_RESAMPLE_HARD_CAP));

                bool recovered = false;

                {
                    llama_token polled = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);
                    if (!llama_vocab_is_eog(vocab, polled)) {
                        next_token = polled;
                        recovered = true;
                        poll_iter_used = 1;
                    }
                }

                if (!recovered) {
                    for (int poll_iter = 0; poll_iter < max_iterations; ++poll_iter) {
                        if (stop_generation) break;
                        llama_token polled = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);
                        if (!llama_vocab_is_eog(vocab, polled)) {
                            next_token = polled;
                            recovered = true;
                            poll_iter_used = poll_iter + 2;
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

                    had_eog_recovery = true;
                }

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
                        auto_continue = false;
                    }
                    break;
                }
            }

            string token_str = common_token_to_piece(ctx, next_token).c_str();
            generated_text += token_str;
            full_response += token_str;

            if (think_start == string::npos) {
                think_start = generated_text.find(Tokens::THINK_START);
            }
            if (think_start != string::npos && think_end == string::npos) {
                think_end = generated_text.find(Tokens::THINK_END, think_start);
            }
            in_thinking_block = (think_start != string::npos && think_end == string::npos);

            if (tool_start == string::npos) {
                size_t search_from = func_search_pos;
                while (true) {
                    tool_start = generated_text.find(FUNC_START, search_from);
                    if (tool_start == string::npos) break;

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
                tool_end = find_tool_end_robust(generated_text, tool_start);
                if (tool_end != string::npos) {
                    size_t exact_pos = generated_text.find(FUNC_END, tool_start);
                    if (exact_pos == string::npos) {
                        repair_malformed_tool_end(generated_text, tool_end);
                        tool_end = generated_text.find(FUNC_END, tool_start);
                    }
                }
            }

            in_tool_call_stream = (tool_start != string::npos && tool_end == string::npos);

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

                tool_end = find_tool_end_robust(generated_text, tool_start);
                if (tool_end != string::npos) {
                    size_t exact_pos = generated_text.find(FUNC_END, tool_start);
                    if (exact_pos == string::npos) {
                        repair_malformed_tool_end(generated_text, tool_end);
                        tool_end = generated_text.find(FUNC_END, tool_start);
                    }
                }
                trigger_tool_execution = true;
                break;
            }

            if (tool_end != string::npos) {
                trigger_tool_execution = true;
                break;
            }

            if (!in_tool_call_stream && !in_thinking_block) {
                size_t safe_len = generated_text.length();
                string fstart(FUNC_START);
                string tstart(Tokens::THINK_START);

                if (think_start != string::npos && think_end != string::npos) {
                    size_t think_block_end = think_end + string(Tokens::THINK_END).length();

                    think_buffer.clear();
                    think_buffering = true;

                    if (print_pos < think_block_end) {
                        if (!think_buffering && print_pos <= think_end) {
                            string close_tag = generated_text.substr(print_pos, think_block_end - print_pos);
                            console_think(close_tag.c_str());
                            consoleThinkFlush();
                        }
                        print_pos = think_block_end;
                    }
                }

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
                size_t safe_len = generated_text.length();
                string tstart(Tokens::THINK_START);
                string tend(Tokens::THINK_END);

                for (size_t len = 1; len <= tend.length() && len <= generated_text.length(); ++len) {
                    if (generated_text.compare(generated_text.length() - len, len, tend, 0, len) == 0) {
                        safe_len = generated_text.length() - len;
                        break;
                    }
                }

                if (safe_len > print_pos) {
                    string think_output = generated_text.substr(print_pos, safe_len - print_pos);

                    if (think_buffering) {
                        _strip_think_and_tool_tags(think_output);

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
                            think_buffer.clear();
                            think_output = think_output.substr(content_start);
                            console_think(think_output.c_str());
                            consoleThinkFlush();
                            think_buffering = false;
                        } else {
                            think_buffer += think_output;
                        }
                    } else {
                        _strip_think_and_tool_tags(think_output);
                        console_think(think_output.c_str());
                        consoleThinkFlush();
                    }
                    print_pos = safe_len;
                }
            } else {
                if (!unprinted_text.empty()) {
                    console(unprinted_text.c_str());
                    consoleFlush();
                    stream(unprinted_text);
                    unprinted_text = "";
                }
                print_pos = generated_text.length();
            }

            t_count++;

            if (t_count % 50 == 0 && !in_tool_call_stream && !in_thinking_block) {
                cerr << "\r\033[K[Generating... " << t_count << " tokens]";
                cerr.flush();
            }

            {
                static int recent_token_count = 0;
                static auto last_rate_check = chrono::high_resolution_clock::now();
                recent_token_count++;

                auto now = chrono::high_resolution_clock::now();
                double elapsed_window = chrono::duration<double>(now - last_rate_check).count();
                if (elapsed_window >= 10.0) {
                    double rate = recent_token_count / elapsed_window;
                    if (rate < 0.5 && t_count > 20) {
                        diag("System: Token generation rate critically low (" +
                             to_string((int)(rate * 10) / 10.0) + " tok/s). Possible stall detected.", "\033[1;33m");
                    }
                    recent_token_count = 0;
                    last_rate_check = now;
                }
            }

            {
                auto now = chrono::high_resolution_clock::now();
                double elapsed_turn = chrono::duration<double>(now - start).count();
                if (elapsed_turn >= turn_timeout_sec) {
                    diag("System: Turn timeout reached (" + std::to_string((int)elapsed_turn) + "s/" + std::to_string((int)turn_timeout_sec) + "s). LLM may be stuck in a hallucination loop. Forcing stop.", "\033[1;33m");
                    auto_continue = false;
                    reincarnate_mode = false;
                    break;
                }
            }

            batch.n_tokens = 0;
            common_batch_add(batch, next_token, n_past++, {0}, true);
            if (!handle_llama_decode_error(ctx, batch)) { sync_n_past(ctx, n_past); reincarnate_mode = false; break; }
        } // END INNER TOKEN LOOP

        cerr << "\r\033[K";

        auto end = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(end - start).count();

        bool stdout_ended_with_newline = prev_stdout_ended_with_newline;
        if (!unprinted_text.empty()) {
            console(unprinted_text.back() != '\n' ? (unprinted_text + "\n").c_str() : unprinted_text.c_str());
            consoleFlush();
            stream(unprinted_text + (unprinted_text.back() != '\n' ? "\n" : ""));
            stdout_ended_with_newline = true;
            unprinted_text = "";
        }

        if (!stdout_ended_with_newline) {
            cout << "\n";
        }

        last_t_count = t_count;
        last_elapsed = elapsed;
        last_n_past = n_past;
        first_turn_done = true;

        prev_stdout_ended_with_newline = true;

        if (stop_generation) {
            stop_generation = 0;
            auto_continue = false;
        }

        // --- TOOL EXECUTION BLOCK ---
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

                if (should_show_tools() && should_output_to_browser()) {
                    string raw_display = tool_call;
                    if (raw_display.length() > 500) {
                        raw_display = raw_display.substr(0, 497) + "...";
                    }
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

                if (invalid_tool_strikes >= 5) {
                    diag("System: " + std::to_string(invalid_tool_strikes) + " consecutive invalid tool calls. Intervention failed, ejecting to prompt.", "\033[1;31m");
                    abort_auto = true;
                } else if (invalid_tool_strikes >= 2) {
                    diag("System: " + std::to_string(invalid_tool_strikes) + " consecutive invalid tool calls. Injecting intervention.", "\033[1;31m");
                    inject_auto_user_msg = true;
                    active_intervention_msg = SYSTEM_PROMPT_REMINDER;
                }
            } else {
                if (!validate_tool_params(tool_name, tool_call)) {
                    invalid_tool_strikes++;

                    if (is_debug && should_show_tools() && should_output_to_browser()) {
                        string raw_display = tool_call;
                        if (raw_display.length() > 500) {
                            raw_display = raw_display.substr(0, 497) + "...";
                        }
                        string safe;
                        for (char c : raw_display) {
                            if (c == '&') safe += "&amp;";
                            else if (c == '<') safe += "&lt;";
                            else if (c == '>') safe += "&gt;";
                            else safe += c;
                        }
                        string error_html = "\n\n<div class='tool-error'>Malformed Tool Call (Strike " + std::to_string(invalid_tool_strikes) + "):<pre><code>" + safe + "</code></pre></div>\n\n";
                        stream_tool_result(error_html);
                    }

                    string schema_hint = string(PARAM_START) + "param_name>value</" + string(PARAM_END);
                    tool_result = "System Error: Malformed tool call. Required parameter tags are missing from your XML. You MUST use the strict schema: <function=TOOL_NAME>" + schema_hint + "...";
                    display_result = tool_result;

                    diag("System: Malformed tool call -- missing required parameter tags (Strike " + std::to_string(invalid_tool_strikes) + ").", "\033[1;31m");

                    if (invalid_tool_strikes >= 5) {
                        diag("System: " + std::to_string(invalid_tool_strikes) + " consecutive malformed tool calls. Intervention failed, ejecting to prompt.", "\033[1;31m");
                        abort_auto = true;
                    } else if (invalid_tool_strikes >= 2) {
                        diag("System: " + std::to_string(invalid_tool_strikes) + " consecutive malformed tool calls. Injecting intervention.", "\033[1;31m");
                        inject_auto_user_msg = true;
                        active_intervention_msg = SYSTEM_PROMPT_REMINDER;
                    }
                } else {
                    invalid_tool_strikes = 0;
                    bool is_mutating_tool = (tool_name == "edit_file" || tool_name == "write_file");

                    bool pre_loop = loop_guard.would_repeat(tool_call);

                    if (pre_loop) {
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
                    } else {
                        tool_result = execute_tool_call(tool_call, clean_files);

                        was_loop = loop_guard.record_and_check(tool_call);

                        if (stop_generation) {
                            diag("Tool Interrupted by User", "\033[31m");
                            stop_generation = 0;
                            abort_auto = true;
                            reincarnate_mode = false;
                        }

                        if (!abort_auto && was_loop) {
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

                string display_for_browser = display_result;
                if (!is_debug && display_for_browser.length() > 500) {
                    display_for_browser = "..." + display_for_browser.substr(display_for_browser.length() - 497);
                }

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
                prev_stdout_ended_with_newline = true;

                chat_log << "\n";
                generated_text = ""; unprinted_text = "";
                tool_start = string::npos; tool_end = string::npos;
                think_start = string::npos; think_end = string::npos;
                in_tool_call_stream = false; in_thinking_block = false;
                think_buffer.clear();
                think_buffering = true;

                string tool_result_section = string(TURN_START) + "user\n[Tool Result]\n" + sanitize(tool_result) + TURN_END + "\n";
                string tool_msg = tool_result_section;

                if (tool_blocked_by_loop || inject_auto_user_msg) {
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
                    double pct = (double)n_past / cparams.n_ctx * 100.0;
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.1f%%", pct);
                    diag("Tool result too large to fit in context (" + std::to_string(t_tokens.size()) + " tokens needed, " + std::to_string(cparams.n_ctx - n_past) + " available). Context usage: " + string(buf) + ".", "\033[1;33m");
                    auto_continue = false;
                } else if (!feed_tokens(t_tokens)) {
                    abort_auto = true;
                } else {
                    g_auto_continue_depth++;
                    if (g_auto_continue_depth > max_auto_continue) {
                        diag("System: Max auto-continue depth reached (" + std::to_string(g_auto_continue_depth) + "/" + std::to_string(max_auto_continue) + "). LLM may be stuck in a loop. Ejecting to prompt.", "\033[1;31m");
                        auto_continue = false;
                    } else {
                        auto_continue = true;
                        continue;
                    }
                }
            } else {
                auto_continue = false;
                generated_text = ""; unprinted_text = "";

                string abort_msg;
                if (invalid_tool_strikes >= 5) {
                    abort_msg = "System Error: You are generating malformed tool calls. Your XML schema is incorrect. Stop and carefully review the required format. Do NOT wrap tool calls in markdown code blocks or other formatting.";
                } else {
                    abort_msg = "System Error: Tool call blocked -- you are repeating yourself. Stop retrying and try a different approach (e.g., use search_file instead of exec_shell for code searches).";
                }
                string tool_result_section = string(TURN_START) + "user\n[Tool Result]\n" + abort_msg + TURN_END + "\n";
                string tool_msg = tool_result_section;

                vector<llama_token> t_tokens = tokenize(tool_msg);
                if (n_past + t_tokens.size() < cparams.n_ctx) {
                    feed_tokens(t_tokens);
                }
            }
        }

        if (!auto_continue && !generated_text.empty()) log_entry("ASSISTANT", generated_text);

        // --- REINCARNATE POST-GENERATION HANDLING ---
        if (reincarnate_mode && !auto_continue) {
            reincarnate_mode = false;

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

    return true;
}
