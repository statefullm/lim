
#include "tool_executor.h"
#include "model.h"
#include "session_utils.h"
#include "output.h"
#include "signals.h"
#include "parsers.h"
#include "tokens.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cctype>
#include <algorithm>


// Forward declarations for functions defined in main.cc
extern void diag(const string& msg, const char* color);
extern bool is_debug;
extern ofstream chat_log;
extern ofstream token_log;

// Truncate large parameter values (OLD_TEXT, NEW_TEXT, TEXT, CONTENT) inside a
// tool call XML string so the raw output fits on one console line.
static string truncate_tool_call_params(const string& tool_call) {
    static const char* long_params[] = {"CONTENT", "OLD_TEXT", "NEW_TEXT", "TEXT"};
    string result = tool_call;
    for (const char* param : long_params) {
        string open_tag = string(PARAM_START) + param + string(PARAM_END);
        string close_tag = string("</") + param + ">";
        size_t pos = 0;
        while ((pos = result.find(open_tag, pos)) != string::npos) {
            pos += open_tag.length();
            size_t end = result.find(close_tag, pos);
            string replacement = "...";
            if (end != string::npos) {
                result.replace(pos, end - pos, "...");
                pos += 3;
            } else {
                // Unclosed param — truncate rest of string after a budget.
                if (pos + 40 < result.length()) {
                    result = result.substr(0, pos + 40) + "...";
                    break;
                }
            }
        }
    }
    // Hard cap: if still too long for one line, truncate the whole thing.
    static constexpr size_t MAX_DISPLAY_LEN = 200;
    if (result.length() > MAX_DISPLAY_LEN) {
        result = result.substr(0, MAX_DISPLAY_LEN) + "...";
    }
    return result;
}

ToolExecutor::Result ToolExecutor::execute(
    SessionState& state,
    string& generated_text,
    const string& full_generated,
    size_t tool_start,
    size_t tool_end,
    bool was_mid_tool_call,
    function<vector<llama_token>(string)> tokenize,
    function<bool(const vector<llama_token>&)> feed_tokens,
    llama_context* ctx,
    int& n_past,
    const llama_context_params& cparams,
    int& g_auto_continue_depth,
    int max_auto_continue,
    bool allow_continue_resume
) {
    Result result;

    // When resuming from a mid-tool-call interrupt, prepend the saved partial text
    // so the extracted tool_call contains the complete XML including FUNC_START.
    string full_gen = full_generated;
    if (!state.partial_tool_text.empty()) {
        full_gen = state.partial_tool_text + full_gen;
        state.partial_tool_text.clear();
        // tool_end was found within generated_text, but full_gen now has
        // state.partial_tool_text prepended. Adjust tool_end to be relative to full_gen.
        // Re-search for FUNC_END in full_gen starting from tool_start to get the correct offset.
        // On resume (tool_start==0), search from the actual FUNC_START position inside
        // state.partial_tool_text to avoid double-counting it (depth would go 1->2 and never return).
        size_t resume_search_from = was_mid_tool_call
            ? full_gen.find(FUNC_START)
            : tool_start;
        tool_end = find_tool_end_robust(full_gen, resume_search_from);
        if (tool_end != string::npos) {
            size_t exact_pos = full_gen.find(FUNC_END, resume_search_from);
            if (exact_pos == string::npos) {
                repair_malformed_tool_end(full_gen, tool_end);
                tool_end = full_gen.find(FUNC_END, resume_search_from);
            }
        }
    }

    string tool_call = full_gen.substr(tool_start, tool_end - tool_start + string(FUNC_END).length());

    string preamble = "";
    if (tool_start > 0) preamble = generated_text.substr(0, tool_start);

    vector<string> strip_tags_vec;
    // Strip full turn markers (user_start, assistant_start, turn_end).
    if (!g_model_tokens.user_turn_start.text.empty()) strip_tags_vec.push_back(g_model_tokens.user_turn_start.text);
    if (!g_model_tokens.assistant_turn_start.text.empty()) strip_tags_vec.push_back(g_model_tokens.assistant_turn_start.text);
    if (!g_model_tokens.turn_end.text.empty()) strip_tags_vec.push_back(g_model_tokens.turn_end.text);
    // Also strip individual base tokens (<|im_end|>, <|eot_id|>, etc.) that can
    // appear as spurious EOGs embedded inside XML tags and attributes.
    collect_base_turn_tokens(strip_tags_vec);
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
        tool_out = execute_tool_call(tool_call, state.file_cache);

        // Handle validation errors reported by the struct.
        if (!tool_out.recognized || !tool_out.params_valid) {
            state.invalid_tool_strikes++;


            string label = !tool_out.recognized ? "Invalid Tool Call" : "Malformed Tool Call";
            diag("System: " + label + " (Strike " + std::to_string(state.invalid_tool_strikes) + ").", "\033[1;31m");

            // Always show the raw tool call so the user can diagnose what went wrong.
            diag("  Tool call: " + truncate_tool_call_params(tool_call), "\033[90m");

            if (is_debug) {
                diag("  Raw tool_call: " + tool_call, "\033[90m");
                diag("  Parsed tool name: \"" + tool_out.parsed_tool_name + "\"", "\033[90m");
                if (!tool_out.recognized) {
                    diag("  Reason: Unknown tool name. Known tools: read_files, search_file, write_file, edit_file, exec_shell, web_search.", "\033[90m");
                }
                if (!tool_out.params_valid && !tool_out.missing_params.empty()) {
                    string mp;
                    for (size_t i = 0; i < tool_out.missing_params.size(); i++) {
                        if (i > 0) mp += ", ";
                        mp += "\"" + tool_out.missing_params[i] + "\"";
                    }
                    diag("  Missing required parameters: " + mp, "\033[90m");
                }
            }

            if (state.invalid_tool_strikes >= 5) {
                diag("System: " + std::to_string(state.invalid_tool_strikes) + " consecutive invalid tool calls. Intervention failed, ejecting to prompt.", "\033[1;31m");
                abort_auto = true;
            } else if (state.invalid_tool_strikes >= 2) {
                diag("System: " + std::to_string(state.invalid_tool_strikes) + " consecutive invalid tool calls. Injecting intervention.", "\033[1;31m");
                inject_auto_user_msg = true;
                active_intervention_msg = SYSTEM_PROMPT_REMINDER;
            }
        } else {
            state.invalid_tool_strikes = 0;

            was_loop = state.loop_guard.record_and_check(tool_call);

            if (stop_generation) {
                diag("Tool Interrupted by User", "\033[31m");
                stop_generation = 0;
                result.was_interrupted = true;
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
            static constexpr size_t STDOUT_TRUNCATE_LIMIT = 500;
            if (should_output_to_stdout() && result_to_print.length() > STDOUT_TRUNCATE_LIMIT) {
                size_t original_len = result_to_print.length();
                result_to_print = result_to_print.substr(0, STDOUT_TRUNCATE_LIMIT) + "\n  ... (truncated, " + std::to_string(original_len) + " chars total -- see browser for full output)\n";
            }
            size_t p = 0;
            while ((p = result_to_print.find('\n')) != string::npos) {
                console("  ", (int)p, result_to_print.c_str(),"\n");
                result_to_print.erase(0, p + 1);
            }
            if (!result_to_print.empty()) console("  ", result_to_print.c_str(),"\n");
        }

        string display_for_browser = tool_out.display;

        if (!display_for_browser.empty()) {
            string safe_result = html_escape(display_for_browser);

            string result_html = "\n\n<div class='tool-result'>Tool Result:<pre><code>" + safe_result + "</code></pre></div>\n\n";
            stream_tool_result(result_html);
        }
        consoleFlush();

        // Log tool result to chat_log with structured label.
        // exec_shell already streams its output incrementally to chat_log in
        // filesystem.cc, so skip it here to avoid duplication.
        if (tool_out.parsed_tool_name != "exec_shell") {
            string logged = tool_out.content;
            // For web_search, log query + snippets but skip full page content
            if (logged.find("Search Results for:") == 0) {
                string filtered;
                size_t i = 0;
                while (i < logged.size()) {
                    size_t page_content = logged.find("Page Content: ", i);
                    if (page_content != string::npos && page_content + 14 < logged.size()) {
                        filtered += logged.substr(i, page_content - i);
                        size_t j = page_content + 14;
                        bool past_blank = false;
                        while (j < logged.size()) {
                            if (logged[j] == '\n') {
                                if (past_blank) break;
                                past_blank = true;
                            } else {
                                past_blank = false;
                            }
                            j++;
                        }
                        filtered += "[Page Content omitted from log]\n";
                        i = j;
                    } else {
                        filtered += logged.substr(i);
                        break;
                    }
                }
                logged = filtered;
            }
            if (!logged.empty()) {
                chat_log << "=== TOOL_RESULT ===\n" << logged << "\n\n";
                chat_log.flush();
            }
        }
        generated_text = "";

        // Build tool result message as a string, then tokenize in one pass.
        vector<llama_token> t_tokens;
        if (tool_blocked_by_loop || inject_auto_user_msg) {
            // Full user turn + assistant prefill, with optional intervention message.
            string tool_content = "[Tool Result]\n" + tool_out.content;
            if (inject_auto_user_msg && !active_intervention_msg.empty()) {
                tool_content += "\n" + active_intervention_msg;
            }

            // Escape PARAM_END and model turn tokens in the content so they
            // don't get misinterpreted as structural boundaries during tokenization.
            escape_parameter_tags(tool_content);
            escape_turn_tags(tool_content);

            string msg = g_model_tokens.user_turn_start.text + tool_content;
            if (g_model_tokens.has_explicit_turn_end())
                msg += g_model_tokens.turn_end.text;
            msg += g_model_tokens.assistant_turn_start.text;
            t_tokens = tokenize(msg);
        } else {
            // User turn + assistant prefill.
            string tool_content = "[Tool Result]\n" + tool_out.content;

            // Escape PARAM_END and model turn tokens in the content.
            escape_parameter_tags(tool_content);
            escape_turn_tags(tool_content);

            string msg = g_model_tokens.user_turn_start.text + tool_content;
            if (g_model_tokens.has_explicit_turn_end())
                msg += g_model_tokens.turn_end.text;
            msg += g_model_tokens.assistant_turn_start.text;
            t_tokens = tokenize(msg);
        }
        if (n_past + (int)t_tokens.size() >= (int)cparams.n_ctx) {
            double pct = (double)n_past / cparams.n_ctx * 100.0;
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f%%", pct);
            diag("Tool result too large to fit in context (" + std::to_string(t_tokens.size()) + " tokens needed, " + std::to_string(cparams.n_ctx - n_past) + " available). Context usage: " + string(buf) + ".", "\033[1;33m");
            state.auto_continue = false;
            result.context_exhausted = true;
        } else if (!feed_tokens(t_tokens)) {
            abort_auto = true;
        } else {
            // Log tool result tokens to token_log when debug is enabled
            log_tokens("FEED TOOL_RESULT", t_tokens, ctx);

            g_auto_continue_depth++;
            if (g_auto_continue_depth > max_auto_continue) {
                diag("System: Max auto-continue depth reached (" + std::to_string(g_auto_continue_depth) + "/" + std::to_string(max_auto_continue) + "). LLM may be stuck in a loop. Ejecting to prompt.", "\033[1;31m");
                state.auto_continue = false;
            } else if (allow_continue_resume && result.was_interrupted) {
                // Interrupted during tool call -- feed result but drop to prompt.
                // User can type "continue" to resume generation.
                diag("Tool execution interrupted. Type '/continue' to let the LLM proceed, or provide input.", "\033[1;33m");
                state.tool_interrupt_pending = true;
                state.auto_continue = false;
            } else {
                diag_speed(n_past, cparams.n_ctx, state.last_t_count, state.last_elapsed, state.last_decode_time);
                state.auto_continue = true;
                result.should_auto_continue = true;
                return result;
            }
        }

        if (abort_auto) {
            state.auto_continue = false;
            generated_text = "";

            string abort_msg;
            if (state.invalid_tool_strikes >= 5) {
                abort_msg = "System Error: You are generating malformed tool calls. Your XML schema is incorrect. Stop and carefully review the required format. Do NOT wrap tool calls in markdown code blocks or other formatting.";
            } else {
                abort_msg = "System Error: Tool call blocked -- you are repeating yourself. Stop retrying and try a different approach (e.g., use search_file instead of exec_shell for code searches).";
            }
            vector<llama_token> abort_tokens = build_tool_result_turn(ctx, abort_msg);
            // build_tool_result_turn wraps: user_start + "[Tool Result]\n" + msg + turn_end + "\n" + assistant_start
            if (n_past + (int)abort_tokens.size() < (int)cparams.n_ctx) {
                feed_tokens(abort_tokens);

                // Log abort tool result tokens to token_log when debug is enabled
                log_tokens("FEED TOOL_RESULT", abort_tokens, ctx);
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
        vector<llama_token> abort_tokens = build_tool_result_turn(ctx, abort_msg);
        if (n_past + (int)abort_tokens.size() < (int)cparams.n_ctx) {
            feed_tokens(abort_tokens);

            // Log abort tool result tokens to token_log when debug is enabled
            log_tokens("FEED TOOL_RESULT", abort_tokens, ctx);
        }
    }

    return result;
}

