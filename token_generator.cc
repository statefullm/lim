#include "token_generator.h"
#include "tokens.h"
#include "output.h"
#include "signals.h"
#include "model.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <iomanip>
#include <readline/readline.h>

using namespace std;
using namespace Tokens;

// Forward declarations for functions defined in main.cc
extern void diag(const string& msg, const char* color);
extern bool is_debug;
extern ofstream chat_log;
extern ofstream token_log;

// --- Helper to escape token piece strings for token log ---
string escape_token_piece(const string& s) {
    string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c == '"') out += "\\\"";
        else out += c;
    }
    return out;
}

// Find tool call end robustly, handling malformed closing tags and nested FUNC_START in content.
size_t find_tool_end_robust(const string& text, size_t from_pos) {
    string fe_str(FUNC_END);
    string fs_str(FUNC_START);
    int depth = 1;
    size_t pos = from_pos;
    while (pos != string::npos && pos < text.length()) {
        size_t next_start = text.find(fs_str, pos);
        size_t next_end = text.find(fe_str, pos);
        if (next_end == string::npos) break;
        if (next_start != string::npos && next_start < next_end) {
            depth++;
            pos = next_start + fs_str.length();
        } else {
            depth--;
            if (depth == 0) return next_end;
            pos = next_end + fe_str.length();
        }
    }
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
void repair_malformed_tool_end(string& text, size_t pos) {
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
void _strip_think_and_tool_tags(string& str) {
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

// File-scope statics for EOG recovery diagnostics
static int eog_event_count = 0;
static int total_poll_iters = 0;
static int max_poll_iters = 0;
static int last_printed_eog = 0;

// File-scope statics for stall detection
static int recent_token_count_tg = 0;
static auto last_rate_check_tg = chrono::high_resolution_clock::now();

TokenGenerator::TokenGenerator(llama_context* ctx, const llama_vocab* vocab,
                               llama_sampler* smpl, llama_batch& batch,
                               int& n_past, const llama_context_params& cparams,
                               double turn_timeout_sec, bool was_mid_tool_call,
                               int last_n_past,
                               std::vector<llama_token>* out_tokens)
    : ctx_(ctx), vocab_(vocab), smpl_(smpl), batch_(batch), n_past_(n_past),
      cparams_(cparams), turn_timeout_sec_(turn_timeout_sec),
      print_pos_(0),
      in_tool_call_stream_(was_mid_tool_call),
      tool_start_(was_mid_tool_call ? 0 : string::npos),
      tool_end_(string::npos),
      trigger_tool_execution_(false),
      func_search_pos_(0),
      had_eog_recovery_(false),
      context_warned_this_turn_(false),
      in_thinking_block_(false),
      think_start_(string::npos),
      think_end_(string::npos),
      think_buffering_(true),
      t_count_(0),
      last_n_past_(last_n_past),
      was_mid_tool_call_(was_mid_tool_call),
      out_tokens_(out_tokens)
{
    generated_text_.reserve(32768);
    unprinted_text_.reserve(1024);
    full_response_.reserve(32768);
}

TokenGenerator::Result TokenGenerator::generate() {
    auto start = chrono::high_resolution_clock::now();

    bool was_interrupted = false;
    bool early_exit = false;

    while (true) {
        if (stop_generation) {
            diag("Task Interrupted by User", "\033[31m");
            stream("\n\n[Task Interrupted by User]\n\n");
            stop_generation = 0;
            g_was_interrupted = 0;
            was_interrupted = true;
            rl_redisplay();
            break;
        }

        {
            auto now = chrono::high_resolution_clock::now();
            double elapsed_turn = chrono::duration<double>(now - start).count();
            if (elapsed_turn >= turn_timeout_sec_) {
                diag("System: Turn timeout reached (" + std::to_string((int)elapsed_turn) + "s/" + std::to_string((int)turn_timeout_sec_) + "s). Pausing generation.", "\033[1;33m");
                stream("\n\n[Turn Timeout Reached]\n\n");
                stop_generation = 0;
                g_was_interrupted = 0;
                was_interrupted = true;
                rl_redisplay();
                break;
            }
        }

        {
            int context_90pct = (int)(cparams_.n_ctx * 0.9);
            if (n_past_ >= context_90pct && !context_warned_this_turn_) {
                context_warned_this_turn_ = true;
                if (!unprinted_text_.empty()) {
                    console(unprinted_text_.c_str());
                    consoleFlush();
                    stream(unprinted_text_);
                    unprinted_text_ = "";
                }
                diag("Context approaching limit (" + std::to_string(n_past_) + "/" + std::to_string(cparams_.n_ctx) + "). Type '/reincarnate' to start a fresh session, or '/clear' to reset.", "\033[1;33m");
            }
        }

        if (n_past_ >= (int)cparams_.n_ctx - 10) {
            string ctx_diag;
            if (n_past_ != last_n_past_) {
                ostringstream oss;
                oss << " (n_past=" << n_past_ << ", last_n_past=" << last_n_past_ << ", n_ctx=" << cparams_.n_ctx << ")";
                ctx_diag = oss.str();
            }
            diag("Context Window Exhausted!" + ctx_diag + ". Type '/clear' to reset.", "\033[31m");
            if (!unprinted_text_.empty()) {
                console(unprinted_text_.c_str());
                consoleFlush();
                stream(unprinted_text_);
            }
            early_exit = true;
            break;
        }

        if (batch_.n_tokens < 1) {
            diag("Error: no tokens in batch to sample from", "\033[31m");
            early_exit = true;
            break;
        }
        llama_token next_token = llama_sampler_sample(smpl_, ctx_, batch_.n_tokens - 1);

        // Track this sampled token for compact save/restore
        if (out_tokens_) out_tokens_->push_back(next_token);

        if (llama_vocab_is_eog(vocab_, next_token)) {
            size_t active_ts = generated_text_.find(FUNC_START);
            size_t active_te = find_tool_end_robust(generated_text_, active_ts != string::npos ? active_ts : 0);
            bool inside_unclosed_tool = (active_ts != string::npos && (active_te == string::npos || active_ts > active_te));

            int poll_iter_used = 0;
            static constexpr int DEFAULT_EOG_RESAMPLE_MAX = 30;
            static constexpr int EOG_RESAMPLE_HARD_CAP = 200;
            const char* eog_env = getenv("LLLM_EOG_RESAMPLE_MAX");
            int max_iterations = (eog_env != nullptr && strlen(eog_env) > 0) ? atoi(eog_env) : DEFAULT_EOG_RESAMPLE_MAX;
            max_iterations = std::max(1, std::min(max_iterations, EOG_RESAMPLE_HARD_CAP));

            bool recovered = false;
            {
                llama_token polled = llama_sampler_sample(smpl_, ctx_, batch_.n_tokens - 1);
                if (!llama_vocab_is_eog(vocab_, polled)) {
                    next_token = polled;
                    recovered = true;
                    poll_iter_used = 1;
                }
            }
            if (!recovered) {
                for (int poll_iter = 0; poll_iter < max_iterations; ++poll_iter) {
                    if (stop_generation) break;
                    llama_token polled = llama_sampler_sample(smpl_, ctx_, batch_.n_tokens - 1);
                    if (!llama_vocab_is_eog(vocab_, polled)) {
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
                string recovered_piece = common_token_to_piece(ctx_, next_token);
                if (is_debug) {
                    message("\033[90m[EOG recovery: token=" + recovered_piece + " | polls=" + std::to_string(poll_iter_used) + "/" + std::to_string(max_iterations) + "]\033[0m\n");
                    cout.flush();
                    if (eog_event_count - last_printed_eog >= 10) {
                        double avg = (double)total_poll_iters / eog_event_count;
                        char avg_buf[16];
                        snprintf(avg_buf, sizeof(avg_buf), "%.1f", avg);
                        message("\033[90m[EOG diagnostic: " + std::to_string(eog_event_count) + " spurious EOGs recovered | "
                            "avg polls: " + string(avg_buf) + "/" + std::to_string(max_iterations) +
                            " | max: " + std::to_string(max_poll_iters) + "/" + std::to_string(max_iterations) + "]\033[0m\n");
                        cout.flush();
                        last_printed_eog = eog_event_count;
                    }
                }
                had_eog_recovery_ = true;
            }

            if (!recovered) {
                if (inside_unclosed_tool) {
                    if (is_debug) {
                        message("\033[31m[System: Premature End-Of-Turn detected after polling timeout. Auto-recovering tags...]\033[0m\n");
                        cout.flush();
                    }
                    size_t trailing_slash = generated_text_.rfind("</");
                    if (trailing_slash != string::npos && trailing_slash > active_ts) {
                        size_t drop_len = generated_text_.length() - trailing_slash;
                        generated_text_.erase(trailing_slash);
                        full_response_.erase(full_response_.length() - drop_len);
                    }
                    string forced_close = "\n" + string(TURN_END) + "\n" + string(FUNC_END) + "\n";
                    generated_text_ += forced_close;
                    full_response_ += forced_close;
                    tool_start_ = active_ts;
                    tool_end_ = generated_text_.find(FUNC_END, active_ts);
                    trigger_tool_execution_ = true;
                }
                break;
            }
        }

        string token_str = common_token_to_piece(ctx_, next_token).c_str();
        generated_text_ += token_str;
        full_response_ += token_str;

        if (is_debug && token_log.is_open()) {
            token_log << t_count_ << " " << next_token << " \"" << escape_token_piece(token_str) << "\"\n";
            token_log.flush();
        }

        if (think_start_ == string::npos) {
            think_start_ = generated_text_.find(Tokens::THINK_START);
        }
        if (think_start_ != string::npos && think_end_ == string::npos) {
            think_end_ = generated_text_.find(Tokens::THINK_END, think_start_);
        }
        in_thinking_block_ = (think_start_ != string::npos && think_end_ == string::npos);

        if (tool_start_ == string::npos) {
            size_t search_from = func_search_pos_;
            while (true) {
                tool_start_ = generated_text_.find(FUNC_START, search_from);
                if (tool_start_ == string::npos) break;
                if (think_start_ != string::npos &&
                    tool_start_ >= think_start_ &&
                    (think_end_ == string::npos || tool_start_ < think_end_)) {
                    search_from = think_end_;
                    continue;
                }
                break;
            }
            if (tool_start_ == string::npos) {
                func_search_pos_ = generated_text_.length() > 20 ? generated_text_.length() - 20 : 0;
            }
        }
        if (tool_start_ != string::npos && tool_end_ == string::npos) {
            size_t search_from = was_mid_tool_call_ ? 0 : tool_start_;
            tool_end_ = find_tool_end_robust(generated_text_, search_from);
            if (tool_end_ != string::npos) {
                size_t exact_pos = generated_text_.find(FUNC_END, search_from);
                if (exact_pos == string::npos) {
                    repair_malformed_tool_end(generated_text_, tool_end_);
                    tool_end_ = generated_text_.find(FUNC_END, search_from);
                }
            }
        }

        in_tool_call_stream_ = (tool_start_ != string::npos && tool_end_ == string::npos);

        if (in_tool_call_stream_ && generated_text_.length() >= 4 &&
            generated_text_.compare(generated_text_.length() - 4, 4, DOUBLE_OPEN) == 0) {
            diag("System: Infinite slash loop detected. Auto-recovering...", "\033[31m");
            size_t bad_pos = generated_text_.rfind(DOUBLE_OPEN);
            if (bad_pos != string::npos && bad_pos > tool_start_) {
                size_t drop_len = generated_text_.length() - bad_pos;
                generated_text_.erase(bad_pos);
                full_response_.erase(full_response_.length() - drop_len);
            }
            string forced_close = "\n" + string(TURN_END) + "\n" + string(FUNC_END) + "\n";
            generated_text_ += forced_close;
            full_response_ += forced_close;
            tool_end_ = find_tool_end_robust(generated_text_, was_mid_tool_call_ ? 0 : tool_start_);
            if (tool_end_ != string::npos) {
                size_t exact_pos = generated_text_.find(FUNC_END, was_mid_tool_call_ ? 0 : tool_start_);
                if (exact_pos == string::npos) {
                    repair_malformed_tool_end(generated_text_, tool_end_);
                    tool_end_ = generated_text_.find(FUNC_END, was_mid_tool_call_ ? 0 : tool_start_);
                }
            }
            trigger_tool_execution_ = true;
            break;
        }

        if (tool_end_ != string::npos) {
            trigger_tool_execution_ = true;
            break;
        }

        // --- OUTPUT RENDERING ---
        if (!in_tool_call_stream_ && !in_thinking_block_) {
            size_t safe_len = generated_text_.length();
            string fstart(FUNC_START);
            string tstart(Tokens::THINK_START);

            if (think_start_ != string::npos && think_end_ != string::npos) {
                size_t think_block_end = think_end_ + string(Tokens::THINK_END).length();
                think_buffer_.clear();
                think_buffering_ = true;
                if (print_pos_ < think_block_end) {
                    if (!think_buffering_ && print_pos_ <= think_end_) {
                        string close_tag = generated_text_.substr(print_pos_, think_block_end - print_pos_);
                        console_think(close_tag.c_str());
                        consoleThinkFlush();
                    }
                    print_pos_ = think_block_end;
                }
            }

            for (size_t len = 1; len <= max(fstart.length(), tstart.length()) && len <= generated_text_.length(); ++len) {
                if (generated_text_.compare(generated_text_.length() - len, len, fstart, 0, len) == 0 ||
                    generated_text_.compare(generated_text_.length() - len, len, tstart, 0, len) == 0) {
                    safe_len = generated_text_.length() - len;
                    break;
                }
            }

            if (safe_len > print_pos_) {
                unprinted_text_ += generated_text_.substr(print_pos_, safe_len - print_pos_);
                print_pos_ = safe_len;
            }

            if (!unprinted_text_.empty() && (t_count_ % 10 == 0 || unprinted_text_.back() == '\n')) {
                console(unprinted_text_.c_str());
                consoleFlush();
                stream(unprinted_text_);
                unprinted_text_ = "";
            }
        } else if (in_thinking_block_) {
            size_t safe_len = generated_text_.length();
            string tstart(Tokens::THINK_START);
            string tend(Tokens::THINK_END);

            for (size_t len = 1; len <= tend.length() && len <= generated_text_.length(); ++len) {
                if (generated_text_.compare(generated_text_.length() - len, len, tend, 0, len) == 0) {
                    safe_len = generated_text_.length() - len;
                    break;
                }
            }

            if (safe_len > print_pos_) {
                string think_output = generated_text_.substr(print_pos_, safe_len - print_pos_);
                if (think_buffering_) {
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
                        think_buffer_.clear();
                        think_output = think_output.substr(content_start);
                        console_think(think_output.c_str());
                        consoleThinkFlush();
                        think_buffering_ = false;
                    } else {
                        think_buffer_ += think_output;
                    }
                } else {
                    _strip_think_and_tool_tags(think_output);
                    console_think(think_output.c_str());
                    consoleThinkFlush();
                }
                print_pos_ = safe_len;
            }
        } else {
            if (!unprinted_text_.empty()) {
                console(unprinted_text_.c_str());
                consoleFlush();
                stream(unprinted_text_);
                unprinted_text_ = "";
            }
            print_pos_ = generated_text_.length();
        }

        t_count_++;

        if (t_count_ % 50 == 0 && !in_tool_call_stream_ && !in_thinking_block_) {
            cerr << "\r\033[K[Generating... " << t_count_ << " tokens]";
            cerr.flush();
        }

        {
            recent_token_count_tg++;
            auto now = chrono::high_resolution_clock::now();
            double elapsed_window = chrono::duration<double>(now - last_rate_check_tg).count();
            if (elapsed_window >= 10.0) {
                double rate = recent_token_count_tg / elapsed_window;
                if (rate < 0.5 && t_count_ > 20) {
                    diag("System: Token generation rate critically low (" +
                         to_string((int)(rate * 10) / 10.0) + " tok/s). Possible stall detected.", "\033[1;33m");
                }
                recent_token_count_tg = 0;
                last_rate_check_tg = now;
            }
        }

        batch_.n_tokens = 0;
        common_batch_add(batch_, next_token, n_past_++, {0}, true);
        if (!handle_llama_decode_error(ctx_, batch_)) {
            sync_n_past(ctx_, n_past_);
            early_exit = true;
            break;
        }
    } // END INNER TOKEN LOOP

    // Flush remaining unprinted text and clear progress line
    cerr << "\r\033[K";
    if (!unprinted_text_.empty()) {
        if (unprinted_text_.back() != '\n') {
            console((unprinted_text_ + "\n").c_str());
            consoleFlush();
            stream(unprinted_text_ + "\n");
        } else {
            console(unprinted_text_.c_str());
            consoleFlush();
            stream(unprinted_text_);
        }
        unprinted_text_ = "";
    }

    // Build result
    Result result;
    result.text = generated_text_;
    result.token_count = t_count_;
    result.tool_start = tool_start_;
    result.tool_end = tool_end_;
    result.has_tool_call = trigger_tool_execution_ && tool_start_ != string::npos && tool_end_ != string::npos;
    result.was_interrupted = was_interrupted;
    result.early_exit = early_exit;
    return result;
}
