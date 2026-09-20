#include "token_generator.h"
#include "tokens.h"
#include "output.h"
#include "session_utils.h"
#include "signals.h"
#include "model.h"
#include "mtp.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <iomanip>
#include <string_view>
#include <readline/readline.h>

using namespace std;
using namespace Tokens;

// Forward declarations for functions defined in main.cc
extern void diag(const string& msg, const char* color);
extern bool is_debug;
extern ofstream chat_log;
extern ofstream token_log;
extern bool honest_speed;
extern int speed_update_interval;

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

// Find tool call end robustly, handling malformed closing tags, nested FUNC_START in content,
// and FUNC_START/FUNC_END embedded within parameter values (ignored via in_parameter guard).
size_t find_tool_end_robust(const string& text, size_t from_pos, bool* out_in_parameter) {
    string fe_str(FUNC_END);
    string fs_str(FUNC_START);
    string ps_str(PARAM_START);
    string pe_str(PARAM_END);
    int depth = 1;
    bool in_parameter = false;
    size_t pos = from_pos;
    // Skip the FUNC_START at from_pos -- caller already knows it's there (depth starts at 1).
    if (text.compare(pos, fs_str.length(), fs_str) == 0) {
        pos += fs_str.length();
    }
    while (pos != string::npos && pos < text.length()) {
        size_t next_start = text.find(fs_str, pos);
        size_t next_end = text.find(fe_str, pos);
        size_t next_ps = text.find(ps_str, pos);
        size_t next_pe = in_parameter ? text.find(pe_str, pos) : string::npos;

        // Determine the earliest event
        size_t events[4] = {next_start, next_end, next_ps, next_pe};
        int type = -1; // 0=start, 1=end, 2=param_start, 3=param_end
        for (int i = 0; i < 4; i++) {
            if (events[i] == string::npos) continue;
            if (type == -1 || events[i] < events[type]) type = i;
        }
        if (type == -1) break;

        if (type == 2) {
            // Entered a parameter region
            in_parameter = true;
            pos = next_ps + ps_str.length();
            continue;
        }
        if (type == 3) {
            // Exited a parameter region
            in_parameter = false;
            pos = next_pe + pe_str.length();
            continue;
        }
        if (in_parameter) {
            // FUNC_START or FUNC_END inside parameter value -- ignore
            pos = events[type] + (type == 0 ? fs_str.length() : fe_str.length());
            continue;
        }
        if (type == 0) {
            depth++;
            pos = next_start + fs_str.length();
        } else {
            depth--;
            if (depth == 0) {
                if (out_in_parameter) *out_in_parameter = in_parameter;
                return next_end;
            }
            pos = next_end + fe_str.length();
        }
    }
    // Partial-match fallback: look for an incomplete FUNC_END (e.g., split across tokens).
    // Partial-match fallback: single forward pass from pos, O(n).
    string partial = fe_str.substr(0, fe_str.length() - 1);
    while (pos < text.length()) {
        size_t ps = text.find(ps_str, pos);
        size_t pe = in_parameter ? text.find(pe_str, pos) : string::npos;
        size_t pp = text.find(partial, pos);

        // Find earliest event
        if (ps != string::npos && (pe == string::npos || ps < pe) && (pp == string::npos || ps < pp)) {
            in_parameter = true;
            pos = ps + ps_str.length();
            continue;
        }
        if (pe != string::npos && (pp == string::npos || pe < pp)) {
            in_parameter = false;
            pos = pe + pe_str.length();
            continue;
        }
        if (pp == string::npos) break;

        // Found partial FUNC_END match at pp.
        if (in_parameter) {
            pos = pp + partial.length();
            continue;
        }
        if (pp + partial.length() < text.length()) {
            char next_c = text[pp + partial.length()];
            if (next_c == '>') {
                if (out_in_parameter) *out_in_parameter = in_parameter;
                return pp;
            }
            size_t garbage_end = text.find('>', pp + partial.length());
            if (garbage_end != string::npos) {
                if (out_in_parameter) *out_in_parameter = in_parameter;
                return pp + partial.length();
            }
            // Partial match with no '>' after it -- the model clearly intended to
            // close the tag but produced garbage instead.  Accept as completed so
            // the caller can repair the missing '>' and break out of the loop.
            if (out_in_parameter) *out_in_parameter = in_parameter;
            return pp + partial.length();
        }
        // Partial match at end of text -- accept as completed.
        if (out_in_parameter) *out_in_parameter = in_parameter;
        return pp + partial.length();
    }
    if (out_in_parameter) *out_in_parameter = in_parameter;
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
        g_model_tokens.think_start, g_model_tokens.think_end,
        FUNC_START, FUNC_END,
        PARAM_START, PARAM_END
    };
    strip_tags(str, all_tags);
}

// File-scope statics for EOG recovery diagnostics
static int eog_event_count = 0;
static int total_poll_iters = 0;
static int max_poll_iters = 0;
static int last_printed_eog = 0;

TokenGenerator::TokenGenerator(llama_context* ctx, const llama_vocab* vocab,
                               llama_sampler* smpl, llama_batch& batch,
                               int& n_past, const llama_context_params& cparams,
                               double turn_timeout_sec, bool was_mid_tool_call,
                               int last_n_past,
                               std::vector<llama_token>* out_tokens,
                               double feed_time,
                               bool is_reincarnating,
                               std::function<void(bool lockstep)> on_tool_start,
                               bool was_mid_thinking_block)
    : ctx_(ctx), vocab_(vocab), smpl_(smpl), batch_(batch), n_past_(n_past),
      cparams_(cparams), turn_timeout_sec_(turn_timeout_sec), feed_time_(feed_time),
      print_pos_(0),
      in_tool_call_stream_(was_mid_tool_call),
      in_parameter_(false),
      tool_start_(was_mid_tool_call ? 0 : string::npos),
      tool_end_(string::npos),
      trigger_tool_execution_(false),
      func_search_pos_(0),
      context_warned_this_turn_(false),
      // Resuming mid-think: the opening tag was already in the context before
      // the interrupt, so start with the block open (think_start_ = 0 keeps
      // the open-tag re-search and the depth count from double-counting;
      // raw tool tokens are skipped as prose until think_end is found) and
      // skip the buffered first-chunk path so the opening tag is not emitted
      // to the terminal a second time.
      in_thinking_block_(was_mid_thinking_block),
      think_start_(was_mid_thinking_block ? 0 : string::npos),
      think_end_(string::npos),
      think_depth_(1),
      think_scan_pos_(0),
      think_buffering_(!was_mid_thinking_block),
      t_count_(0),
      on_tool_start_(std::move(on_tool_start)),
      last_n_past_(last_n_past),
      was_mid_tool_call_(was_mid_tool_call),
      is_reincarnating_(is_reincarnating),
      out_tokens_(out_tokens),
      tool_call_outside_param_count_(0),
      eog_recovered_this_token_(false)
{
    generated_text_.reserve(32768);
    unprinted_text_.reserve(1024);
}

TokenGenerator::Result TokenGenerator::generate() {
    auto start = chrono::high_resolution_clock::now();


    bool was_interrupted = false;
    bool early_exit = false;
    bool stuck_in_tool_call = false;
    bool ended_on_eog = false;
    // Swallowed in-block EOG count (EOG block in the token loop below).
    // Resets only on an EOG sampled outside the block, so the cap bounds
    // the TOTAL in-block swallows since the last out-of-block EOG, even
    // when they are separated by regular tokens: an EOG-word-EOG-word
    // degenerate loop still hits the cap and ends the turn.
    int think_eog_count = 0;

    // Set when FUNC_START just became fully present in generated_text_ this
    // iteration; the on_tool_start_ hook is then fired after the end-of-iteration
    // feed so n_past lands exactly right after FUNC_START.  Fires at most once
    // per generation (tool_start_ never transitions back to npos, and a resume
    // mid-tool-call starts with tool_start_ == 0 so no transition occurs).
    bool tool_start_hook_pending = false;

    // Silent-loop threshold: max tokens allowed outside parameters while inside
    // a tool call before aborting. Constant for the run -- read once per turn,
    // not per token.
    static constexpr int DEFAULT_OUTSIDE_PARAM_LIMIT = 100;
    const char* outside_param_env = getenv("LIM_TOOL_IGNORE");
    int outside_param_limit = (outside_param_env != nullptr && strlen(outside_param_env) > 0) ? atoi(outside_param_env) : DEFAULT_OUTSIDE_PARAM_LIMIT;
    outside_param_limit = std::max(1, outside_param_limit);

    // Generation timing: matches llama-cli's "Generation: X t/s" (t_token_generation).
    // t_gen_start is set after sampling the first token (with synchronize), and
    // t_gen_end is updated after every subsequent sample+sync, including the last.
    // This measures N sampling operations + (N-1) decode cycles, identical to
    // llama-cli's server-context.cpp timing window.
    double gen_wall_time = 0.0;
    auto t_gen_start = chrono::high_resolution_clock::now();
    auto t_gen_end   = t_gen_start;

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
            if (n_past_ >= context_90pct && !context_warned_this_turn_ && !is_reincarnating_) {
                context_warned_this_turn_ = true;
                if (!unprinted_text_.empty()) {
                    console(unprinted_text_.c_str());
                    consoleFlush();
                    stream(unprinted_text_);
                    g_stdout_ended_with_newline = (unprinted_text_.back() == '\n');
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
                g_stdout_ended_with_newline = (unprinted_text_.back() == '\n');
            }
            early_exit = true;
            break;
        }

        if (batch_.n_tokens < 1) {
            diag("Error: no tokens in batch to sample from", "\033[31m");
            early_exit = true;
            break;
        }
        // MTP: during an in-progress round the batch is the verify batch and
        // this iteration samples a specific verify row (the bonus, sampled
        // after all drafts matched, sits at the last row -- same index as the
        // default).  Normal iterations sample the batch's last row as before.
        const int mtp_sample_row = spec_in_progress_ ? spec_verify_row_ : batch_.n_tokens - 1;
        llama_token next_token = llama_sampler_sample(smpl_, ctx_, mtp_sample_row);

        // Generation timing: record start on first sampled token, end on every sample.
        // llama_sampler_sample already synchronizes the context internally via
        // llama_get_sampled_*_ith -> ctx->synchronize(), so no extra sync needed.
        if (!honest_speed) {
            if (t_count_ == 0) {
                t_gen_start = chrono::high_resolution_clock::now();
            }
            t_gen_end = chrono::high_resolution_clock::now();
        }

        // Feed the sampled token into the KV cache and the token tracker in
        // lockstep: the tracker (state_.all_context_tokens) must contain
        // exactly the tokens present in the KV cache, or save files outgrow
        // the fast restore cache (one token short per turn) and the next
        // restore is rejected.  The loop-break paths (tool call completed,
        // final EOG, degeneration recovery) call this before breaking so the
        // token that ends the turn is decoded like every other generated
        // token instead of being tracked without a decode.
        auto feed_token = [&]() -> bool {
            if (spec_no_feed_) {
                // Matched draft token: the KV cell at n_past_ holds (or, for
                // the origin, is about to be created by) the verify batch,
                // which decodes this exact token.  Advance the tracker and
                // n_past in lockstep without a decode.  The tracker/KV
                // consistency invariant holds at every turn exit: a break
                // between an origin commit and the verify decode is healed
                // by the post-loop cleanup (single-row cell fill).
                if (out_tokens_) out_tokens_->push_back(next_token);
                n_past_++;
                spec_no_feed_ = false;
                return true;
            }
            if (out_tokens_) out_tokens_->push_back(next_token);
            batch_.n_tokens = 0;
            common_batch_add(batch_, next_token, n_past_++, {0}, true);
            if (!handle_llama_decode_error(ctx_, batch_)) {
                sync_n_past(ctx_, n_past_);
                return false;
            }
            return true;
        };

        if (llama_vocab_is_eog(vocab_, next_token)) {
            // Tracked tool state from the previous iteration, computed on
            // this same generated_text_ (the EOG piece is not appended yet).
            // A raw re-search here would match prose tokens inside the
            // (closed) thinking block and mis-pair them with a real call.
            size_t active_ts = tool_start_;
            bool inside_unclosed_tool = (tool_start_ != string::npos && tool_end_ == string::npos);

            int poll_iter_used = 0;
            static constexpr int DEFAULT_EOG_RESAMPLE_MAX = 256;
            const char* eog_env = getenv("LIM_EOG_RESAMPLE_MAX");
            int max_iterations = (eog_env != nullptr && strlen(eog_env) > 0) ? atoi(eog_env) : DEFAULT_EOG_RESAMPLE_MAX;
            max_iterations = std::max(1, max_iterations);

            // EOG while the thinking block is open: the model paused
            // mid-reasoning.  Swallow it (piece appended + fed at the bottom
            // of the loop) and let it continue -- no resample, and no tool
            // repair (raw FUNC_START/FUNC_END tokens inside the block are
            // prose, never a call).  This is the loop's only EOG break,
            // though: a model stuck emitting EOGs inside the block would
            // otherwise never end the turn.  Past max_iterations consecutive
            // swallows, fall through to the normal EOG termination below.
            bool swallow = false;
            if (in_thinking_block_) {
                think_eog_count++;
                if (think_eog_count > max_iterations) {
                    if (is_debug) {
                        message("\033[90m[EOG inside thinking block: " + std::to_string(think_eog_count) +
                                " consecutive -- ending turn]\033[0m\n");
                        cout.flush();
                    }
                } else {
                    swallow = true;
                }
            } else {
                think_eog_count = 0;
            }

            bool recovered = false;
            if (!swallow) {
                // Poll the row the token came from (a verify row during an
                // in-progress MTP round, otherwise the batch's last row).
                llama_token polled = llama_sampler_sample(smpl_, ctx_, mtp_sample_row);
                if (!llama_vocab_is_eog(vocab_, polled)) {
                    next_token = polled;
                    recovered = true;
                    poll_iter_used = 1;
                }
                if (!recovered) {
                    for (int poll_iter = 0; poll_iter < max_iterations; ++poll_iter) {
                        if (stop_generation) break;
                        llama_token polled = llama_sampler_sample(smpl_, ctx_, mtp_sample_row);
                        if (!llama_vocab_is_eog(vocab_, polled)) {
                            next_token = polled;
                            recovered = true;
                            poll_iter_used = poll_iter + 2;
                            break;
                        }
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
                eog_recovered_this_token_ = true;

                // Reset sampler state after EOG recovery. Each call to
                // llama_sampler_sample during polling accepted the rejected
                // EOG token into the penalties ring buffer, polluting the
                // history with tokens that were never actually generated.
                // A full chain reset clears this phantom history so the
                // penalties sampler reflects only real context.
                llama_sampler_reset(smpl_);
            }

            if (!recovered && !swallow) {
                if (inside_unclosed_tool) {
                    if (is_debug) {
                        message("\033[31m[System: Premature End-Of-Turn detected after polling timeout. Auto-recovering tags...]\033[0m\n");
                        cout.flush();
                    }
                    size_t trailing_slash = generated_text_.rfind("</");
                    if (trailing_slash != string::npos && trailing_slash > active_ts) {
                        generated_text_.erase(trailing_slash);
                    }
                    string forced_close = "\n" + string(FUNC_END) + "\n";
                    generated_text_ += forced_close;
                    tool_end_ = generated_text_.length() - string(FUNC_END).length();
                    trigger_tool_execution_ = true;
                }

                // Note: bare PARAM_START/FUNC_END in prose is not degeneration -- the
                // model may be discussing tool calls.  The inside_unclosed_tool repair
                // above handles genuine unclosed tool calls.  The tool executor
                // validates structure on its own for anything that does look like a call.

                // The EOG token is fed (decoded + tracked, or no-feed when it
                // matched a verify-row draft below) but its piece is never
                // appended to generated_text_ (this break precedes the
                // detokenize block): flag it so canonical-text maintainers
                // can mirror it.
                ended_on_eog = true;

                // MTP: if this EOG matched a verify-row draft, its KV cell
                // already holds the token -- feed without re-decoding (which
                // would advance the hybrid recurrent state an extra step).
                // The break skips the comparison block below, so mirror its
                // match bookkeeping here: this EOG is a committed verify row,
                // and the post-loop cleanup rolls back / closes the round
                // based on spec_committed_verify_.
                if (spec_in_progress_ && spec_verify_row_ < spec_verify_rows_ &&
                    next_token == spec_draft_[spec_draft_idx_ - 1]) {
                    spec_no_feed_ = true;
                    spec_committed_verify_ = spec_verify_row_ + 1;
                    if (g_mtp) g_mtp->on_verify_match();
                }

                if (!feed_token()) early_exit = true;
                break;
            }
        }

        // --- MTP speculative comparison ---
        // Placed after EOG recovery so the comparison uses the token that is
        // actually committed; EOG recovery's sampler-reset side effect then
        // applies to later verify rows exactly as it does in the
        // non-speculative path.  The sampled token at each verify row is
        // always the target's own fresh sample from the full sampler chain,
        // so the committed sequence is distribution-exact regardless of the
        // sampling configuration.
        if (spec_bonus_pending_) {
            // Bonus token (all drafts matched): no comparison, fed normally.
            spec_bonus_pending_ = false;
            if (g_mtp) g_mtp->on_bonus();
        } else if (spec_origin_pending_) {
            spec_origin_pending_ = false;
            if (next_token == spec_draft_[0]) {
                // d1 matched: commit it WITHOUT a decode (spec_no_feed_).
                // The verify batch below decodes d1..dk in one pass, so d1's
                // KV cell comes from there -- and its row's logits are what
                // sample d2's position, so every draft is verified at its own
                // position (the common_sampler_sample_and_accept_n protocol).
                // A separate d1 feed would discard the row's logits, shifting
                // every verify-row sample one position ahead of its draft.
                spec_no_feed_ = true;
                spec_verify_pending_ = true;
                spec_in_progress_ = true;
                spec_verify_row_ = 0;
                spec_verify_rows_ = (int)spec_draft_.size() - 1;
                spec_committed_verify_ = 0;
                spec_draft_idx_ = 2;
                if (g_mtp) g_mtp->on_origin_match();
            } else {
                // d1 rejected: discard the drafts; this token is handled
                // exactly like a non-speculative token.
                spec_draft_.clear();
                if (g_mtp) {
                    g_mtp->on_origin_mismatch();
                    g_mtp->note_round(1);
                }
            }
        } else if (spec_in_progress_ && spec_verify_row_ < spec_verify_rows_) {
            if (next_token == spec_draft_[spec_draft_idx_ - 1]) {
                // Matched: the KV cell at this position already holds the
                // identical token (decoded with the verify batch) -- commit
                // it without a decode.
                spec_no_feed_ = true;
                spec_committed_verify_ = spec_verify_row_ + 1;
                spec_verify_row_++;
                spec_draft_idx_++;
                if (g_mtp) g_mtp->on_verify_match();
                if (spec_verify_row_ >= spec_verify_rows_) {
                    // All drafts matched: the next sample (last verify row)
                    // is a free bonus token.
                    spec_in_progress_ = false;
                    spec_bonus_pending_ = true;
                    spec_round_complete_pending_ = true;
                }
            } else {
                // Mismatch: roll back the uncommitted draft cells.  The
                // n_rs_seq snapshot window covers the hybrid recurrent
                // state; the attention KV truncates in the same call.
                const int keep_pos = n_past_;
                if (!llama_memory_seq_rm(llama_get_memory(ctx_), 0, keep_pos, -1)) {
                    // Fallback (should be unreachable): regenerate the KV
                    // from the committed prefix, as the /undo fallback does.
                    diag("MTP: draft rollback failed; regenerating KV cache", "\033[31m");
                    if (!spec_rollback_redecode()) {
                        early_exit = true;
                        break;
                    }
                }
                if (g_mtp) {
                    g_mtp->on_verify_mismatch(spec_draft_idx_);
                    g_mtp->note_round(1 + spec_committed_verify_ + 1);
                }
                spec_in_progress_ = false;
                spec_no_feed_ = false;
                spec_verify_row_ = 0;
                spec_draft_.clear();
                // next_token is the target's own token; feed_token below
                // commits it as in the normal path.
            }
        }

        // Detokenize directly into a pre-allocated buffer to avoid per-token allocation.
        static constexpr int TOKEN_BUF_SIZE = 128;
        char token_buf[TOKEN_BUF_SIZE];
        const int n_chars = llama_token_to_piece(vocab_, next_token, token_buf, TOKEN_BUF_SIZE, 0, true);
        if (n_chars < 0) {
            // Buffer too small: fall back to heap allocation.
            // n_chars is negative and indicates the required size.
            int needed = -n_chars;
            string token_heap(needed + 1, '\0');
            const int n2 = llama_token_to_piece(vocab_, next_token, &token_heap[0], needed + 1, 0, true);
            GGML_ASSERT(n2 > 0 && "heap-allocated buffer still too small for token piece");
            generated_text_.append(token_heap.data(), n2);

            if (is_debug && token_log.is_open()) {
                token_log << t_count_ << " " << next_token << " \"" << escape_token_piece(token_heap.substr(0, n2)) << "\"\n";
                token_log.flush();
            }
        } else if (n_chars > 0) {
            string_view token_sv(token_buf, n_chars);
            generated_text_.append(token_sv.data(), token_sv.size());

            if (is_debug && token_log.is_open()) {
                string token_str(token_sv);
                token_log << t_count_ << " " << next_token << " \"" << escape_token_piece(token_str) << "\"\n";
                token_log.flush();
            }
        }
        // n_chars == 0 means unknown/out-of-range token; skip silently.

        // Think-tag detection: drives output rendering, and excludes the
        // thinking-block region from tool-call detection below.  Raw tool
        // tokens inside thinking are prose (the model may discuss the tool
        // schema while reasoning); real tool calls are sequential with
        // thinking, never nested, so a token inside the block can never be
        // a call.
        if (!g_model_tokens.think_start.empty() && !g_model_tokens.think_end.empty()) {
            if (think_start_ == string::npos) {
                think_start_ = generated_text_.find(g_model_tokens.think_start);
                if (think_start_ != string::npos) {
                    // Depth 1 = the outer opening tag; scanning resumes just past it.
                    think_depth_ = 1;
                    think_scan_pos_ = think_start_ + g_model_tokens.think_start.length();
                }
            }
            if (think_start_ != string::npos && think_end_ == string::npos) {
                // Depth-match the closing tag that pairs with the OUTER opening tag.
                // The model sometimes emits spurious think tags inside its reasoning;
                // nested pairs are ignored so an inner </think> can't terminate the
                // block early (which would leak the rest of the reasoning into the
                // visible response and prematurely collapse the browser's thinking box).
                // Incremental scan: each character is examined at most once -> O(N).
                size_t scan = think_scan_pos_;
                if (scan > generated_text_.length()) scan = generated_text_.length();
                while (true) {
                    size_t p_open  = generated_text_.find(g_model_tokens.think_start, scan);
                    size_t p_close = generated_text_.find(g_model_tokens.think_end,   scan);
                    if (p_open == string::npos && p_close == string::npos) break;
                    bool is_open = (p_open != string::npos && (p_close == string::npos || p_open < p_close));
                    size_t p = is_open ? p_open : p_close;
                    think_depth_ += is_open ? 1 : -1;
                    if (!is_open && think_depth_ <= 0) {
                        think_end_ = p;   // outer closing tag found
                        break;
                    }
                    scan = p + 1;
                }
                // Resume next token from the further of: where this scan stopped
                // (so already-counted tags are never re-counted) and the tail window
                // (so a partial tag split across a token boundary is still found).
                size_t max_tag_len = std::max(g_model_tokens.think_start.length(), g_model_tokens.think_end.length());
                size_t tail = generated_text_.length() > (max_tag_len - 1) ? generated_text_.length() - (max_tag_len - 1) : 0;
                think_scan_pos_ = std::max(scan, tail);
            }
        }
        in_thinking_block_ = (think_start_ != string::npos && think_end_ == string::npos);

        if (tool_start_ == string::npos) {
            size_t search_from = func_search_pos_;
            while (true) {
                size_t found = generated_text_.find(FUNC_START, search_from);
                if (found == string::npos) break;
                // Raw tool tokens inside the thinking block are prose, never
                // a call (calls are sequential with thinking, never nested):
                // skip past them so they can't start a phantom call.  When
                // think_start_ is npos the condition is false for every
                // found position, so non-thinking turns scan exactly as
                // before.
                if (found >= think_start_ &&
                    (think_end_ == string::npos ||
                     found < think_end_ + g_model_tokens.think_end.length())) {
                    search_from = found + string(FUNC_START).length();
                    continue;
                }
                tool_start_ = found;
                // FUNC_START just became fully present: schedule the hook for
                // after this token is fed (see end of iteration).
                tool_start_hook_pending = true;
                break;
            }
            if (tool_start_ == string::npos) {
                func_search_pos_ = generated_text_.length() > 20 ? generated_text_.length() - 20 : 0;
            }
        }
        if (tool_start_ != string::npos && tool_end_ == string::npos) {
            size_t search_from = was_mid_tool_call_ ? 0 : tool_start_;
            tool_end_ = find_tool_end_robust(generated_text_, search_from, &in_parameter_);
            if (tool_end_ != string::npos) {
                size_t exact_pos = generated_text_.find(FUNC_END, search_from);
                if (exact_pos == string::npos) {
                    repair_malformed_tool_end(generated_text_, tool_end_);
                    tool_end_ = generated_text_.find(FUNC_END, search_from);
                }
            }
        }

        in_tool_call_stream_ = (tool_start_ != string::npos && tool_end_ == string::npos);

        if (in_tool_call_stream_ && !in_parameter_ && generated_text_.length() >= 4 &&
            generated_text_.compare(generated_text_.length() - 4, 4, DOUBLE_OPEN) == 0) {
            diag("System: Infinite slash loop detected. Auto-recovering...", "\033[31m");
            size_t bad_pos = generated_text_.rfind(DOUBLE_OPEN);
            if (bad_pos != string::npos && bad_pos > tool_start_) {
                generated_text_.erase(bad_pos);
            }
            string forced_close = "\n" + string(FUNC_END) + "\n";
            if (tool_end_ != string::npos) {
                size_t exact_pos = generated_text_.find(FUNC_END, was_mid_tool_call_ ? 0 : tool_start_);
                if (exact_pos == string::npos) {
                    repair_malformed_tool_end(generated_text_, tool_end_);
                    tool_end_ = generated_text_.find(FUNC_END, was_mid_tool_call_ ? 0 : tool_start_);
                }
            }
            trigger_tool_execution_ = true;
            if (!feed_token()) early_exit = true;
            break;
        }

        if (tool_end_ != string::npos) {
            // Truncate generated_text_ to the end of the tool call, discarding
            // any trailing garbage (e.g., ^L spam from spurious EOG recovery).
            size_t tool_call_end = tool_end_ + strlen(FUNC_END);
            generated_text_.resize(tool_call_end);
            trigger_tool_execution_ = true;
            if (!feed_token()) early_exit = true;
            break;
        }

        // Silent-loop detection: if we're inside what looks like a tool call
        // (FUNC_START found) but no complete tool call has been found yet,
        // and we're NOT inside a parameter, the model may be generating garbage.
        // Outside parameters, tool call tags are just a few tokens.  If the
        // threshold passes without PARAM_START or FUNC_END, something is broken.
        if (tool_call_outside_param_count_ >= outside_param_limit) {
            diag("System: Stuck generating tool call after " + std::to_string(tool_call_outside_param_count_) + " tokens outside parameters.", "\033[1;31m");
            stop_generation = 1;
            stuck_in_tool_call = true;
            if (!feed_token()) early_exit = true;
            break;
        }

        // --- OUTPUT RENDERING ---
        if (!in_tool_call_stream_ && !in_thinking_block_) {
            size_t safe_len = generated_text_.length();
            string fstart(FUNC_START);
            string tstart(g_model_tokens.think_start);

            if (think_start_ != string::npos && think_end_ != string::npos) {
                size_t think_block_end = think_end_ + g_model_tokens.think_end.length();
                bool was_empty = think_buffering_;
                // If we had content (was_empty == false), output the closing tag to stdout.
                if (!was_empty && should_output_to_stdout()) {
                    console(g_model_tokens.think_end.c_str());
                    console("\n");
                    consoleFlush();
                    g_stdout_ended_with_newline = true;
                }
                think_buffering_ = true;
                if (print_pos_ < think_block_end) {
                    print_pos_ = think_block_end;
                }
                // Only skip newlines right after advancing past the think block end,
                // not on every subsequent token.
                if (was_empty && print_pos_ == think_block_end) {
                    while (print_pos_ < generated_text_.length() && generated_text_[print_pos_] == '\n') {
                        print_pos_++;
                    }
                }
            }

            // Only check the last N characters for tag prefixes via a small window.
            // Avoids touching the potentially-megabyte generated_text_ on every token.
            {
                size_t max_tag_len = max(fstart.length(), tstart.length());
                if (generated_text_.length() >= max_tag_len) {
                    string_view tail(generated_text_.data() + generated_text_.length() - max_tag_len, max_tag_len);
                    for (size_t len = 1; len <= max_tag_len; ++len) {
                        if ((tail.length() >= len && tail.compare(tail.length() - len, len, fstart.data(), len) == 0) ||
                            (tail.length() >= len && tail.compare(tail.length() - len, len, tstart.data(), len) == 0)) {
                            safe_len = generated_text_.length() - len;
                            break;
                        }
                    }
                }
            }

            if (safe_len > print_pos_) {
                unprinted_text_.append(generated_text_.data() + print_pos_, safe_len - print_pos_);
                print_pos_ = safe_len;
            }

            if (!unprinted_text_.empty() && unprinted_text_.back() == '\n') {
                console(unprinted_text_.c_str());
                consoleFlush();
                stream(unprinted_text_);
                g_stdout_ended_with_newline = (unprinted_text_.back() == '\n');
                unprinted_text_ = "";
            }
        } else if (in_thinking_block_) {
            size_t safe_len = generated_text_.length();
            string tend(g_model_tokens.think_end);

            // Exclude both opening and closing think tags from streamed output
            // so they never leak into the browser.  Full-match check first,
            // then fall back to prefix matching for partial tags split across
            // token boundaries (e.g., after spurious EOG recovery).
            if (!tend.empty()) {
                size_t last_occurrence = generated_text_.rfind(tend);
                if (last_occurrence != string::npos &&
                    last_occurrence + tend.length() == generated_text_.length()) {
                    safe_len = last_occurrence;
                } else {
                    for (size_t len = 1; len <= tend.length() && len <= generated_text_.length(); ++len) {
                        if (generated_text_.compare(generated_text_.length() - len, len, tend, 0, len) == 0) {
                            safe_len = generated_text_.length() - len;
                            break;
                        }
                    }
                }
            }

            // Also suppress partial opening think tags at the tail.
            {
                string tstart(g_model_tokens.think_start);
                if (!tstart.empty()) {
                    for (size_t len = 1; len <= tstart.length() && len <= generated_text_.length(); ++len) {
                        if (generated_text_.compare(generated_text_.length() - len, len, tstart, 0, len) == 0) {
                            if (safe_len > generated_text_.length() - len) {
                                safe_len = generated_text_.length() - len;
                            }
                            break;
                        }
                    }
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
                        think_output = think_output.substr(content_start);
                        // Output the opening think tag to stdout when content begins.
                        if (should_output_to_stdout()) {
                            console(g_model_tokens.think_start.c_str());
                            console("\n");
                            consoleFlush();
                            g_stdout_ended_with_newline = true;
                        }
                        console_think(think_output.c_str());
                        consoleThinkFlush();
                        stream_think(think_output);
                        if (!think_output.empty()) g_stdout_ended_with_newline = (think_output.back() == '\n');
                        think_buffering_ = false;
                    }
                } else {
                    _strip_think_and_tool_tags(think_output);
                    if (!think_output.empty()) {
                        console_think(think_output.c_str());
                        consoleThinkFlush();
                        stream_think(think_output);
                        g_stdout_ended_with_newline = (think_output.back() == '\n');
                    }
                }
                print_pos_ = safe_len;
            }
        } else {
            if (!unprinted_text_.empty()) {
                console(unprinted_text_.c_str());
                consoleFlush();
                stream(unprinted_text_);
                g_stdout_ended_with_newline = (unprinted_text_.back() == '\n');
                unprinted_text_ = "";
            }
            print_pos_ = generated_text_.length();
        }

        t_count_++;

        // Track tokens outside parameters while inside a tool call stream.
        // Skip the region between FUNC_START and the first PARAM_START -- that's
        // just the function name and any preamble, always legitimate.
        // Ignore form whitespace (newlines, spaces) -- those are formatting, not garbage.
        if (in_tool_call_stream_) {
            if (in_parameter_) tool_call_seen_parameter_ = true;
            if (tool_call_seen_parameter_ && !in_parameter_) {
                // Check if the token just appended is purely form whitespace.
                int piece_len = (n_chars > 0) ? n_chars : std::max(0, -n_chars - 1);
                bool is_form_ws = true;
                for (int i = 0; i < piece_len && is_form_ws; i++) {
                    if (!isspace((unsigned char)generated_text_[generated_text_.size() - piece_len + i])) {
                        is_form_ws = false;
                    }
                }
                if (!is_form_ws && !eog_recovered_this_token_) tool_call_outside_param_count_++;            }
        }

        {
            // Speed/context diagnostic for the browser status bar.
            // Update every N tokens so progress stays visible at any generation rate.
            // For stdout output (LIM_OUTPUT=1 or 3), skip mid-generation diagnostics;
            // they are printed once at turn end in session.cc before the >>> prompt.
            if (t_count_ > 5 && t_count_ % speed_update_interval == 0) {
                auto now = chrono::high_resolution_clock::now();
                double total_elapsed = chrono::duration<double>(now - start).count() + feed_time_;
                if (total_elapsed > 0) {
                    // Pick denominator based on honest_speed global
                    double denom = total_elapsed;  // default: wall clock ("honest")
                    if (!honest_speed && t_count_ >= 1) {
                        denom = chrono::duration<double>(now - t_gen_start).count();
                    }
                    if (denom > 0) {
                        // Browser status bar only: the TPS log line is written
                        // once per turn in diag_speed() at turn end.
                        if (should_output_to_browser()) {
                            stream_speed(format_speed_ctx(round_int(t_count_ / denom), n_past_, (int)cparams_.n_ctx));
                        }
                    }
                }
            }
        }

        // Reset EOG-recovery flag for the next iteration.
        eog_recovered_this_token_ = false;

        if (!feed_token()) {
            early_exit = true;
            break;
        }

        // --- MTP: post-feed round advancement ---
        if (spec_verify_pending_) {
            // d1 (the origin token) was just committed no-feed: its KV cell
            // does not exist yet.  Decode the full verify batch [d1..dk] in
            // one pass -- every position is new, so the recurrent state
            // advances through them exactly once, and each row's logits
            // verify the next draft at its own position (row 0 verifies d2,
            // ..., row k-2 verifies dk; row k-1 yields the bonus), exactly
            // like common_sampler_sample_and_accept_n.  One forward
            // evaluates every draft at once: the speculative speedup.
            spec_verify_pending_ = false;
            batch_.n_tokens = 0;
            const int k = (int)spec_draft_.size();
            for (int i = 0; i < k; i++) {
                common_batch_add(batch_, spec_draft_[i], n_past_ - 1 + i, {0}, true);
            }
            if (!handle_llama_decode_error(ctx_, batch_)) {
                // Decode failed (KV exhausted or aborted): the verify cells
                // may be partially written.  Drop the uncommitted draft
                // cells (keep d1's if it made it in), then either keep d1
                // committed -- truncating the batch to its row so a later
                // /continue samples valid logits -- or, if no verify cell
                // survived, drop d1 from the tracker (it has no KV cell)
                // and empty the batch so /continue cannot sample stale
                // logits.  MTP may have been invalidated by the hook along
                // the way.
                llama_memory_seq_rm(llama_get_memory(ctx_), 0, n_past_, -1);
                if (llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) >= n_past_ - 1) {
                    batch_.n_tokens = 1;
                    if (g_mtp) g_mtp->note_round(1);
                } else {
                    if (out_tokens_) out_tokens_->pop_back();
                    n_past_--;
                    batch_.n_tokens = 0;
                    if (g_mtp) g_mtp->note_round(0);
                }
                spec_in_progress_ = false;
                spec_bonus_pending_ = false;
                spec_round_complete_pending_ = false;
                spec_draft_.clear();
                early_exit = true;
                break;
            }
            if (spec_verify_rows_ > 0) {
                // Sampling of the verify rows starts next iteration
                // (spec_verify_row_ = 0, spec_in_progress_ already true).
            } else {
                // k = 1: no verify rows; the next sample (from the d1 row)
                // is a free bonus token.
                spec_in_progress_ = false;
                spec_bonus_pending_ = true;
                spec_round_complete_pending_ = true;
            }
        } else if (spec_round_complete_pending_) {
            // The bonus token was just fed: the round is complete.
            spec_round_complete_pending_ = false;
            if (g_mtp) g_mtp->note_round(1 + spec_verify_rows_ + 1);
        } else if (!spec_in_progress_ && !spec_bonus_pending_ && !spec_verify_pending_ &&
                   batch_.n_tokens == 1 && g_mtp && g_mtp->can_draft()) {
            // Set up the next round: draft the tokens right after the row
            // just fed (the mirror hook updated the mirror during that feed,
            // with MTP logits on the last row).
            auto drafts = g_mtp->draft();
            if (!drafts.empty()) {
                spec_draft_ = std::move(drafts);
                spec_origin_pending_ = true;
            }
        }

        // Fire the FUNC_START hook after the feed+decode: n_past is exactly
        // right after FUNC_START and the recurrent (R/S) state is up to date,
        // so a checkpoint saved by the hook matches this position.
        if (tool_start_hook_pending) {
            tool_start_hook_pending = false;
            // Fire the hook unconditionally so tool_correction_n_past always
            // tracks the latest FUNC_START position.  The lockstep flag tells
            // the hook whether the R/S state matches n_past_ (safe to save) or
            // is ahead (uncommitted verify cells -- skip the checkpoint save;
            // the correction path will eject instead of rolling back to a stale
            // position).
            if (on_tool_start_) {
                bool lockstep = !(spec_in_progress_ && spec_committed_verify_ < spec_verify_rows_);
                on_tool_start_(lockstep);
            }
        }
    } // END INNER TOKEN LOOP

    // --- MTP: post-loop round abort cleanup ---
    // The loop only exits via break.  If a spec round was left active (tool
    // call completed mid-round, EOG end-of-turn, interrupt, timeout, context
    // exhaustion), roll back any uncommitted draft cells and close the round
    // so the KV/tracker state is exactly the committed prefix.
    if (spec_verify_pending_ || spec_in_progress_ || spec_bonus_pending_ ||
        spec_round_complete_pending_ || spec_origin_pending_) {
        if (spec_verify_pending_) {
            // The origin token was committed no-feed but the verify batch
            // never decoded (a break -- e.g. a completed tool call -- landed
            // between the commit and the post-feed advancement).  Give it
            // its KV cell with a single-row decode: position n_past_-1 is
            // new, so this is a plain decode and the recurrent state lands
            // in lockstep.  The emitted context (a tool call included)
            // stays complete.
            spec_verify_pending_ = false;
            batch_.n_tokens = 0;
            common_batch_add(batch_, spec_draft_[0], n_past_ - 1, {0}, true);
            if (!handle_llama_decode_error(ctx_, batch_)) {
                // The token has no guaranteed KV cell now: roll back to the
                // pre-round state (drops a partially written cell if any)
                // and drop it from the tracker to keep the invariant.
                n_past_--;
                llama_memory_seq_rm(llama_get_memory(ctx_), 0, n_past_, -1);
                if (out_tokens_) out_tokens_->pop_back();
                early_exit = true;
            }
            spec_in_progress_ = false;
            spec_bonus_pending_ = false;
            spec_round_complete_pending_ = false;
            spec_draft_.clear();
        }
        // Stale verify cells exist only when the verify batch was decoded
        // but not all of its rows were committed (n_past_ is behind the
        // verify batch tail).  An empty range (nothing decoded, or all
        // committed) is a no-op.
        if (spec_in_progress_ && spec_committed_verify_ < spec_verify_rows_) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx_), 0, n_past_, -1)) {
                diag("MTP: abort rollback failed; regenerating KV cache", "\033[31m");
                if (!spec_rollback_redecode()) {
                    early_exit = true;
                }
            } else {
                // The verify batch's rows past the last committed one no
                // longer exist: truncate the batch so a later /continue
                // samples the last valid row (its logits predict n_past_),
                // not a rolled-back row's stale logits.
                batch_.n_tokens = spec_committed_verify_ + 1;
            }
        }
        if (g_mtp) {
            if (spec_origin_pending_) {
                // The break landed before the origin comparison ran: count
                // the origin as not matched so the origin_ok denominator
                // always equals the round count.
                g_mtp->on_origin_mismatch();
            }
            if (spec_round_complete_pending_) {
                // Bonus was pending but never fed (break before its feed).
                g_mtp->note_round(1 + spec_verify_rows_);
            } else {
                // A round left at the origin (spec_origin_pending_: the turn
                // ended before the comparison block ran) committed its one
                // fed token.  (A top-of-loop exit before the origin sample
                // overcounts by one -- rare, and the harmless direction.)
                g_mtp->note_round(1 + spec_committed_verify_);
            }
            g_mtp->on_abort(spec_committed_verify_, spec_verify_rows_);
        }
        spec_verify_pending_ = spec_in_progress_ = spec_bonus_pending_ = false;
        spec_round_complete_pending_ = false;
        spec_no_feed_ = false;
        spec_origin_pending_ = false;
        spec_verify_row_ = 0;
        spec_committed_verify_ = 0;
        spec_draft_idx_ = 0;
        spec_draft_.clear();
    }

    // Compute wall-clock generation time (first sample -> last sample).
    if (!honest_speed && t_count_ > 0) {
        gen_wall_time = chrono::duration<double>(t_gen_end - t_gen_start).count();
    }

    // Flush remaining unprinted text
    if (!unprinted_text_.empty()) {
        if (unprinted_text_.back() != '\n') {
            console((unprinted_text_ + "\n").c_str());
            consoleFlush();
            stream(unprinted_text_ + "\n");
            g_stdout_ended_with_newline = true;
        } else {
            console(unprinted_text_.c_str());
            consoleFlush();
            stream(unprinted_text_);
            g_stdout_ended_with_newline = true;
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
    result.stuck_in_tool_call = stuck_in_tool_call;
    result.ended_on_eog = ended_on_eog;
    result.was_in_thinking_block = in_thinking_block_;
    result.decode_time = gen_wall_time;

    return result;
}

// Fallback for a failed MTP draft rollback: clear the main KV cache and
// re-decode the committed prefix (the token tracker holds exactly the tokens
// present in the KV).  The MTP mirror rebuilds via the mirror hook on each
// re-decoded chunk, so the speculator stays consistent (and, if it was the
// cause of the failure, has been invalidated by the hook).
bool TokenGenerator::spec_rollback_redecode() {
    llama_memory_clear(llama_get_memory(ctx_), true);
    n_past_ = 0;
    if (!out_tokens_) return false;
    const size_t n = out_tokens_->size();
    for (size_t i = 0; i < n; i += (size_t)cparams_.n_batch) {
        const size_t chunk = std::min((size_t)cparams_.n_batch, n - i);
        batch_.n_tokens = 0;
        for (size_t j = 0; j < chunk; j++) {
            common_batch_add(batch_, (*out_tokens_)[i + j], n_past_++, {0}, false);
        }
        if (!handle_llama_decode_error(ctx_, batch_, "KV Cache Exhausted (MTP rollback re-decode).", true)) {
            sync_n_past(ctx_, n_past_);
            return false;
        }
        batch_.n_tokens = 0;
    }
    return n_past_ == (int)n;
}
