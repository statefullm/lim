#ifndef TOKEN_GENERATOR_H
#define TOKEN_GENERATOR_H

#include "llama.h"
#include "common.h"
#include <functional>
#include <string>
#include <vector>

// Helper to escape token piece strings for token log
std::string escape_token_piece(const std::string& s);

// Helper functions for tool call detection (shared between session.cc and token_generator.cc)
size_t find_tool_end_robust(const std::string& text, size_t from_pos, bool* out_in_parameter = nullptr);
void repair_malformed_tool_end(std::string& text, size_t pos);
void _strip_think_and_tool_tags(std::string& str);

class TokenGenerator {
public:
  struct Result {
    std::string text;
    bool has_tool_call = false;
    size_t tool_start = std::string::npos;
    size_t tool_end = std::string::npos;
    bool was_interrupted = false;
    // True when the turn was force-ended at the 90% context threshold
    // (housekeeping interrupt): the session records a PERMANENT /undo
    // checkpoint at this position (labeled with the "/continue" placeholder)
    // and auto-resumes the turn without dropping to the prompt; the resumed
    // turn's end checkpoint keeps the turn's prompt label.  The LLM never
    // sees the break.
    bool ctx_limit_interrupt = false;
    int token_count = 0;
    bool early_exit = false;  // context exhaustion or decode error (not normal EOG)
    double decode_time = 0.0;  // Sum of per-token decode intervals (seconds)
    // True when the silent-loop detector aborted generation inside a tool call
    // (FUNC_START present, FUNC_END never completed). The caller should run the
    // standard tool-correction cycle instead of ejecting to prompt.
    bool stuck_in_tool_call = false;
    // True when generation ended on a non-recovered EOG: the EOG token WAS
    // fed (decoded + appended to the out_tokens tracker) but its piece was NOT
    // appended to `text` (the break precedes the detokenize block).  Callers
    // maintaining a canonical text form of the tracker must mirror this, or
    // their re-tokenization comes up one token short at every turn boundary.
    bool ended_on_eog = false;
    // True when generation stopped (interrupt) with the thinking block still
    // open.  /continue uses it to resume the generator in thinking mode so
    // the viewer re-enters the thinking display.
    bool was_in_thinking_block = false;
    // Canary mode: true when the silent capped probe finished without an
    // early exit (decode error / interrupt) -- the origin counts below are a
    // valid verdict.  With canary_origin_total below the caller's judged-
    // round floor there is not enough signal to judge (e.g. the probe was
    // truncated by an unrecovered EOG at a natural turn end): the caller
    // treats the probe as clean (no oracle).
    bool canary_ran = false;
    // Canary mode: origin-match counts over the judged rounds (2..N; round 1
    // is the mirror heal point, and EOG-recovered samples are excluded, so
    // these never count).
    int canary_origin_match = 0;
    int canary_origin_total = 0;
  };

  TokenGenerator(llama_context* ctx, const llama_vocab* vocab,
                 llama_sampler* smpl, llama_batch& batch,
                 int& n_past, const llama_context_params& cparams,
                 double turn_timeout_sec, bool was_mid_tool_call,
                 int last_n_past, int& last_context,
                 std::vector<llama_token>* out_tokens = nullptr,
                 double feed_time = 0.0,
                 bool is_reincarnating = false,
                 std::function<void(bool lockstep)> on_tool_start = nullptr,
                 bool was_mid_thinking_block = false,
                 int canary_max_tokens = 0,
                 bool canary_silent = false);

  Result generate();

private:
  llama_context* ctx_;
  const llama_vocab* vocab_;
  llama_sampler* smpl_;
  llama_batch& batch_;
  int& n_past_;
  // Context position at the previous 90% check (the "old" side of the
  // crossing test).  Session-owned so it survives across the generations of
  // a turn (a feed between generations can cross the line without any check
  // running); generate() updates it on every check.
  int& last_context_;
  const llama_context_params& cparams_;
  double turn_timeout_sec_;
  double feed_time_;  // Time spent feeding/re-decoding tokens before generation

  // Internal state
  std::string generated_text_;
  std::string unprinted_text_;
  size_t print_pos_;
  bool in_tool_call_stream_;
  bool in_parameter_;
  size_t tool_start_;
  size_t tool_end_;
  bool trigger_tool_execution_;
  size_t func_search_pos_;
  bool context_warned_this_turn_;
  bool in_thinking_block_;
  size_t think_start_;
  size_t think_end_;
  int think_depth_;        // open think tags in current block (1 = outer)
  size_t think_scan_pos_;  // incremental scan position for nested think-tag matching
  bool think_buffering_;
  int t_count_;
  // Fired once per generation, after the token that completes FUNC_START has
  // been fed+decoded (n_past is exactly right after FUNC_START). Null when no
  // hook is set (e.g. the correction's own regeneration).
  std::function<void(bool lockstep)> on_tool_start_;
  int last_n_past_;
  bool was_mid_tool_call_;
  bool is_reincarnating_;
  std::vector<llama_token>* out_tokens_;  // If non-null, each sampled token is appended here

  // Silent-loop detector: count tokens generated outside parameters while
  // inside a tool call stream (FUNC_START found, FUNC_END not yet).
  int tool_call_outside_param_count_ = 0;
  // Whether we've seen at least one PARAM_START in the current tool call.
  // Tokens between FUNC_START and the first PARAM_START are not counted,
  // since that region is just the function name / preamble.
  bool tool_call_seen_parameter_ = false;
  // True when the token sampled this iteration came from EOG resampling.
  // Those tokens are spurious artifacts and should not count toward the
  // silent-loop detector.
  bool eog_recovered_this_token_ = false;

  // --- MTP speculative decoding round state (design in mtp.h) ---
  // A round drafts d1..dk for the positions right after the last committed
  // token.  d1 is compared with the next sampled token (origin); on a match
  // d1 is committed WITHOUT a decode and the verify batch [d1..dk] is
  // decoded in one pass (d1's cell comes from it), so each verify row r
  // samples the token right after draft d_{r+1} and is compared with the
  // draft at its own position: matched tokens are committed without a
  // decode (their KV cell already holds the identical token), the first
  // mismatch rolls the context back (n_rs_seq window) and feeds the
  // target's own token.  A free bonus token is sampled from the last verify
  // row when all drafts match.
  std::vector<llama_token> spec_draft_;
  bool spec_origin_pending_ = false;    // next sample (default row) vs d1
  bool spec_in_progress_ = false;       // sampling verify rows (batch_ = V)
  bool spec_bonus_pending_ = false;     // next sample (last V row) = bonus
  bool spec_verify_pending_ = false;    // decode V after this iteration's feed
  bool spec_round_complete_pending_ = false; // round closes when bonus is fed
  bool spec_no_feed_ = false;           // this iteration's feed = no-decode advance
  int spec_verify_row_ = 0;             // verify row to sample this iteration
  int spec_verify_rows_ = 0;            // V row count (k - 1)
  int spec_draft_idx_ = 0;              // 1-based draft idx of current comparison
  int spec_committed_verify_ = 0;       // verify rows committed so far

  // --- Canary mode (pre-turn MTP-mirror validation of a fast restore) ---
  // canary_max_tokens_ > 0: generation is a silent probe capped at N
  // sampled tokens.  canary_silent_: suppress all user-visible output
  // (stdout, browser pipe, token log, speed bar) and the pre-turn
  // housekeeping checks (timeout, 90%-context force-end) for the probe.
  int canary_max_tokens_ = 0;
  bool canary_silent_ = false;
  // Canary round accounting: 1-based index of the origin comparison within
  // this generation; the counters tally origin matches over judged rounds
  // (2..N -- round 1 is the mirror heal point and never counts).
  int canary_round_ = 0;
  int canary_origin_match_ = 0;
  int canary_origin_total_ = 0;

  // Fallback for a failed draft rollback: clear the main KV and re-decode
  // the committed prefix (the token tracker) -- the MTP mirror rebuilds via
  // the mirror hook along the way.  Returns false on decode failure.
  bool spec_rollback_redecode();
};

#endif // TOKEN_GENERATOR_H
