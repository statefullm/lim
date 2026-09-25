
#ifndef MTP_H
#define MTP_H

// MTP (multi-token-prediction) speculative decoding for Qwen3.5/3.8-style
// models that ship a dedicated "nextn" draft head.
//
// Design (mirrors llama.cpp's common_speculative draft-mtp, adapted to LIM's
// immediate-feed token loop):
//
//   * A second llama_context over the SAME model (ctx_type =
//     LLAMA_CONTEXT_TYPE_MTP) holds a 1:1 positional mirror of the main KV
//     cache restricted to the single full-attention MTP layer.  Mirror cell p
//     stores (x_p, h_{p-1}) where h is the main model's final hidden state
//     (extracted per batch via llama_set_embeddings_nextn on the main ctx).
//     The mirror is pure attention: no recurrent state, no checkpointing.
//
//   * After every main-context decode, process() (hooked from
//     handle_llama_decode_error) extends the mirror with that batch's rows.
//     The last row of each batch gets MTP logits so the draft can sample its
//     first proposal from the existing mirror row (no re-decode).
//
//   * draft() proposes up to draft_len tokens: the first from the existing
//     mirror row (logits already computed), the rest by chaining the MTP head
//     (each chain row consumes the previous chain row's MTP output hidden).
//
//   * The main model verifies the drafts in ONE batch decode (the verify
//     batch).  The origin token (d1) is committed without its own decode:
//     the verify batch decodes [d1..dk] at fresh positions in one pass, so
//     d1's row supplies the logits that sample d2's position, and every
//     draft is verified against the target's sample at its own position --
//     the same protocol as common_sampler_sample_and_accept_n.  Each
//     committed token is a fresh sample from the full target sampler chain
//     at its verify row, compared against the draft: accept on match (the
//     KV cell already holds the identical token), stop at the first
//     mismatch (roll the main context back with seq_rm -- supported for the
//     hybrid recurrent state by the n_rs_seq snapshot window -- and feed
//     the target's own token).  Because the verify pass IS the normal
//     sampling code path, each committed token is a fresh sample from the
//     full target chain at its own verify row: distribution-exact under
//     any sampling configuration.  (Not token-identical to a plain run: the
//     verify batch decodes k drafts in ONE forward, whose GEMM/attention
//     shapes differ from k sequential decodes -- last-ulp logit differences,
//     amplified through the hybrid recurrent state.  A greedy MTP run
//     first diverges from plain at a close-argmax flip, both continuations
//     fluent.  Determinism check: two MTP runs at temperature 0 ARE
//     token-identical to each other.)
//
//   * Rejected draft cells in the mirror are never read again: the next
//     draft round re-derives from the last committed row, so the mirror
//     self-heals without explicit rollback.

#include "llama.h"
#include <string>
#include <vector>

class MtpSpeculator {
public:
  // Create the MTP draft context.  Returns nullptr on failure (no MTP head in
  // the model, context creation failed, model does not support RS rollback).
  // On success also enables hidden-state extraction on ctx_main.
  // draft_batch caps the draft context's batch size (LIM_MTP_BATCH): its
  // scheduler work buffers scale with this and dominate the draft context's
  // VRAM cost, while nothing ever decodes more rows than this at once (the
  // draft chain is single-row; the mirror prefill just runs in more chunks).
  static MtpSpeculator* create(llama_context* ctx_main, llama_model* model,
                               const llama_context_params& cparams_main,
                               int draft_len,
                               int draft_batch);

  ~MtpSpeculator();

  bool valid() const { return valid_; }
  llama_pos mirrorPos() const { return mirror_pos_; }

  // True when ctx is the main context this speculator mirrors.
  bool owns(llama_context* ctx) const { return ctx == ctx_main_; }

  // The draft context whose KV mirrors the main context (nullptr when no draft
  // context).  Exposed so the V1 fast-restore cache can persist/restore the
  // mirror KV alongside the main KV, keeping MTP enabled across a fast restore.
  llama_context* draft_ctx() const { return ctx_dft_; }

  // Re-arm the speculator after its mirror KV was restored from a V1 cache:
  // the mirror rows are exactly the saved session's rows, so it matches the
  // restored main context again.  Hidden-state bookkeeping (pending_h_, origin)
  // is refreshed by the first forward feed via process()'s position guard (one
  // slightly-off re-mirrored row at the heal point -- see the restore/rollback
  // comments in session.cc); any stale draft rows past n_past_ are trimmed by
  // that same guard.
  void on_mirror_loaded();

  // Mirror hook: extend the mirror with the rows of a main-context decode.
  // No-op when invalid.  On MTP decode failure the speculator invalidates
  // itself (falls back to normal decoding).
  void process(const llama_batch& batch);

  // Clear the mirror (main context was cleared).  Re-arms the speculator:
  // the mirror rebuilds from the next main decode.
  void clear();

  // Invalidate: the mirror no longer matches the main context and can't be
  // healed by the process() position guard (e.g. a KV-exhausted truncation, or
  // a fast restore whose cache held no mirror state).  Drafting stops until the
  // next clear() or a full re-decode re-mirrors the context.
  void invalidate(const std::string& reason);

  // Draft up to draft_len tokens for the positions right after the last main
  // row (which must have been mirrored by process() and carry MTP logits).
  // Returns an empty vector when invalid or on failure.
  std::vector<llama_token> draft();

  // --- round bookkeeping (stats; acceptance logic lives in token_generator) ---
  void on_origin_match();
  void on_origin_mismatch();
  void on_verify_match();
  void on_verify_mismatch(int draft_idx); // 1-based index of the rejected draft
  void on_bonus();
  void on_abort(int committed_verify_rows, int verify_rows);

  // True when a new round can start (valid + origin ready).
  bool can_draft() const;

  // Report how many tokens a finished round committed (>= 0; 0 only for a
  // round aborted by a failed verify decode, including any bonus otherwise).
  // Feeds the adaptive auto-disable: sustained low acceptance means the
  // draft overhead is not paying for itself.
  void note_round(int committed);

  // Per-turn stats line for the .tps log, e.g.
  //   # MTP: rounds=120 drafted=479 matched=351 mean=2.85 full=9%
  // Returns false when no MTP rounds happened this turn.
  bool stats_line(std::string& out);
  void reset_stats();

  // Ratio of origin matches over origin comparisons since the last reset
  // (0..1; 0 when no origin comparison ran).  Convenience for verdicts such
  // as the pre-turn canary (the canary itself tallies judged rounds 2..N in
  // the generator, since round 1 is the heal point).
  double origin_ratio() const {
    const uint64_t total = s_origin_match_ + s_origin_mismatch_;
    return total > 0 ? (double)s_origin_match_ / (double)total : 0.0;
  }

  // One-line summary (startup, invalidation, turn end) to stderr via diag().
  void log_summary(const std::string& tag);

private:
  MtpSpeculator() = default;

  llama_context* ctx_main_ = nullptr;
  llama_context* ctx_dft_ = nullptr;
  llama_sampler* mtp_smpl_ = nullptr;   // top-k(10) proposal chain
  llama_batch mtp_batch_ = {};          // dual token+embd batch (like the reference)

  int32_t n_embd_ = 0;
  int draft_len_ = 4;
  int n_batch_mtp_ = 512;

  // Mirror bookkeeping (all updated by process()):
  llama_pos mirror_pos_ = -1;           // position of the last main row
  llama_token last_token_ = 0;          // token of the last main row
  bool origin_has_logits_ = false;      // last main row mirrored with MTP logits
  int32_t origin_row_idx_ = 0;          // batch row index of that row in the
                                        // last MTP decode (for sampling)
  std::vector<float> pending_h_;        // main h of the last main row (for the
                                        // next mirror row's embd)
  std::vector<float> mtp_h_last_;       // MTP output hidden of the last main row
                                        // (first chain row's embd)
  std::vector<float> mtp_last_embd_;    // main h used as the last row's embd
                                        // (= previous pending_h_; for the rare
                                        // origin re-decode fallback)

  bool valid_ = false;
  bool warned_ = false;                 // invalidation warning printed once

  // --- stats (per turn) ---
  uint64_t s_rounds_ = 0;               // rounds started (drafts produced)
  uint64_t s_drafted_ = 0;              // draft tokens proposed
  uint64_t s_origin_match_ = 0;
  uint64_t s_origin_mismatch_ = 0;
  uint64_t s_verify_match_ = 0;
  uint64_t s_verify_mismatch_ = 0;
  uint64_t s_bonus_ = 0;
  uint64_t s_aborted_ = 0;
  uint64_t s_committed_ = 0;            // tokens committed via MTP rounds
  // Adaptive off: trailing committed-per-round average.
  int trailing_committed_ = 0;          // sum over last ADAPT_WINDOW rounds
  int trailing_rounds_ = 0;
  int low_streak_ = 0;
};

// Global speculator (nullptr when MTP is disabled).  Set in main.cc.
extern MtpSpeculator* g_mtp;

#endif // MTP_H

