
#include "mtp.h"
#include "common.h"
#include "llama/src/llama-ext.h"

#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

// Forward declarations for diagnostics (defined in main.cc)
extern void diag(const std::string& msg, const char* color);
extern bool is_debug;

// Adaptive auto-disable: if the trailing committed-per-round average stays
// below ADAPT_MIN for ADAPT_STREAK consecutive windows, the draft overhead
// exceeds its benefit and MTP turns itself off (honest fallback).
static constexpr int ADAPT_WINDOW = 32;
static constexpr double ADAPT_MIN = 1.25;
static constexpr int ADAPT_STREAK = 2;

MtpSpeculator* g_mtp = nullptr;

MtpSpeculator* MtpSpeculator::create(llama_context* ctx_main, llama_model* model,
                                     const llama_context_params& cparams_main,
                                     int draft_len,
                                     int draft_batch) {
  if (ctx_main == nullptr || model == nullptr) return nullptr;
  if (llama_model_n_layer_nextn(model) < 1) {
    diag("MTP: model has no nextn (MTP) layers -- MTP disabled", "\033[33m");
    return nullptr;
  }
  // The verify rollback relies on the n_rs_seq recurrent-snapshot window;
  // without it a partial acceptance could not restore the hybrid state.
  if (llama_n_rs_seq(ctx_main) < 1) {
    diag("MTP: main context has no recurrent rollback window (n_rs_seq=0) -- MTP disabled", "\033[33m");
    return nullptr;
  }
  draft_len = std::max(1, std::min(32, draft_len));
  if (draft_len > (int)llama_n_rs_seq(ctx_main)) {
    diag("MTP: draft length " + std::to_string(draft_len) + " exceeds the rollback window (" +
         std::to_string(llama_n_rs_seq(ctx_main)) + ") -- MTP disabled", "\033[33m");
    return nullptr;
  }

  auto* self = new MtpSpeculator();
  self->ctx_main_ = ctx_main;
  self->draft_len_ = draft_len;

  self->n_embd_ = llama_model_n_embd_out(model);
  if (self->n_embd_ <= 0) {
    diag("MTP: cannot determine output embedding width -- MTP disabled", "\033[33m");
    delete self;
    return nullptr;
  }
  self->pending_h_.assign((size_t)self->n_embd_, 0.0f);
  self->mtp_h_last_.assign((size_t)self->n_embd_, 0.0f);
  self->mtp_last_embd_.assign((size_t)self->n_embd_, 0.0f);

  // Draft context: a second context over the same model, MTP layer only.
  // (For QWEN35/3NEXT the memory factory allocates a plain attention KV cache
  // filtered to the MTP layer -- no recurrent state, no hybrid wrapper.)
  llama_context_params mc = cparams_main;
  mc.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
  mc.n_rs_seq = 0;
  mc.n_seq_max = 1;
  mc.n_ctx = cparams_main.n_ctx;
  // Sidecar/shared-head models borrow tok_embd/output from the main context.
  mc.ctx_other = ctx_main;
  // Cap the draft context's batch (LIM_MTP_BATCH, default 128): the
  // scheduler's work buffers (GEMM/attention scratch) scale with it and
  // dominate the draft context's VRAM cost (~1.3 GB at 512 rows on a 27B
  // model).  Nothing decodes more rows than this at once: draft() chains
  // single-row decodes and process() mirrors in n_batch_mtp_ chunks, so a
  // smaller cap only splits the one-time prefill mirror into more
  // single-layer chunks.
  mc.n_batch = std::min<int>(cparams_main.n_batch, draft_batch);
  mc.n_ubatch = std::min<int>(cparams_main.n_ubatch, draft_batch);
  mc.n_outputs_max = 2;
  mc.n_outputs_max_per_seq = 2;
  self->n_batch_mtp_ = mc.n_batch;

  // Mirror KV quantization: always Q8_0, independent of the main cache
  // type.  The draft context holds a SINGLE layer's KV and every proposal
  // is verified by the main model, so the mirror's precision only costs
  // acceptance rate (the verify pass re-samples every committed token from
  // the full-precision target chain) -- never output correctness.  Q8_0 is
  // the most compact type the MMA_F16 flash-attention kernel natively
  // dequantizes: F16 doubles the mirror's size, and other quant types
  // trigger the full-cache F16 pre-conversion whose f16_extra buffer scales
  // with the context.  A Q8_0 mirror works with an F16 main KV -- the MTP
  // layer's inputs are full-precision hidden states, so the quantized-
  // mirror acceptance cost stays second-order.
  mc.type_k = GGML_TYPE_Q8_0;
  mc.type_v = GGML_TYPE_Q8_0;

  // Mirror KV estimate (pre-creation): the draft context holds ONE
  // full-attention layer's KV (n_head_kv x head_dim, keys + values) per
  // context token.  Used to report a sizing hint when the draft context
  // fails to allocate.
  //
  // Head dimensions: prefer the model's GGUF metadata (arch-agnostic
  // suffix match on attention.key_length / attention.value_length -- the
  // explicit key is mandatory when n_embd is not a multiple of n_head,
  // e.g. qwen35-27B: 5120/24 is not integral, key_length=256); fall back
  // to n_embd / n_head when it divides evenly.
  double est_kv_bytes = 0.0;
  {
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    int64_t head_k = -1, head_v = -1;
    {
      char key_buf[128], val_buf[32];
      const int32_t n_meta = llama_model_meta_count(model);
      for (int32_t i = 0; i < n_meta; i++) {
        if (llama_model_meta_key_by_index(model, i, key_buf, sizeof(key_buf)) <= 0) continue;
        const std::string k = key_buf;
        auto suffix_val = [&k, model, i](const char* suffix) -> int64_t {
          const size_t n = strlen(suffix);
          if (k.size() <= n || k.compare(k.size() - n, n, suffix) != 0) return -1;
          char vbuf[32];
          return llama_model_meta_val_str_by_index(model, i, vbuf, sizeof(vbuf)) > 0
                 ? atoi(vbuf) : -1;
        };
        if (head_k < 0) head_k = suffix_val(".attention.key_length");
        if (head_v < 0) head_v = suffix_val(".attention.value_length");
        if (head_k > 0 && head_v > 0) break;
      }
    }
    if (head_k < 0 || head_v < 0) {
      const int32_t n_embd_trunk = llama_model_n_embd(model);
      const int32_t n_head = llama_model_n_head(model);
      if (n_head > 0 && n_embd_trunk > 0 && n_embd_trunk % n_head == 0) {
        head_k = head_v = n_embd_trunk / n_head;
      }
    }
    if (n_head_kv > 0 && head_k > 0 && head_v > 0) {
      // Bytes per value from one quantization block (32 elements covers
      // every type LIM accepts: F16 blk=1, Q4_0/Q5_*/Q8_* blk=32).
      const double bpp_k = ggml_row_size(mc.type_k, 32) / 32.0;
      const double bpp_v = ggml_row_size(mc.type_v, 32) / 32.0;
      est_kv_bytes = (double)mc.n_ctx * (double)n_head_kv *
                     ((double)head_k * bpp_k + (double)head_v * bpp_v);
    }
  }
  auto kv_hint = [](double bytes, int n_ctx) {
    char buf[160];
    snprintf(buf, sizeof(buf), "%.2f GB for %d tokens (%.1f MB per 1000 tokens)",
             bytes / (1024.0 * 1024.0 * 1024.0), n_ctx,
             bytes / (1024.0 * 1024.0) * 1000.0 / (double)n_ctx);
    return std::string(buf);
  };

  self->ctx_dft_ = llama_init_from_model(model, mc);
  if (!self->ctx_dft_) {
    std::string msg = "MTP: failed to create MTP draft context (out of memory?) -- MTP disabled";
    if (est_kv_bytes > 0) {
      msg += "; estimated mirror KV: " + kv_hint(est_kv_bytes, mc.n_ctx) +
             "; total also includes compute buffers (scale with the draft batch, " +
             std::to_string(mc.n_batch) + " rows) -- lower LIM_CTX and/or "
             "LIM_MTP_BATCH to make room";
    }
    diag(msg, "\033[33m");
    delete self;
    return nullptr;
  }

  // Report the mirror's actual memory footprint (measured, not estimated):
  // the KV cache plus this context's compute buffers.  Model weights are
  // shared with the main context and are not part of the incremental cost.
  // The KV scales linearly with LIM_CTX (shrinking the context by 10k
  // tokens frees that much VRAM again); the compute buffers scale with the
  // draft batch (LIM_MTP_BATCH), not with the context.  Diagnostic detail:
  // printed only with LIM_DEBUG=1 (the OOM path prints its own estimate).
  if (is_debug) {
    size_t kv_bytes = 0, compute_bytes = 0;
    for (const auto & entry : llama_get_memory_breakdown(self->ctx_dft_)) {
      kv_bytes += entry.second.context;
      compute_bytes += entry.second.compute;
    }
    char buf[280];
    snprintf(buf, sizeof(buf),
             "MTP: draft context memory = %.2f GB for %d tokens "
             "(KV %.2f GB -- %.1f MB per 1000 tokens, scales with LIM_CTX; "
             "compute buffers %.0f MB -- scale with LIM_MTP_BATCH=%d)",
             (double)(kv_bytes + compute_bytes) / (1024.0 * 1024.0 * 1024.0),
             mc.n_ctx,
             (double)kv_bytes / (1024.0 * 1024.0 * 1024.0),
             (double)kv_bytes / (1024.0 * 1024.0) * 1000.0 / (double)mc.n_ctx,
             (double)compute_bytes / (1024.0 * 1024.0),
             mc.n_batch);
    diag(buf, "\033[32m");
  }

  // MTP ctx: extract its own pre-norm hidden (masked: rows with logits only)
  // for the draft chain.  Main ctx: extract the final hidden state for ALL
  // rows (unmasked, dense) -- this is the mirror's shifted embd input.
  llama_set_embeddings_nextn(self->ctx_dft_, true, /*masked*/ true);
  llama_set_embeddings_nextn(ctx_main, true, /*masked*/ false);

  // Proposal sampler: top-k(10) then greedy (argmax).  The raw
  // llama_sampler_sample builds cur_p with selected=-1, runs the chain, and
  // asserts selected is in range; top-k only narrows/sorts and never selects,
  // so a selecting sampler must be appended.  Greedy picks the argmax of the
  // top-10 (= the global argmax), i.e. the intended "greedy-ish proposal".
  // The draft distribution only affects acceptance rate, never exactness (the
  // verify pass re-samples each committed token from the full target chain).
  self->mtp_smpl_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(self->mtp_smpl_, llama_sampler_init_top_k(10));
  llama_sampler_chain_add(self->mtp_smpl_, llama_sampler_init_greedy());

  // Dual token+embd batch (llama_batch_init allocates only one of the two).
  self->mtp_batch_ = llama_batch_init(self->n_batch_mtp_, self->n_embd_, 1);
  self->mtp_batch_.token = (llama_token*)std::malloc(sizeof(llama_token) * (size_t)self->n_batch_mtp_);
  if (self->mtp_batch_.token == nullptr) {
    diag("MTP: batch allocation failed -- MTP disabled", "\033[33m");
    delete self;
    return nullptr;
  }

  self->valid_ = true;
  diag("MTP: speculative decoding enabled (draft=" + std::to_string(draft_len) +
       ", mirror ctx " + std::to_string(mc.n_ctx) + " tokens, " +
       std::string(ggml_type_name(mc.type_k)) + "/" +
       std::string(ggml_type_name(mc.type_v)) + " KV)", "\033[32m");
  return self;
}

MtpSpeculator::~MtpSpeculator() {
  // The token array was malloc'd by us (llama_batch_init allocates only one
  // of token/embd); free it before llama_batch_free would free it again.
  if (mtp_batch_.token != nullptr) {
    std::free(mtp_batch_.token);
    mtp_batch_.token = nullptr;
  }
  llama_batch_free(mtp_batch_);
  if (mtp_smpl_) llama_sampler_free(mtp_smpl_);
  if (ctx_dft_) llama_free(ctx_dft_);
}

void MtpSpeculator::process(const llama_batch& batch) {
  if (!valid_ || batch.n_tokens <= 0 || batch.token == nullptr || batch.embd != nullptr) {
    return;
  }
  const float* h = llama_get_embeddings_nextn(ctx_main_);
  if (!h) {
    invalidate("main context returned no nextn hidden states");
    return;
  }

  const size_t row_bytes = (size_t)n_embd_ * sizeof(float);
  const int n = batch.n_tokens;

  // Position guard: the mirror KV requires strictly-increasing positions
  // (a new row must start after the mirror's current max, X < Y; a
  // non-increasing write is a hard ret=-1).  Two writers advance the mirror
  // independently of the main context, so an incoming batch can start at or
  // behind the mirror's current max:
  //   * draft() chains the MTP head over the positions right after the last
  //     committed row, pushing the mirror ahead of main.  The verify batch
  //     (decoded at exactly those positions) then re-mirrors onto the rows
  //     draft() already wrote -- this collides on every round;
  //   * a mismatch/abort rollback (and /undo, fast-restore) truncates the
  //     main context without touching the mirror, so the next forward feed
  //     lands behind the mirror's stale max.
  // When that happens, roll the mirror back to the batch's first position and
  // let the decode below rebuild those rows.  The batch's embd (the main
  // hidden states) is the canonical mirror content, so this self-heals the
  // mirror for every desync source at the single point where it is written.
  // (The guard runs once, before the chunked loop, on the whole batch's min
  // position, so it holds for the first row of every chunk.)
  {
    llama_pos batch_min_pos = batch.pos[0];
    for (int i = 1; i < n; i++) {
      if (batch.pos[i] < batch_min_pos) batch_min_pos = batch.pos[i];
    }
    if (batch_min_pos <= llama_memory_seq_pos_max(llama_get_memory(ctx_dft_), 0)) {
      if (!llama_memory_seq_rm(llama_get_memory(ctx_dft_), 0, batch_min_pos, -1)) {
        invalidate("MTP mirror rollback failed (seq_rm)");
        return;
      }
    }
  }

  origin_has_logits_ = false;
  for (int start = 0; start < n; start += n_batch_mtp_) {
    const int chunk = std::min(n_batch_mtp_, n - start);
    common_batch_clear(mtp_batch_);
    for (int i = 0; i < chunk; i++) {
      const int g = start + i;
      const bool is_last = (g == n - 1);
      common_batch_add(mtp_batch_, batch.token[g], batch.pos[g], {0}, is_last);
      // Mirror cell (x_g, h_{g-1}): the embd is the main hidden state of the
      // previous row (pending_h_ for the very first row of the context).
      const float* embd = (g == 0) ? pending_h_.data() : h + (size_t)(g - 1) * n_embd_;
      std::memcpy(mtp_batch_.embd + (size_t)i * n_embd_, embd, row_bytes);
    }
    const int rc = llama_decode(ctx_dft_, mtp_batch_);
    if (rc != 0) {
      invalidate("MTP mirror decode failed (rc=" + std::to_string(rc) + ")");
      return;
    }
    if (start + chunk == n) {
      // The last main row carries MTP logits: the draft's first proposal is
      // sampled from this existing mirror row with no extra decode.
      origin_row_idx_ = mtp_batch_.n_tokens - 1;
      origin_has_logits_ = true;
      const float* mh = llama_get_embeddings_nextn_ith(ctx_dft_, origin_row_idx_);
      if (mh) mtp_h_last_.assign(mh, mh + n_embd_);
    }
  }

  // Update hidden-state bookkeeping.  mtp_last_embd_ keeps the embd that the
  // last row was written with (previous pending_h_), for the rare origin
  // re-decode fallback in draft().
  const float* h_last = h + (size_t)(n - 1) * n_embd_;
  const float* h_prev = (n >= 2) ? h + (size_t)(n - 2) * n_embd_ : pending_h_.data();
  mtp_last_embd_.assign(h_prev, h_prev + n_embd_);
  pending_h_.assign(h_last, h_last + n_embd_);

  mirror_pos_ = batch.pos[n - 1];
  last_token_ = batch.token[n - 1];
}

void MtpSpeculator::clear() {
  if (ctx_dft_) {
    llama_memory_clear(llama_get_memory(ctx_dft_), true);
  }
  mirror_pos_ = -1;
  origin_has_logits_ = false;
  origin_row_idx_ = 0;
  pending_h_.assign((size_t)n_embd_, 0.0f);
  mtp_h_last_.assign((size_t)n_embd_, 0.0f);
  mtp_last_embd_.assign((size_t)n_embd_, 0.0f);
  valid_ = true;
  warned_ = false;
  // Visible confirmation: the invalidation notice says speculation is off,
  // and this is the path that turns it back on (/clear, re-decode fallbacks,
  // slow restore).  The mirror rebuilds via the mirror hook as the context
  // re-decodes.
  diag("MTP: mirror cleared -- speculative decoding enabled", "\033[32m");
}

void MtpSpeculator::on_mirror_loaded() {
  if (!ctx_dft_) return;
  // The mirror KV was restored from a V1 fast-restore cache: its rows are
  // exactly the saved session's rows, so it matches the restored main context
  // again.  Re-arm mirror_pos_ from the loaded memory; hidden-state
  // bookkeeping (pending_h_, origin) is deliberately NOT reconstructed -- it
  // is refreshed by the first forward feed via process()'s position guard, and
  // the one slightly-off re-mirrored row at the heal point only affects the
  // very first draft round's acceptance (the verify pass re-samples every
  // committed token from the main model, so output correctness is untouched).
  // Drafting is held off (origin_has_logits_ = false) until that first feed.
  // A few stale draft rows may sit past n_past_ (draft chains are mirrored
  // too); the same position guard trims them on the first main decode.
  mirror_pos_ = llama_memory_seq_pos_max(llama_get_memory(ctx_dft_), 0);
  origin_has_logits_ = false;
  origin_row_idx_ = 0;
  valid_ = true;
  warned_ = false;
  diag("MTP: mirror restored from cache -- speculative decoding enabled", "\033[32m");
}

void MtpSpeculator::invalidate(const std::string& reason) {
  if (!valid_) return;
  valid_ = false;
  if (!warned_) {
    warned_ = true;
    diag("MTP: " + reason + " -- speculative decoding off (normal decoding continues; /clear or a re-decode restore re-enables it)", "\033[33m");
  }
}

bool MtpSpeculator::can_draft() const {
  return valid_ && origin_has_logits_ && mirror_pos_ >= 0 &&
         (int)pending_h_.size() == n_embd_ && (int)mtp_h_last_.size() == n_embd_;
}

std::vector<llama_token> MtpSpeculator::draft() {
  std::vector<llama_token> out;
  if (!can_draft()) return out;

  llama_sampler_reset(mtp_smpl_);
  const size_t row_bytes = (size_t)n_embd_ * sizeof(float);
  std::vector<float> hrow;
  hrow.reserve((size_t)n_embd_);

  llama_token d;
  if (origin_has_logits_) {
    // First proposal: the existing mirror row (logits computed by process()).
    d = llama_sampler_sample(mtp_smpl_, ctx_dft_, origin_row_idx_);
    const float* mh = llama_get_embeddings_nextn_ith(ctx_dft_, origin_row_idx_);
    if (mh) hrow.assign(mh, mh + n_embd_);
  } else {
    // Rare fallback: the origin row has no logits (e.g. the round was aborted
    // mid-verify).  Re-decode that one mirror row (identical cell content).
    common_batch_clear(mtp_batch_);
    common_batch_add(mtp_batch_, last_token_, mirror_pos_, {0}, true);
    std::memcpy(mtp_batch_.embd, mtp_last_embd_.data(), row_bytes);
    if (llama_decode(ctx_dft_, mtp_batch_) != 0) {
      invalidate("MTP origin re-decode failed");
      return out;
    }
    d = llama_sampler_sample(mtp_smpl_, ctx_dft_, mtp_batch_.n_tokens - 1);
    const float* mh = llama_get_embeddings_nextn_ith(ctx_dft_, mtp_batch_.n_tokens - 1);
    if (mh) hrow.assign(mh, mh + n_embd_);
  }

  if (d == LLAMA_TOKEN_NULL || hrow.size() != (size_t)n_embd_) {
    invalidate("MTP draft sampling failed");
    return out;
  }
  llama_sampler_accept(mtp_smpl_, d);
  out.push_back(d);

  // Chain: each next proposal is the MTP head's prediction given the
  // previous chain row (its input is the previous chain row's MTP output).
  llama_pos pos = mirror_pos_ + 1;
  for (int j = 1; j < draft_len_; j++) {
    common_batch_clear(mtp_batch_);
    common_batch_add(mtp_batch_, d, pos, {0}, true);
    std::memcpy(mtp_batch_.embd, hrow.data(), row_bytes);
    if (llama_decode(ctx_dft_, mtp_batch_) != 0) {
      invalidate("MTP draft chain decode failed");
      return {};
    }
    llama_token d2 = llama_sampler_sample(mtp_smpl_, ctx_dft_, mtp_batch_.n_tokens - 1);
    const float* mh = llama_get_embeddings_nextn_ith(ctx_dft_, mtp_batch_.n_tokens - 1);
    if (d2 == LLAMA_TOKEN_NULL || !mh) {
      invalidate("MTP draft chain sampling failed");
      return {};
    }
    hrow.assign(mh, mh + n_embd_);
    llama_sampler_accept(mtp_smpl_, d2);
    out.push_back(d2);
    d = d2;
    pos++;
  }

  s_rounds_++;
  s_drafted_ += out.size();
  return out;
}

// --- round bookkeeping -----------------------------------------------------

void MtpSpeculator::on_origin_match()    { s_origin_match_++; }
void MtpSpeculator::on_origin_mismatch() { s_origin_mismatch_++; }
void MtpSpeculator::on_verify_match()    { s_verify_match_++; }
void MtpSpeculator::on_verify_mismatch(int /*draft_idx*/) { /* per-mismatch detail unused */ }
void MtpSpeculator::on_bonus()           { s_bonus_++; }
void MtpSpeculator::on_abort(int /*committed_verify_rows*/, int /*verify_rows*/) {
  s_aborted_++;
}

void MtpSpeculator::note_round(int committed) {
  s_committed_ += committed;
  trailing_committed_ += committed;
  trailing_rounds_++;
  if (trailing_rounds_ >= ADAPT_WINDOW) {
    const double avg = (double)trailing_committed_ / (double)ADAPT_WINDOW;
    trailing_committed_ = 0;
    trailing_rounds_ = 0;
    if (avg < ADAPT_MIN) {
      low_streak_++;
      if (low_streak_ >= ADAPT_STREAK) {
        invalidate("sustained low acceptance (mean " +
                   std::to_string((int)(avg * 100) / 100) + " tokens/round < " +
                   std::to_string((int)(ADAPT_MIN * 100) / 100) + ")");
      }
    } else {
      low_streak_ = 0;
    }
  }
}

bool MtpSpeculator::stats_line(std::string& out) {
  if (s_rounds_ == 0) return false;
  const double mean = (double)s_committed_ / (double)s_rounds_;
  const int full_pct = (int)(100.0 * (double)s_bonus_ / (double)s_rounds_);
  char buf[256];
  snprintf(buf, sizeof(buf),
           "# MTP: rounds=%llu drafted=%llu origin_ok=%llu/%llu verify_ok=%llu bonus=%llu aborted=%llu mean=%.2f full=%d%%\n",
           (unsigned long long)s_rounds_,
           (unsigned long long)s_drafted_,
           (unsigned long long)s_origin_match_,
           (unsigned long long)(s_origin_match_ + s_origin_mismatch_),
           (unsigned long long)s_verify_match_,
           (unsigned long long)s_bonus_,
           (unsigned long long)s_aborted_,
           mean, full_pct);
  out = buf;
  return true;
}

void MtpSpeculator::reset_stats() {
  s_rounds_ = 0;
  s_drafted_ = 0;
  s_origin_match_ = 0;
  s_origin_mismatch_ = 0;
  s_verify_match_ = 0;
  s_verify_mismatch_ = 0;
  s_bonus_ = 0;
  s_aborted_ = 0;
  s_committed_ = 0;
  trailing_committed_ = 0;
  trailing_rounds_ = 0;
  low_streak_ = 0;
}

void MtpSpeculator::log_summary(const std::string& tag) {
  std::string line;
  if (stats_line(line)) {
    diag("MTP " + tag + ": " + std::string(line), "\033[90m");
  }
}

