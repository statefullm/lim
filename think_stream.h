#ifndef THINK_STREAM_H
#define THINK_STREAM_H

#include <cstddef>
#include <string>
#include <functional>

// Batched browser think-stream (SEG_THINK) flush policy.
//
// The think path used to call stream_think() once per token, so the
// browser received one ~4-5 B segment-prefixed chunk per token.  This
// policy buffers think text and flushes in line/size chunks instead:
// a newline tail or THINK_STREAM_MAX bytes, plus every region
// transition and the turn end (callers drive those with flush()/
// discard()).  The wire format is unchanged: each flush is still one
// prefix+content write -- the writer below is stream_think, which adds
// the SEG_THINK prefix and HTML-escapes the chunk (output.cc).
//
// Model- and I/O-independent (the writer is injected), so the policy
// unit-tests without a model.  Byte-identity: html_escape_sentinel's
// per-byte decisions are chunk-local (the only forward context is a
// run of pre-existing sentinel chars immediately after the current
// byte, which the old per-token chunking split just as often), so the
// concatenation of the flushed payloads of a turn is byte-identical to
// the old per-token streaming for any input whose sentinel runs do not
// straddle a flush boundary differently than a token boundary -- in
// practice, all model output.
class ThinkStream {
 public:
  static constexpr size_t MAX = 1024;  // flush cap (bytes of think text)

  using Writer = std::function<void(const std::string&)>;

  explicit ThinkStream(Writer writer) : writer_(std::move(writer)) {}

  // Append a think piece (already tag-stripped and, for the first
  // piece of a block, leading-whitespace-suppressed by the caller).
  // Flushes when the buffer ends in a newline or reaches MAX bytes.
  void append(const std::string& piece) {
    if (piece.empty()) return;
    buf_ += piece;
    if (buf_.back() == '\n' || buf_.size() >= MAX) flush();
  }

  // Emit the pending content as one final chunk (no-op when empty).
  void flush() {
    if (buf_.empty()) return;
    writer_(buf_);
    buf_.clear();
  }

  // Drop the pending content without emitting (canary silence: the
  // probe must produce no browser output at all).
  void discard() { buf_.clear(); }

 private:
  std::string buf_;
  Writer writer_;
};

#endif // THINK_STREAM_H

