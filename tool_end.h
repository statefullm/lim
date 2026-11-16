#ifndef TOOL_END_H
#define TOOL_END_H

#include <cstddef>
#include <string>

// Robustly find a tool call's end: the start position of the closing
// FUNC_END tag, or string::npos while none is (robustly) complete.  Handles
// malformed closing tags, nested FUNC_START in content, and
// FUNC_START/FUNC_END embedded within parameter values (ignored via the
// in_parameter guard).  A parameter region opens with PARAM_START or the
// bare <name> alias (PARAM_NAMES); PARAM_END is the only exit.
//
// The full re-scan: O(text) per call.  For the per-token hot path use
// ToolEndScanner, the incremental counterpart that processes only the
// appended tail and is byte-for-byte equivalent at every prefix.
size_t find_tool_end_robust(const std::string& text, size_t from_pos, bool* out_in_parameter = nullptr);

// Repair malformed closing tag at position returned by find_tool_end_robust.
void repair_malformed_tool_end(std::string& text, size_t pos);

// Incremental counterpart of find_tool_end_robust.  The text only ever grows
// at the tail, and each scan() processes only the appended tail, so total
// work across calls is O(total text) instead of O(text) per call.
//
// At every prefix, (tool_end(), in_parameter()) equals what
// find_tool_end_robust returns when re-scanned over that prefix (verified by
// the differential test test_tool_end.cc).
//
// The walk is a deterministic state machine over the marker event sequence
// (FUNC_START/FUNC_END depth, in-parameter flag): generated_text_ only grows
// at the tail, so the walk state over the consumed prefix never changes, and
// only marker events in the newly appended tail need processing.  Two
// sub-problems, handled as in the sibling incremental scanners in
// token_generator.cc:
//   1. Markers split across a token boundary: after each call the scan
//      frontier is clamped to a tail window (longest marker minus one), so a
//      partially present marker is still found once it completes.  A marker
//      that is not yet complete must start inside that window, and no marker
//      can start inside another marker's byte span ('<' occurs only in the
//      first byte), so the clamped frontier never runs past a marker that
//      can still complete.
//   2. The partial-FUNC_END fallback runs only while the main scan is
//      exhausted (no complete marker events remain) -- exactly when the
//      reference runs it -- and only needs to inspect the last
//      len(FUNC_END) - 1 characters with the current in-parameter state.
//      While the main scan is exhausted inside a parameter the fallback can
//      never fire (no PARAM_START/PARAM_END/bare open remains, so the
//      in-parameter state is frozen and the reference's fallback walk always
//      ends in npos).
class ToolEndScanner {
public:
  // from_pos: position of the call's own FUNC_START (normal call) or 0 when
  // resuming a call interrupted mid-tool-call (the first tokens may complete
  // a leading FUNC_START, which is then skipped exactly like in the full
  // re-scan).  Default-construct + begin() works too: a scanner may be
  // started later, when the call's FUNC_START becomes known.
  ToolEndScanner() {}
  explicit ToolEndScanner(size_t from_pos) { begin(from_pos); }
  void begin(size_t from_pos) {
    from_pos_ = from_pos;
    pos_ = from_pos;  // the reference never searches before from_pos
    depth_ = 1;
    in_param_ = false;
    skip_done_ = false;
    tool_end_ = std::string::npos;
    fallback_active_ = false;
    fallback_pos_ = 0;
  }

  // Advance the scan.  'text' must be an extension of the text passed to the
  // previous call (tail appends only).  Returns false while the end is
  // undetermined (the full re-scan would return npos).
  bool scan(const std::string& text);

  // End position as find_tool_end_robust reports it: start of the closing
  // tag, possibly incomplete (the caller runs repair_malformed_tool_end).
  // string::npos while undetermined.
  size_t tool_end() const { return tool_end_; }
  // Inside a parameter region?  The reference's out_in_parameter at the same
  // prefix.
  bool in_parameter() const { return in_param_; }

private:
  size_t from_pos_;
  size_t pos_ = 0;             // main-scan frontier
  int depth_ = 1;              // FUNC_START nesting depth (1 = the call itself)
  bool in_param_ = false;      // inside a parameter region
  bool skip_done_ = false;     // leading-FUNC_START skip resolved or applied
  size_t tool_end_ = std::string::npos;
  // Partial-FUNC_END fallback (see class comment): active while the main
  // scan is exhausted with in_param_ false and no partial found yet.
  bool fallback_active_ = false;
  size_t fallback_pos_ = 0;
};

#endif // TOOL_END_H
