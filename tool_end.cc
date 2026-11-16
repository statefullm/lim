#include "tool_end.h"
#include "parsers.h"
#include "tokens.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

using namespace std;
using namespace Tokens;

// Longest marker the main scan searches for (PARAM_END); the frontier-clamp
// tail window is this minus one.  The longest bare tag (<content>) is
// shorter than PARAM_END.
static constexpr size_t TOOL_END_MAX_MARK = strlen(PARAM_END);
// The incomplete FUNC_END the fallback searches for: FUNC_END minus its
// closing '>'.
static constexpr size_t TOOL_END_PARTIAL_LEN = strlen(FUNC_END) - 1;

// Find tool call end robustly, handling malformed closing tags, nested FUNC_START in content,
// and FUNC_START/FUNC_END embedded within parameter values (ignored via in_parameter guard).
// A parameter region opens with PARAM_START or the bare <name> alias
// (PARAM_NAMES); PARAM_END is the only exit.
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
    // The bare <name> alias opens a parameter region too.  Only searched
    // outside a parameter: values may legitimately contain bare tags.
    size_t bare_len = 0;
    size_t next_bare = in_parameter ? string::npos : find_bare_param_open(text, pos, bare_len);

    // Determine the earliest event
    size_t events[5] = {next_start, next_end, next_ps, next_pe, next_bare};
    int type = -1; // 0=start, 1=end, 2=param_start, 3=param_end, 4=bare_param_start
    for (int i = 0; i < 5; i++) {
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
    if (type == 4) {
      // Entered a parameter region via the bare <name> alias.
      in_parameter = true;
      pos = next_bare + bare_len;
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
    size_t bare_len = 0;
    size_t pb = in_parameter ? string::npos : find_bare_param_open(text, pos, bare_len);
    if (ps != string::npos && (pe == string::npos || ps < pe) && (pp == string::npos || ps < pp) && (pb == string::npos || ps < pb)) {
      in_parameter = true;
      pos = ps + ps_str.length();
      continue;
    }
    if (pe != string::npos && (pp == string::npos || pe < pp) && (pb == string::npos || pe < pb)) {
      in_parameter = false;
      pos = pe + pe_str.length();
      continue;
    }
    if (pb != string::npos && (pp == string::npos || pb < pp)) {
      // Entered a parameter region via the bare <name> alias.
      in_parameter = true;
      pos = pb + bare_len;
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

bool ToolEndScanner::scan(const string& text) {
  if (tool_end_ != string::npos) return true;
  const size_t len = text.length();

  // Leading FUNC_START skip: the reference re-checks
  // text.compare(from_pos_, FUNC_START) on every call; the text only grows
  // at the tail, so the check is stable once the bytes are decided.  While
  // it is still undetermined (the tail matches a proper FUNC_START prefix)
  // no complete marker can start at or after from_pos_ (a marker's only
  // '<' is its first byte), so the scans below are no-ops until it
  // resolves.
  if (!skip_done_) {
    const size_t fs_len = strlen(FUNC_START);
    const size_t avail = (len > from_pos_) ? min(fs_len, len - from_pos_) : 0;
    if (avail > 0) {
      if (text.compare(from_pos_, avail, FUNC_START, 0, avail) != 0) {
        skip_done_ = true;  // mismatch: no skip
      } else if (avail == fs_len) {
        pos_ = from_pos_ + fs_len;  // skip applied
        skip_done_ = true;
      }
    }
  }

  // --- Main scan: complete marker events, left to right ---
  size_t scan = pos_;
  while (scan < len) {
    const size_t next_start = text.find(FUNC_START, scan);
    const size_t next_end   = text.find(FUNC_END, scan);
    const size_t next_ps    = text.find(PARAM_START, scan);
    const size_t next_pe    = in_param_ ? text.find(PARAM_END, scan) : string::npos;
    // The bare <name> alias opens a parameter region too; only searched
    // outside a parameter (values may legitimately contain bare tags).
    size_t bare_len = 0;
    const size_t next_bare = in_param_ ? string::npos
                                       : find_bare_param_open(text, scan, bare_len);

    // Earliest event (same order/tie-break as the reference).
    size_t events[5] = {next_start, next_end, next_ps, next_pe, next_bare};
    int type = -1;  // 0=start 1=end 2=param_start 3=param_end 4=bare_param_start
    for (int i = 0; i < 5; i++) {
      if (events[i] == string::npos) continue;
      if (type == -1 || events[i] < events[type]) type = i;
    }
    if (type == -1) break;

    // A complete event supersedes any pending partial-FUNC_END fallback:
    // the reference runs the fallback only after its main walk exhausts.
    fallback_active_ = false;

    if (type == 2) {
      in_param_ = true;
      scan = next_ps + strlen(PARAM_START);
      continue;
    }
    if (type == 3) {
      in_param_ = false;
      scan = next_pe + strlen(PARAM_END);
      continue;
    }
    if (type == 4) {
      in_param_ = true;
      scan = next_bare + bare_len;
      continue;
    }
    if (in_param_) {
      // FUNC_START or FUNC_END inside a parameter value -- ignore.
      scan = events[type] + (type == 0 ? strlen(FUNC_START) : strlen(FUNC_END));
      continue;
    }
    if (type == 0) {
      depth_++;
      scan = next_start + strlen(FUNC_START);
    } else {
      depth_--;
      if (depth_ == 0) {
        tool_end_ = next_end;
        return true;
      }
      scan = next_end + strlen(FUNC_END);
    }
  }

  // --- Main scan exhausted at 'scan': settle the frontier ---
  // A marker that is not yet complete must start inside the last
  // (TOOL_END_MAX_MARK - 1) characters, and no marker can start inside
  // another marker's byte span ('<' occurs only in the first byte), so the
  // walk above could not have run past a marker that can still complete:
  // clamping the frontier to the tail window never skips one, and each
  // call's scans stay bounded by O(tail + appended).
  const size_t tail = (len > TOOL_END_MAX_MARK - 1) ? len - (TOOL_END_MAX_MARK - 1) : 0;
  pos_ = max(scan, tail);

  // --- Partial-FUNC_END fallback (reference: only when no complete events
  // remain, from the exhaustion position) ---
  if (in_param_) {
    // Frozen: no PARAM_START/PARAM_END/bare open exists at or beyond the
    // exhaustion, so the in-parameter state can never clear while the main
    // scan stays exhausted and the reference's fallback walk always ends in
    // npos here.
    fallback_active_ = false;
    return false;
  }
  if (!fallback_active_) {
    fallback_active_ = true;
    // Never ahead of the exhaustion position (the reference starts there,
    // so a partial before it must not be seen) and never behind a live
    // partial (the invariant that kept it findable is maintained below).
    fallback_pos_ = max(fallback_pos_, scan);
  }
  const string partial(FUNC_END, TOOL_END_PARTIAL_LEN);
  const size_t pp = text.find(partial, fallback_pos_);
  if (pp == string::npos) {
    // No partial yet: a live one (not yet complete) must start inside the
    // last (TOOL_END_PARTIAL_LEN - 1) characters.
    const size_t tail_p = (len > TOOL_END_PARTIAL_LEN - 1) ? len - (TOOL_END_PARTIAL_LEN - 1) : 0;
    fallback_pos_ = max(fallback_pos_, tail_p);
    return false;
  }
  // The reference's return rules (the garbage-'>' search it performs there
  // has no effect on the returned position).
  if (pp + TOOL_END_PARTIAL_LEN < len && text[pp + TOOL_END_PARTIAL_LEN] == '>') {
    tool_end_ = pp;  // the '>' completes a full FUNC_END
  } else {
    // Garbage before the '>', no '>' at all, or the partial at the end of
    // the text: accept at the partial's end; the caller repairs.
    tool_end_ = pp + TOOL_END_PARTIAL_LEN;
  }
  fallback_active_ = false;
  return true;
}
