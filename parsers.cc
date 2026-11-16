#include "parsers.h"
#include "tokens.h"
#include "model.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream> // Required for stringstream

using namespace std;
using namespace Tokens;
extern bool is_debug;

// Generic escape: insert one copy of 'esc_char' after first char of every occurrence
// of 'token'.  Handles recursive escaping: if already escaped, adds another esc_char.
// Single pass into a reserved output: each byte is copied once, so the cost is
// O(n + occurrences * token length) instead of an O(n) memmove per occurrence.
// (Resuming past the esc run on a mismatch -- rather than one char -- is
// output-identical: esc chars that are also 'first' chars can only form
// non-matching candidates inside that same run, and the run's own suffix
// check decides for all of them at once.  Likewise, on a match, resuming
// past the token instead of the esc-th byte of the written suffix is
// output-identical for the tokens escaped here: the first char never
// completes a second match from inside the token body.)
void escape_tag(string& str, const string& token, char esc_char) {
  if (token.size() < 2) return;
  const char first = token[0];
  const string suffix = token.substr(1);

  string out;
  out.reserve(str.size() + 32);
  size_t i = 0;
  while (i < str.size()) {
    if (str[i] != first) {
      out += str[i];
      i++;
      continue;
    }
    // The esc_char run after the first character, then the token body.
    size_t j = i + 1;
    size_t esc = 0;
    while (j < str.size() && str[j] == esc_char) { esc++; j++; }
    const bool match = (j + suffix.size() <= str.size()) &&
                       str.compare(j, suffix.size(), suffix) == 0;
    out += first;
    out.append(esc, esc_char);
    if (match) {
      out += esc_char;  // the inserted escape joins the existing run
      out += suffix;
      i = j + suffix.size();
    } else {
      i = j;
    }
  }
  str = std::move(out);
}

// Generic unescape: remove one copy of 'esc_char' after first char of every occurrence
// of 'token'.  If no extra esc_char, leave alone (raw token -- should not appear in
// practice).  Single pass, the mirror of escape_tag.
void unescape_one_token(string& str, const string& token, char esc_char) {
  if (token.size() < 2) return;
  const char first = token[0];
  const string suffix = token.substr(1);

  string out;
  out.reserve(str.size());
  size_t i = 0;
  while (i < str.size()) {
    if (str[i] != first) {
      out += str[i];
      i++;
      continue;
    }
    // The esc_char run after the first character, then the token body.
    size_t j = i + 1;
    size_t esc = 0;
    while (j < str.size() && str[j] == esc_char) { esc++; j++; }
    const bool match = (j + suffix.size() <= str.size()) &&
                       str.compare(j, suffix.size(), suffix) == 0;
    if (match) {
      // Drop one esc_char: first + (esc-1) esc_char + suffix (or the raw
      // token untouched when esc == 0).
      out += first;
      if (esc > 1) out.append(esc - 1, esc_char);
      out += suffix;
      i = j + suffix.size();
    } else {
      out += first;
      out.append(esc, esc_char);
      i = j;
    }
  }
  str = std::move(out);
}

void escape_parameter_tags(string& str)     { escape_tag(str, PARAM_END, ESCAPE_CHAR); }
void unescape_parameter_tags(string& str)   { unescape_one_token(str, PARAM_END, ESCAPE_CHAR); }

// Extract base turn tokens from model turn markers.
void collect_base_turn_tokens(vector<string>& out) {
  vector<string> markers;
  if (!g_model_tokens.user_turn_start.text.empty()) markers.push_back(g_model_tokens.user_turn_start.text);
  if (!g_model_tokens.assistant_turn_start.text.empty()) markers.push_back(g_model_tokens.assistant_turn_start.text);
  if (!g_model_tokens.system_turn_start.text.empty()) markers.push_back(g_model_tokens.system_turn_start.text);
  if (!g_model_tokens.turn_end.text.empty()) markers.push_back(g_model_tokens.turn_end.text);

  for (auto &m : markers) {
    size_t pos = 0;
    while ((pos = m.find("<|", pos)) != string::npos) {
      size_t end = m.find("|>", pos + 2);
      if (end != string::npos) {
        string token = m.substr(pos, end - pos + 2);
        if (find(out.begin(), out.end(), token) == out.end())
          out.push_back(token);
        pos = end + 2;
      } else break;
    }
    if (m.find("<|") == string::npos && !m.empty()) {
      if (find(out.begin(), out.end(), m) == out.end())
        out.push_back(m);
    }
  }
}

static void _escape_turn_tokens_impl(string& str, bool do_escape) {
  vector<string> tokens;
  collect_base_turn_tokens(tokens);
  for (auto &t : tokens) {
    if (do_escape) escape_tag(str, t, ESCAPE_CHAR);
    else unescape_one_token(str, t, ESCAPE_CHAR);
  }
}

void escape_turn_tags(string& str) {
  _escape_turn_tokens_impl(str, true);
}

void unescape_turn_tags(string& str) {
  _escape_turn_tokens_impl(str, false);
}

// Find the next PARAM_END starting from 'from'.
// Returns string::npos if no PARAM_END is found.
static size_t find_param_end(const string& text, size_t from) {
  // The escaped form has an ESCAPE_CHAR after the first character, so str.find(PARAM_END)
  // naturally skips escaped occurrences. Just find the raw token.
  return text.find(PARAM_END, from);
}

// Strip surrounding matching quotes (single or double) from a string.
// Handles cases where whitespace surrounds the quoted content:
//   "main.cc"        -> main.cc
//    "main.cc"       -> main.cc  (whitespace before quote)
//   "main.cc"        -> main.cc  (whitespace after quote)
//   int main() {     -> int main() {  (no quotes, returned unchanged)
string strip_quotes(const string& s) {
  // Find first and last non-whitespace characters
  size_t first = s.find_first_not_of(" \t\r\n");
  size_t last = s.find_last_not_of(" \t\r\n");

  if (first == string::npos || last < first) return "";  // All whitespace or empty

  char fc = s[first];
  char lc = s[last];

  // If matching quotes surround the trimmed content, strip them and return inner text
  if ((fc == '"' && lc == '"') || (fc == '\'' && lc == '\'')) {
    return s.substr(first + 1, last - first - 1);
  }

  // No surrounding quotes - return original string unchanged to preserve
  // exact whitespace for edit_file old/new parameters
  return s;
}

// Strip quote characters and surrounding whitespace from a string (used for
// forgiving tag-name matching: PARAM_START paths> -> paths).  Names never
// contain whitespace, so trimming is safe.  Values are never run through
// here -- exact value whitespace (edit_file old/new) is preserved by the
// value extractors.  Quotes are stripped before trimming so quoted names
// with inner padding (" paths ") also normalize.
string strip_quotes_from_name(const string& s) {
  string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c != '"' && c != '\'') out += c;
  }
  size_t first = out.find_first_not_of(" \t\r\n");
  if (first == string::npos) return "";
  size_t last = out.find_last_not_of(" \t\r\n");
  return out.substr(first, last - first + 1);
}

// Trim leading/trailing characters that appear in 'chars'.
string trim_chars(const string& s, const string& chars) {
  size_t first = s.find_first_not_of(chars);
  if (first == string::npos) return "";
  size_t last = s.find_last_not_of(chars);
  return s.substr(first, last - first + 1);
}

// Extract the 'path' parameter value, trimming leading/trailing whitespace
// (space, tab, CR, LF).  Paths never contain newlines, so edge newlines are
// just the model placing the value on its own line.  An INTERNAL newline
// means a missing PARAM_END closing tag (the value bled into following
// parameter text), and param_has_newline flags that as a malformed call
// before the handler acts on the path.
string extract_path_arg(const string& tool_call) {
  return trim_chars(extract_string_arg_bounded(tool_call, "path"), " \t\r\n");
}

// True when the cleaned form (quotes removed, whitespace-trimmed at the
// edges -- strip_quotes_from_name's rule) of text[from, gt) equals
// arg_name.  Exits at the first divergence, so a mismatch costs at most the
// argument name's length (plus quotes interleaved in that span), never the
// region's length.
static bool param_name_matches(const string& text, size_t from, size_t gt, const string& arg_name) {
  auto ws_or_quote = [](char c) {
    return c == '"' || c == '\'' || c == ' ' || c == '\t' || c == '\r' || c == '\n';
  };
  size_t i = from;
  while (i < gt && ws_or_quote(text[i])) i++;  // leading edge
  for (size_t j = 0; j < arg_name.size(); j++) {
    // Quotes are removable anywhere; any other non-matching char (including
    // inner whitespace, which the cleaned form keeps) fails.
    while (i < gt && (text[i] == '"' || text[i] == '\'')) i++;
    if (i >= gt || text[i] != arg_name[j]) return false;
    i++;
  }
  while (i < gt && ws_or_quote(text[i])) i++;  // trailing edge
  return i == gt;
}

// Find a parameter tag by name.  Accepts either the canonical form
//   PARAM_STARTname>...PARAM_END
// or a bare tag form (LLM occasionally emits this):
//   <name>...PARAM_END
// Sets 'content_begin' to the position just past the opening '>'.
// Returns true if found, false otherwise.
//
// O(n) per form: PARAM_START occurrences are found in order (the marker
// cannot overlap itself -- '<' occurs only in its first byte -- so after a
// mismatched occurrence the next one can start at the earliest right after
// the current marker), and each occurrence's closing '>' comes from a
// monotonically advancing cursor (the first '>' after a later search point
// can never be before an earlier one), so every byte is scanned at most
// once and the name test early-exits on the first divergence.
static bool find_param_tag(const string& tool_call, const string& arg_name, size_t& content_begin) {
  // First pass: look for canonical PARAM_STARTname> form.
  {
    const size_t ps_len = strlen(PARAM_START);
    size_t pos = 0;
    size_t gt_cursor = string::npos;  // first '>' found so far (or none yet)
    bool gt_exhausted = false;        // no '>' remains: all later names are empty
    while ((pos = tool_call.find(PARAM_START, pos)) != string::npos) {
      const size_t from = pos + ps_len;
      size_t gt;
      if (gt_exhausted) {
        gt = string::npos;
      } else if (gt_cursor != string::npos && gt_cursor >= from) {
        // No '>' in [from, gt_cursor) (it was the first after an earlier
        // point): gt_cursor is still this occurrence's first '>'.
        gt = gt_cursor;
      } else {
        gt = tool_call.find('>', from);
        if (gt == string::npos) gt_exhausted = true;
        gt_cursor = gt;
      }
      if (gt != string::npos && param_name_matches(tool_call, from, gt, arg_name)) {
        content_begin = gt + 1;
        return true;
      }
      pos += ps_len;
    }
  }

  // Second pass: fall back to the bare <name> form (first occurrence).
  const string bare_open = "<" + arg_name + ">";
  const size_t pos = tool_call.find(bare_open);
  if (pos != string::npos) {
    content_begin = pos + bare_open.length();
    return true;
  }

  return false;
}

// Presence check shared by the validation gate (tools.cc): true if the
// parameter is present in either accepted form (canonical or bare tag),
// so a call the extractors accept is not rejected by the gate.
bool param_present(const string& tool_call, const string& arg_name) {
  size_t content_begin = 0;
  return find_param_tag(tool_call, arg_name, content_begin);
}

// Earliest bare parameter opening tag (<name> for any name in PARAM_NAMES)
// at or after 'from'; sets 'tag_len' to the length of the tag found.
size_t find_bare_param_open(const string& text, size_t from, size_t& tag_len) {
  static const vector<string> tags = [] {
    vector<string> v;
    v.reserve(sizeof(PARAM_NAMES) / sizeof(PARAM_NAMES[0]));
    for (const char* name : PARAM_NAMES) v.push_back(string("<") + name + ">");
    return v;
  }();
  tag_len = 0;
  size_t best = string::npos;
  for (const string& tag : tags) {
    size_t pos = text.find(tag, from);
    if (pos != string::npos && (best == string::npos || pos < best)) {
      best = pos;
      tag_len = tag.length();
    }
  }
  return best;
}

// Extract the raw parameter block: text between the opening tag
// (canonical PARAM_START name> form or the bare-tag fallback) and the
// closing PARAM_END, or through the end of the tool call if no PARAM_END
// is present.  Returns "" if the parameter tag is not found.
static string extract_param_block(const string& tool_call, const string& arg_name) {
  size_t content_begin = 0;
  if (!find_param_tag(tool_call, arg_name, content_begin)) {
    return "";
  }
  size_t end = find_param_end(tool_call, content_begin);
  if (end == string::npos) end = tool_call.length();
  return tool_call.substr(content_begin, end - content_begin);
}

// Raw variant of extract_string_arg_bounded: returns the parameter block
// without quote stripping.  Used for value-level validation (e.g. detecting
// newlines in the "paths" array value, which the array parser's splitting
// would otherwise hide).
string extract_raw_arg_bounded(const string& tool_call, const string& arg_name) {
  return extract_param_block(tool_call, arg_name);
}

// Linear-time string parser for XML schema
string extract_string_arg_bounded(const string& tool_call, const string& arg_name) {
  // Note: newlines are intentionally NOT truncated here (old/new, content,
  // command, and query values must keep exact whitespace).  For "path",
  // leading/trailing newlines are stripped by extract_path_arg (paths never
  // contain newlines); an INTERNAL newline in a path means the model forgot
  // the closing PARAM_END tag -- the value extends to the next parameter's
  // PARAM_END -- and the handler routes it through the correction cycle
  // (param_has_newline checks) before executing anything.
  return strip_quotes(extract_param_block(tool_call, arg_name));
}

// Linear-time array parser for XML schema (Comma separated).
// The read_files newline gate rejects values with internal newlines before
// this runs, so the value is a single line (possibly with leading/trailing
// whitespace or newlines), optionally wrapped in square brackets.  Items are
// comma-separated; each is trimmed and stripped of surrounding quotes.
vector<string> extract_array_arg_bounded(const string& tool_call, const string& arg_name) {
  vector<string> result;

  string val = extract_param_block(tool_call, arg_name);
  if (val.empty()) return result;

  // If the llm bleeds its schema (</<), truncate the parameter exactly at the bleed.
  size_t bleed_pos = val.find("</<");
  if (bleed_pos != string::npos) {
    val = val.substr(0, bleed_pos);
  }

  // --- Strip outer square brackets if present ---
  size_t first_bracket = val.find('[');
  size_t last_bracket  = val.rfind(']');
  if (first_bracket != string::npos && last_bracket != string::npos && last_bracket > first_bracket) {
    val = val.substr(first_bracket + 1, last_bracket - first_bracket - 1);
  }

  // Split by comma.  Each non-empty item is trimmed of whitespace (including
  // any newlines at the edges) and stripped of surrounding quotes.
  stringstream comma_ss(val);
  string item;
  while (getline(comma_ss, item, ',')) {
    size_t first = item.find_first_not_of(" \t\r\n");
    if (first == string::npos) continue;
    size_t last = item.find_last_not_of(" \t\r\n");
    result.push_back(strip_quotes(item.substr(first, last - first + 1)));
  }
  return result;
}
