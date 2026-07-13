
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
static void escape_one_token(string& str, const string& token, char esc_char) {
    if (token.size() < 2) return;
    char first = token[0];
    const string suffix = token.substr(1);

    size_t start_pos = 0;
    while (start_pos < str.length()) {
        size_t pos = str.find(first, start_pos);
        if (pos == string::npos) break;

        // Skip any esc_char after the first character.
        size_t scan = pos + 1;
        while (scan < str.length() && str[scan] == esc_char) scan++;

        // Check if suffix matches.
        bool match = true;
        for (size_t k = 0; k < suffix.length(); k++) {
            if (scan + k >= str.length() || str[scan + k] != suffix[k]) { match = false; break; }
        }

        if (match) {
            str.insert(pos + 1, 1, esc_char);
            start_pos = pos + 2 + suffix.length();
        } else {
            start_pos = pos + 1;
        }
    }
}

// Generic unescape: remove one copy of 'esc_char' after first char of every occurrence
// of 'token'.  If no extra esc_char, leave alone (raw token -- should not appear in practice).
static void unescape_one_token(string& str, const string& token, char esc_char) {
    if (token.size() < 2) return;
    char first = token[0];
    const string suffix = token.substr(1);

    size_t start_pos = 0;
    while (start_pos < str.length()) {
        size_t pos = str.find(first, start_pos);
        if (pos == string::npos) break;

        // Count esc_char after the first character.
        size_t scan = pos + 1;
        int num_bs = 0;
        while (scan < str.length() && str[scan] == esc_char) { num_bs++; scan++; }

        // Check if suffix matches after the esc_char.
        bool match = true;
        for (size_t k = 0; k < suffix.length(); k++) {
            if (scan + k >= str.length() || str[scan + k] != suffix[k]) { match = false; break; }
        }

        if (match) {
            if (num_bs > 0) {
                // Remove one esc_char: first + (num_bs-1) esc_char + suffix.
                string replacement(1, first);
                for (int b = 0; b < num_bs - 1; b++) replacement += esc_char;
                replacement += suffix;
                str.replace(pos, scan - pos + suffix.length(), replacement);
                start_pos = pos + replacement.length();
            } else {
                // No esc_char -- raw token, leave alone.
                start_pos = pos + 1 + suffix.length();
            }
        } else {
            start_pos = pos + 1;
        }
    }
}

void escape_parameter_tags(string& str)     { escape_one_token(str, PARAM_END, ESCAPE_CHAR); }
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
        if (do_escape) escape_one_token(str, t, ESCAPE_CHAR);
        else unescape_one_token(str, t, ESCAPE_CHAR);
    }
}

void escape_turn_tags(string& str) {
    _escape_turn_tokens_impl(str, true);
}

void unescape_turn_tags(string& str) {
    _escape_turn_tokens_impl(str, false);
}

// Find the next unescaped PARAM_END starting from 'from'.
// Skips over backslash-escaped forms (ESCAPE_CHAR after first char).
// Returns string::npos if no unescaped PARAM_END is found.
static size_t find_unescaped_param_end(const string& text, size_t from) {
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
static string strip_quotes(const string& s) {
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

// Strip quote characters from a string (used for forgiving tag-name matching).
static string strip_quotes_from_name(const string& s) {
    string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c != '"' && c != '\'') out += c;
    }
    return out;
}

// Extract the raw param name from a PARAM_START tag, stripping stray quotes.
// Given text starting at PARAM_START, finds the '>' and returns the trimmed,
// unquoted name between them.  Also returns 'content_begin' pointing just past '>'.
static string extract_param_name(const string& tool_call, size_t tag_pos, size_t& content_begin) {
    size_t after_prefix = tag_pos + strlen(PARAM_START);
    size_t gt = tool_call.find('>', after_prefix);
    if (gt == string::npos) return "";

    content_begin = gt + 1;
    return strip_quotes_from_name(tool_call.substr(after_prefix, gt - after_prefix));
}

// Find a parameter tag by name.  Accepts either the canonical form
//   <parameter=name>...PARAM_END
// or a bare tag form (LLM occasionally emits this):
//   <name>...PARAM_END
// Sets 'content_begin' to the position just past the opening '>'.
// Returns true if found, false otherwise.
static bool find_param_tag(const string& tool_call, const string& arg_name, size_t& content_begin) {
    // First pass: look for canonical <parameter=name> form.
    {
        size_t pos = 0;
        while ((pos = tool_call.find(PARAM_START, pos)) != string::npos) {
            size_t cb = 0;
            string found_name = extract_param_name(tool_call, pos, cb);
            if (found_name == arg_name) {
                content_begin = cb;
                return true;
            }
            pos++;
        }
    }

    // Second pass: fall back to bare <name> form.
    string bare_open = "<" + arg_name + ">";
    size_t pos = 0;
    while ((pos = tool_call.find(bare_open, pos)) != string::npos) {
        content_begin = pos + bare_open.length();
        return true;
    }

    return false;
}

// Linear-time string parser for XML schema
string extract_string_arg_bounded(const string& tool_call, const string& arg_name) {
    size_t content_begin = 0;
    if (!find_param_tag(tool_call, arg_name, content_begin)) {
        return "";
    }

    size_t end = find_unescaped_param_end(tool_call, content_begin);
    if (end == string::npos) end = tool_call.length();

    // For the path parameter, treat a newline as an implicit PARAM_END.
    // If the LLM forgets to close the path param, the newline prevents
    // bleeding into subsequent parameters.
    if (arg_name == "path") {
        size_t nl = tool_call.find('\n', content_begin);
        if (nl != string::npos && nl < end) {
            end = nl;
        }
    }

    string val = tool_call.substr(content_begin, end - content_begin);
    return strip_quotes(val);
}

// Linear-time array parser for XML schema (Newline/Comma separated)
// Handles: newline-separated items, comma-separated items, and square-bracket lists.
// Brackets can wrap a single line or span multiple lines. Delimiters are newlines
// and commas. Leading/trailing whitespace is trimmed from each item.
vector<string> extract_array_arg_bounded(const string& tool_call, const string& arg_name) {
    vector<string> result;

    size_t content_begin = 0;
    if (!find_param_tag(tool_call, arg_name, content_begin)) {
        return result;
    }

    size_t end = find_unescaped_param_end(tool_call, content_begin);
    if (end == string::npos) end = tool_call.length();

    string val = tool_call.substr(content_begin, end - content_begin);

    // If the llm bleeds its schema (</<), truncate the parameter exactly at the bleed.
    size_t bleed_pos = val.find("</<");
    if (bleed_pos != string::npos) {
        val = val.substr(0, bleed_pos);
    }

    // --- Strip outer square brackets if present (handles multi-line bracket lists) ---
    size_t first_bracket = val.find('[');
    size_t last_bracket  = val.rfind(']');
    if (first_bracket != string::npos && last_bracket != string::npos && last_bracket > first_bracket) {
        val = val.substr(first_bracket + 1, last_bracket - first_bracket - 1);
    }

    // Split the parameter block by newlines. Each non-empty line is a candidate item.
    stringstream ss(val);
    string line;
    while (getline(ss, line)) {
        stringstream comma_ss(line);
        string item;
        while (getline(comma_ss, item, ',')) {
            size_t first = item.find_first_not_of(" \t\r\n");
            if (first == string::npos) continue;
            size_t last = item.find_last_not_of(" \t\r\n");
            result.push_back(strip_quotes(item.substr(first, last - first + 1)));
        }
    }
    return result;
}

