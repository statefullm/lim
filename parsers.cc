
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

// Generic escape: insert one '\' after first char of every occurrence of 'token'.
// Handles recursive escaping: if already escaped, adds another '\'.
static void escape_one_token(string& str, const string& token) {
    if (token.size() < 2) return;
    char first = token[0];
    const string suffix = token.substr(1);

    size_t start_pos = 0;
    while (start_pos < str.length()) {
        size_t pos = str.find(first, start_pos);
        if (pos == string::npos) break;

        // Skip any backslashes after the first character.
        size_t scan = pos + 1;
        while (scan < str.length() && str[scan] == '\\') scan++;

        // Check if suffix matches.
        bool match = true;
        for (size_t k = 0; k < suffix.length(); k++) {
            if (scan + k >= str.length() || str[scan + k] != suffix[k]) { match = false; break; }
        }

        if (match) {
            str.insert(pos + 1, 1, '\\');
            start_pos = pos + 2 + suffix.length();
        } else {
            start_pos = pos + 1;
        }
    }
}

// Generic unescape: remove one '\' after first char of every occurrence of 'token'.
// If no extra backslash, leave alone (raw token -- should not appear in practice).
static void unescape_one_token(string& str, const string& token) {
    if (token.size() < 2) return;
    char first = token[0];
    const string suffix = token.substr(1);

    size_t start_pos = 0;
    while (start_pos < str.length()) {
        size_t pos = str.find(first, start_pos);
        if (pos == string::npos) break;

        // Count backslashes after the first character.
        size_t scan = pos + 1;
        int num_bs = 0;
        while (scan < str.length() && str[scan] == '\\') { num_bs++; scan++; }

        // Check if suffix matches after the backslashes.
        bool match = true;
        for (size_t k = 0; k < suffix.length(); k++) {
            if (scan + k >= str.length() || str[scan + k] != suffix[k]) { match = false; break; }
        }

        if (match) {
            if (num_bs > 0) {
                // Remove one backslash: first + (num_bs-1) backslashes + suffix.
                string replacement(1, first);
                for (int b = 0; b < num_bs - 1; b++) replacement += '\\';
                replacement += suffix;
                str.replace(pos, scan - pos + suffix.length(), replacement);
                start_pos = pos + replacement.length();
            } else {
                // No backslashes -- raw token, leave alone.
                start_pos = pos + 1 + suffix.length();
            }
        } else {
            start_pos = pos + 1;
        }
    }
}

void escape_parameter_tags(string& str)     { escape_one_token(str, PARAM_END); }
void unescape_parameter_tags(string& str)   { unescape_one_token(str, PARAM_END); }

// Extract base turn tokens from model turn markers.
static void collect_base_turn_tokens(vector<string>& out) {
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
        if (do_escape) escape_one_token(str, t);
        else unescape_one_token(str, t);
    }
}

void escape_turn_tags(string& str) {
    _escape_turn_tokens_impl(str, true);
}

void unescape_turn_tags(string& str) {
    _escape_turn_tokens_impl(str, false);
}

// Find the next unescaped PARAM_END starting from 'from'.
// Skips over backslash-escaped forms (backslash after first char).
// Returns string::npos if no unescaped PARAM_END is found.
static size_t find_unescaped_param_end(const string& text, size_t from) {
    // The escaped form has a '\' after the first character, so str.find(PARAM_END)
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

// Linear-time string parser for XML schema
string extract_string_arg_bounded(const string& tool_call, const string& arg_name) {
    string search_key = string(PARAM_START) + arg_name + ">";
    size_t pos = tool_call.find(search_key);
    if (pos == string::npos) return "";

    size_t start = pos + search_key.length();
    size_t end = find_unescaped_param_end(tool_call, start);
    if (end == string::npos) end = tool_call.length();

    string val = tool_call.substr(start, end - start);

    // With the single-line tool protocol, parameter values are taken verbatim.
    // No structural newlines exist between tags, so we preserve all content exactly.

    return strip_quotes(val);
}

// Linear-time array parser for XML schema (Newline/Comma separated)
// Handles: newline-separated items, comma-separated items, and square-bracket lists.
// Brackets can wrap a single line or span multiple lines. Delimiters are newlines
// and commas. Leading/trailing whitespace is trimmed from each item.
vector<string> extract_array_arg_bounded(const string& tool_call, const string& arg_name) {
    vector<string> result;
    string search_key = string(PARAM_START) + arg_name + ">";
    size_t pos = tool_call.find(search_key);
    if (pos == string::npos) return result;

    size_t start = pos + search_key.length();
    size_t end = find_unescaped_param_end(tool_call, start);
    if (end == string::npos) end = tool_call.length();

    string val = tool_call.substr(start, end - start);

    // If the llm bleeds its schema (</<), truncate the parameter exactly at the bleed.
    size_t bleed_pos = val.find("</<");
    if (bleed_pos != string::npos) {
        val = val.substr(0, bleed_pos);
    }

    // --- Strip outer square brackets if present (handles multi-line bracket lists) ---
    // Find the first '[' and last ']' in the entire block. If they form a valid pair
    // wrapping content, strip them so we can split by newline/commas uniformly.
    size_t first_bracket = val.find('[');
    size_t last_bracket  = val.rfind(']');
    if (first_bracket != string::npos && last_bracket != string::npos && last_bracket > first_bracket) {
        // Strip content before the first '[' and after the last ']'
        val = val.substr(first_bracket + 1, last_bracket - first_bracket - 1);
    }

    // Split the parameter block by newlines. Each non-empty line is a candidate item.
    stringstream ss(val);
    string line;
    while (getline(ss, line)) {
        // Also split by commas just in case the LLM outputs: file1.cc, file2.cc
        stringstream comma_ss(line);
        string item;
        while (getline(comma_ss, item, ',')) {
            // Trim leading and trailing whitespace/newlines
            size_t first = item.find_first_not_of(" \t\r\n");
            if (first == string::npos) continue;
            size_t last = item.find_last_not_of(" \t\r\n");
            result.push_back(strip_quotes(item.substr(first, last - first + 1)));
        }
    }
    return result;
}

// Linear-time integer parser for XML schema
int extract_int_arg_bounded(const string& tool_call, const string& arg_name) {
    string search_key = string(PARAM_START) + arg_name + ">";
    size_t pos = tool_call.find(search_key);
    if (pos == string::npos) return 0;

    size_t start = pos + search_key.length();
    size_t end = find_unescaped_param_end(tool_call, start);
    if (end == string::npos) end = tool_call.length();

    string val = tool_call.substr(start, end - start);

    size_t bleed_pos = val.find("</<");
    if (bleed_pos != string::npos) {
        val = val.substr(0, bleed_pos);
    }

    size_t first = val.find_first_not_of(" \t\n\r");
    if (first == string::npos) return 0;

    char* endptr = nullptr;
    long result = strtol(val.c_str() + first, &endptr, 10);
    if (*endptr != '\0' && *endptr != '\n' && *endptr != '\r' && *endptr != ' ' && *endptr != '\t') {
        return 0;  // Non-numeric trailing content
    }
    return static_cast<int>(result);
}

string remove_trailing_spaces(const string& str) {
    size_t end = str.find_last_not_of(" \t\n\r");
    return (end == string::npos) ? "" : str.substr(0, end + 1);
}

