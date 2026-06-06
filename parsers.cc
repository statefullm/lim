#include "parsers.h"
#include "tokens.h"
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream> // Required for stringstream

using namespace std;
using namespace Tokens;

// The suffix to match after '<' and optional backslashes, derived from PARAM_END.
static const string param_end_suffix = string(PARAM_END + 1);

void escape_parameter_tags(string& str) {
    size_t start_pos = 0;
    while (start_pos < str.length()) {
        size_t lt_pos = str.find('<', start_pos);
        if (lt_pos == string::npos) break;

        size_t scan = lt_pos + 1;
        int num_backslashes = 0;
        while (scan < str.length() && str[scan] == '\\') {
            num_backslashes++;
            scan++;
        }

        bool match = true;
        for (size_t k = 0; k < param_end_suffix.length(); k++) {
            if (scan + k >= str.length() || str[scan + k] != param_end_suffix[k]) {
                match = false;
                break;
            }
        }

        if (match) {
            size_t token_start = lt_pos;
            size_t token_end = scan + param_end_suffix.length();

            // Build replacement: PARAM_END with num_backslashes extra '\' inserted after '<'
            string replacement("<");
            for (int b = 0; b <= num_backslashes; b++) replacement += '\\';
            replacement += param_end_suffix;

            str.replace(token_start, token_end - token_start, replacement);
            start_pos = token_start + replacement.length();
        } else {
            start_pos = lt_pos + 1;
        }
    }
}

void unescape_parameter_tags(string& str) {
    size_t start_pos = 0;
    while (start_pos < str.length()) {
        size_t lt_pos = str.find('<', start_pos);
        if (lt_pos == string::npos) break;

        size_t scan = lt_pos + 1;
        int num_backslashes = 0;
        while (scan < str.length() && str[scan] == '\\') {
            num_backslashes++;
            scan++;
        }

        bool match = true;
        for (size_t k = 0; k < param_end_suffix.length(); k++) {
            if (scan + k >= str.length() || str[scan + k] != param_end_suffix[k]) {
                match = false;
                break;
            }
        }

        if (match) {
            size_t token_start = lt_pos;
            size_t token_end = scan + param_end_suffix.length();

            // Build replacement: PARAM_END with (num_backslashes-1) '\' after '<', min 0
            int new_bs = (num_backslashes > 0) ? (num_backslashes - 1) : 0;
            string replacement("<");
            for (int b = 0; b < new_bs; b++) replacement += '\\';
            replacement += param_end_suffix;

            str.replace(token_start, token_end - token_start, replacement);
            start_pos = token_start + replacement.length();
        } else {
            start_pos = lt_pos + 1;
        }
    }
}

// Find the next unescaped PARAM_END starting from 'from'.
// Skips over backslash-escaped forms (PARAM_END_ESC).
// Returns string::npos if no unescaped PARAM_END is found.
static size_t find_unescaped_param_end(const string& text, size_t from) {
    size_t param_len = strlen(PARAM_END);
    while (from + param_len <= text.length()) {
        size_t pos = text.find(PARAM_END, from);
        if (pos == string::npos) return string::npos;

        // Check if there's a backslash immediately before the '<'
        if (pos > 0 && text[pos - 1] == '\\') {
            // Count consecutive backslashes before '<'
            size_t bs_start = pos - 1;
            while (bs_start > 0 && text[bs_start - 1] == '\\') {
                bs_start--;
            }
            int num_bs = pos - bs_start;
            if (num_bs % 2 == 1) {
                // Odd number of backslashes => escaped, skip past this match
                from = pos + param_len;
                continue;
            }
        }
        return pos; // Unescaped PARAM_END found
    }
    return string::npos;
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
