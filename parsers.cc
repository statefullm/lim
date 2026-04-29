#include "parsers.h"
#include "tokens.h"
#include <cctype>
#include <cstdlib>
#include <sstream> // Required for stringstream

using namespace std;
using namespace Tokens;

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
    size_t end = tool_call.find(PARAM_END, start);
    if (end == string::npos) end = tool_call.length();

    string val = tool_call.substr(start, end - start);

    // Strip only the structural newline the LLM injects immediately after/before tags.
    // Do NOT trim horizontal whitespace (spaces/tabs) -- those are significant for
    // exact-match editing (edit_file old/new parameters must match file content exactly).
    if (!val.empty() && val.front() == '\n') val.erase(val.begin());
    if (!val.empty() && val.back() == '\n') val.pop_back();
    // Also strip a trailing \r if present (Windows line endings)
    if (!val.empty() && val.back() == '\r') val.pop_back();

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
    size_t end = tool_call.find(PARAM_END, start);
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
    size_t end = tool_call.find(PARAM_END, start);
    if (end == string::npos) end = tool_call.length();

    string val = tool_call.substr(start, end - start);

    size_t bleed_pos = val.find("</<");
    if (bleed_pos != string::npos) {
        val = val.substr(0, bleed_pos);
    }

    size_t first = val.find_first_not_of(" \t\n\r");
    if (first == string::npos) return 0;

    return std::atoi(val.c_str() + first);
}

string remove_trailing_spaces(const string& str) {
    size_t end = str.find_last_not_of(" \t\n\r");
    return (end == string::npos) ? "" : str.substr(0, end + 1);
}
