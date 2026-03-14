#include "parsers.h"
#include "tokens.h"
#include <cctype>
#include <cstdlib>

using namespace std;

// Linear-time string parser for XML schema
string extract_string_arg_bounded(const string& tool_call, const string& arg_name) {
    string search_key = string(Tokens::PARAM_START) + arg_name + ">";
    size_t pos = tool_call.find(search_key);
    if (pos == string::npos) return "";

    size_t start = pos + search_key.length();
    size_t end = tool_call.find(Tokens::PARAM_END, start);
    if (end == string::npos) end = tool_call.length();

    string val = tool_call.substr(start, end - start);

    // Trim leading and trailing whitespace/newlines injected by the model
    size_t first = val.find_first_not_of(" \t\n\r");
    if (first == string::npos) return "";
    size_t last = val.find_last_not_of(" \t\n\r");

    return val.substr(first, last - first + 1);
}

// Linear-time array parser for XML schema
vector<string> extract_array_arg_bounded(const string& tool_call, const string& arg_name) {
    vector<string> result;
    string search_key = string(Tokens::PARAM_START) + arg_name + ">";
    size_t pos = tool_call.find(search_key);
    if (pos == string::npos) return result;

    size_t start = pos + search_key.length();
    size_t end = tool_call.find(Tokens::PARAM_END, start);
    if (end == string::npos) end = tool_call.length();

    string val = tool_call.substr(start, end - start);

    // Find the array brackets inside the parameter block
    size_t arr_start = val.find('[');
    if (arr_start == string::npos) return result;

    bool in_string = false;
    char quote_char = '\0';
    string current_item = "";

    // Parse standard JSON-style array items
    for (size_t i = arr_start + 1; i < val.length(); ++i) {
        char c = val[i];
        if (in_string) {
            if (c == '\\' && i + 1 < val.length()) {
                char next = val[i+1];
                if (next == 'n') current_item += '\n';
                else if (next == 't') current_item += '\t';
                else current_item += next;
                i++;
            } else if (c == quote_char) {
                in_string = false;
                result.push_back(current_item);
                current_item = "";
            } else {
                current_item += c;
            }
        } else {
            if (c == '"' || c == '\'') {
                in_string = true;
                quote_char = c;
            } else if (c == ']') {
                break;
            }
        }
    }
    return result;
}

// Linear-time integer parser for XML schema
int extract_int_arg_bounded(const string& tool_call, const string& arg_name) {
    string search_key = string(Tokens::PARAM_START) + arg_name + ">";
    size_t pos = tool_call.find(search_key);
    if (pos == string::npos) return 0;

    size_t start = pos + search_key.length();
    size_t end = tool_call.find(Tokens::PARAM_END, start);
    if (end == string::npos) end = tool_call.length();

    string val = tool_call.substr(start, end - start);

    size_t first = val.find_first_not_of(" \t\n\r");
    if (first == string::npos) return 0;

    return std::atoi(val.c_str() + first);
}

string remove_trailing_spaces(const string& str) {
    size_t end = str.find_last_not_of(" \t\n\r");
    return (end == string::npos) ? "" : str.substr(0, end + 1);
}
