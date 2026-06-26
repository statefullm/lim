#ifndef PARSERS_H
#define PARSERS_H

#include <string>
#include <vector>

using namespace std;

// --- Recursive Tokens::PARAM_END escape/unescape ---
void escape_parameter_tags(string& str);
void unescape_parameter_tags(string& str);

// --- Model turn token escape/unescape (runtime-dependent tokens) ---
// Escapes/unescapes model-specific turn delimiters in tool output content.
// Uses the same generic function as PARAM_END: insert '\' after first char.
// Must be called after model tokens are initialized (post init_model_tokens).
void escape_turn_tags(string& str);
void unescape_turn_tags(string& str);

// Linear-time parsers - O(n) time complexity with general data robustness
// No artificial limits on string length or array size
// Handles arbitrary-length tool calls safely
string extract_string_arg_bounded(const string& tool_call, const string& arg_name);

vector<string> extract_array_arg_bounded(const string& tool_call, const string& arg_name);

int extract_int_arg_bounded(const string& tool_call, const string& arg_name);

// Remove trailing spaces from a string (used for cleaning tool arguments)
string remove_trailing_spaces(const string& str);

#endif // PARSERS_H
