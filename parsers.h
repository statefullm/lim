#ifndef PARSERS_H
#define PARSERS_H

#include <string>
#include <vector>

using namespace std;

// Linear-time parsers - O(n) time complexity with general data robustness
// No artificial limits on string length or array size
// Handles arbitrary-length tool calls safely
string extract_string_arg_bounded(const string& tool_call, const string& arg_name);

vector<string> extract_array_arg_bounded(const string& tool_call, const string& arg_name);

int extract_int_arg_bounded(const string& tool_call, const string& arg_name);

// Remove trailing spaces from a string (used for cleaning tool arguments)
string remove_trailing_spaces(const string& str);

#endif // PARSERS_H
