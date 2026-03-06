#include "parsers.h"
#include <cctype>
#include <algorithm>
#include <cstdlib>

using namespace std;

/*
 * TIME COMPLEXITY ANALYSIS:
 *
 * ALL FUNCTIONS ARE TRULY O(n) LINEAR TIME:
 *
 * 1. extract_string_arg_bounded():
 *    - Single find() call: O(n)
 *    - Single pass parsing loop: O(n)
 *    - Fixed-length substr() calls: O(1) since length=3 constant
 *    - Total: O(n) linear
 *
 * 2. extract_array_arg_bounded():
 *    - Single find() call: O(n)
 *    - Single pass parsing loop: O(n)
 *    - Fixed-length substr() calls: O(1)
 *    - Total: O(n) linear
 *
 * 3. extract_int_arg_bounded():
 *    - Single find() call: O(n)
 *    - Single pass to find digits: O(n)
 *    - Total: O(n) linear
 *
 * KEY DIFFERENCE FROM OLD IMPLEMENTATION:
 * The old llm.cc parsers used variable-length substring comparisons
 * in the parsing loop, making them O(n*m) where m could approach n,
 * resulting in effectively quadratic behavior for large inputs.
 *
 * This implementation uses FIXED-LENGTH substr() calls (always 3 chars)
 * which compile to O(1) operations, ensuring true linear time complexity.
 */

// Linear-time string parser - O(n) time complexity with general data robustness
// Handles arbitrary-length strings without artificial limits
string extract_string_arg_bounded(const string& tool_call, const string& arg_name) {

  // Linear search for argument name
  string search_key = arg_name + "=";
  size_t pos = tool_call.find(search_key);
  if (pos == string::npos) return "";

  // Find opening quote after equals sign
  size_t start = pos + search_key.length();
  while (start < tool_call.length() && isspace(tool_call[start])) start++;
  if (start >= tool_call.length()) return "";

  // Detect quote type: single, double, or triple quotes
  char quote_char = '\0';
  size_t quote_len = 1; // Default to single quote

  // Check for triple quotes first
  if (tool_call.substr(start, 3) == "\"\"\"") {
    quote_char = '"';
    quote_len = 3;
    start += 3;
  } else if (tool_call.substr(start, 3) == "'''") {
    quote_char = '\'';
    quote_len = 3;
    start += 3;
  } else if (tool_call[start] == '"' || tool_call[start] == '\'') {
    quote_char = tool_call[start];
    start += 1;
  } else if (tool_call[start] == '`') {
    quote_char = '`';
    start += 1;
  } else {
    return ""; // No quote found
  }

  // Single-pass parsing with escape sequence handling
  string result;
  bool escaped = false;

  for (size_t i = start; i < tool_call.length(); ++i) {
    char c = tool_call[i];

    if (escaped) {
      // Handle escape sequences
      if (c == 'n') result += '\n';
      else if (c == 't') result += '\t';
      else if (c == '\\') result += '\\';
      else if (c == '"' || c == '\'') result += c;
      else if (c == '`') result += c;
      else result += c;
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else if (c == quote_char) {
      // Check for triple quote closing
      if (quote_len == 3 && i + 2 < tool_call.length() &&
          tool_call[i+1] == quote_char && tool_call[i+2] == quote_char) {
        return result; // Found triple quote closing
      } else if (quote_len == 1) {
        return result; // Found single quote closing
      }
    } else {
      // Add regular character to result
      result += c;
    }
  }

  return result; // Return parsed content (unclosed string)
}

// Linear-time array parser - O(n) time complexity with general data robustness
// Handles arbitrary-length arrays and strings without artificial limits
vector<string> extract_array_arg_bounded(const string& tool_call, const string& arg_name) {
  vector<string> result;

  // Linear search for argument name
  string search_key = arg_name + "=";
  size_t pos = tool_call.find(search_key);
  if (pos == string::npos) return result;

  // Find opening bracket
  size_t start = tool_call.find('[', pos);
  if (start == string::npos) return result;

  // Single-pass parsing with proper quote handling
  string current_str;
  char quote_char = '\0';
  bool in_string = false;
  bool escaped = false;
  int bracket_depth = 0;

  for (size_t i = start; i < tool_call.length(); ++i) {
    char c = tool_call[i];

    if (escaped) {
      // Handle escape sequences in strings
      if (c == 'n') current_str += '\n';
      else if (c == 't') current_str += '\t';
      else if (c == '\\') current_str += '\\';
      else if (c == '"' || c == '\'') current_str += c;
      else current_str += c;
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else if (in_string) {
      // Check for string termination with proper boundary validation
      bool match = true;
      size_t quote_len_local = 1; // Default to single quote

      // Determine quote length
      if (quote_char == '"' && i + 2 < tool_call.length() &&
          tool_call[i+1] == '"' && tool_call[i+2] == '"') {
        quote_len_local = 3;
      } else if (quote_char == '\'' && i + 2 < tool_call.length() &&
                 tool_call[i+1] == '\'' && tool_call[i+2] == '\'') {
        quote_len_local = 3;
      } else if (quote_char == '`' && i + 2 < tool_call.length() &&
                 tool_call[i+1] == '`' && tool_call[i+2] == '`') {
        quote_len_local = 3;
      }

      for (size_t j = 0; j < quote_len_local; ++j) {
        if (i + j >= tool_call.length() || tool_call[i + j] != quote_char) {
          match = false; break;
        }
      }

      if (match) {
        // Validate end boundary: must be followed by comma, bracket, whitespace, or backtick
        bool valid_end = false;
        size_t peek = i + quote_len_local;
        while (peek < tool_call.length() && isspace(tool_call[peek])) peek++;
        if (peek >= tool_call.length() || tool_call[peek] == ',' || tool_call[peek] == ']' || tool_call[peek] == '`') {
            valid_end = true;
        }

        if (valid_end) {
            result.push_back(current_str);
            current_str.clear();
            in_string = false;
            quote_char = '\0';
            i += quote_len_local - 1; // Skip remaining quotes
        } else {
          current_str += c;
        }
      } else {
        current_str += c;
      }
    } else {
      // Detect quote type for array elements
      if (tool_call.substr(i, 3) == "\"\"\"") {
        in_string = true;
        quote_char = '"';
        i += 2; // Skip remaining quotes (loop will increment i again)
      } else if (tool_call.substr(i, 3) == "'''") {
        in_string = true;
        quote_char = '\'';
        i += 2; // Skip remaining quotes (loop will increment i again)
      } else if (c == '"' || c == '\'') {
        in_string = true;
        quote_char = c;
      } else if (c == '`') {
        in_string = true;
        quote_char = c;
      } else if (c == '[') {
        bracket_depth++;
      } else if (c == ']') {
        bracket_depth--;
        if (bracket_depth == 0) return result; // Array complete
      }
    }
  }

  return result; // Return partial array if unclosed
}

// Linear-time integer parser - O(n) time complexity
// Note: This implementation is truly linear - single pass through the string
int extract_int_arg_bounded(const string& tool_call, const string& arg_name) {

  string search_key = arg_name + "=";
  size_t pos = tool_call.find(search_key);
  if (pos == string::npos) return 0;

  size_t start = pos + search_key.length();
  while (start < tool_call.length() && isspace(tool_call[start])) start++;

  // Skip leading quotes if present
  if (start < tool_call.length() && (tool_call[start] == '"' || tool_call[start] == '\'')) {
    start++;
  }

  size_t end = start;
  while (end < tool_call.length() && tool_call[end] >= '0' && tool_call[end] <= '9') end++;

  if (end > start) {
    try { return stoi(tool_call.substr(start, end - start)); }
    catch (...) {}
  }
  return 0;
}
