#ifndef PARSERS_H
#define PARSERS_H

#include <string>
#include <vector>

// --- Recursive Tokens::PARAM_END escape/unescape ---
void escape_parameter_tags(std::string& str);
void unescape_parameter_tags(std::string& str);

// --- Model turn token escape/unescape (runtime-dependent tokens) ---
// Escapes/unescapes model-specific turn delimiters in tool output content.
// Uses the same generic function as PARAM_END: insert '\' after first char.
// Must be called after model tokens are initialized (post init_model_tokens).
void escape_turn_tags(std::string& str);
void unescape_turn_tags(std::string& str);

// Linear-time parsers - O(n) time complexity with general data robustness
// No artificial limits on string length or array size
// Handles arbitrary-length tool calls safely
std::string extract_string_arg_bounded(const std::string& tool_call, const std::string& arg_name);

std::vector<std::string> extract_array_arg_bounded(const std::string& tool_call, const std::string& arg_name);

// Collect base turn tokens from model turn markers.
// Extracts all individual <|...|> special tokens present in the model's
// turn delimiters (user_start, assistant_start, system_start, turn_end).
void collect_base_turn_tokens(std::vector<std::string>& out);

#endif // PARSERS_H
