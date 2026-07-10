#pragma once

namespace Tokens {
// --- XML Parameter Schema ---
constexpr const char* THINK_START = "<think>";
constexpr const char* THINK_END   = "</think>";

// --- XML Tool Calling Schema ---
constexpr const char* FUNC_START = "<function=";
constexpr const char* FUNC_END   = "</function>";

// --- XML Parameter Schema ---
constexpr const char* PARAM_START = "<parameter=";
constexpr const char* PARAM_END   = "</parameter>";

// --- Escaped Tokens (For protecting the C++ parser) ---
// Escape contract: prepend '\' before first character of token.
// PARAM_END = "</parameter>"  -> escaped: "\</parameter>"
// Escaped form computed at runtime by escape_parameter_tags().
// Not needed as a compile-time constant.

constexpr const char* DOUBLE_OPEN = "</</";
}
