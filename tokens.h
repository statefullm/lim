#pragma once

namespace Tokens {
// --- XML Tool Calling Schema ---
constexpr const char* FUNC_START = "<function=";
constexpr const char* FUNC_END   = "</function>";

// --- XML Parameter Schema ---
constexpr const char* PARAM_START = "<parameter=";
constexpr const char* PARAM_END   = "</parameter>";

// --- Escaped Tokens (For protecting the C++ parser) ---
// Escape contract: prepend ESCAPE_CHAR before first character of token.
// PARAM_END = "</parameter>"  -> escaped: "\</parameter>"
// Escaped form computed at runtime by escape_parameter_tags().
// Not needed as a compile-time constant.

constexpr const char* DOUBLE_OPEN = "</</";

// --- Escape Characters ---
// Generic backslash escape for PARAM_END and turn tokens in parsers.cc.
// Inserted after the first character of a token to produce its escaped form.
constexpr char ESCAPE_CHAR        = '\\';

// Sentinel character used in HTML sentinel tokens like lt, gt, mth_N
// to represent special characters (<, >) and math markers that must not be
// interpreted as HTML tags by the browser.
constexpr char HTML_SENTINEL_CHAR = '%';

// Backslash escape specifically for HTML sentinel tokens in output.cc and
// viewer.html.  Inserted after the first character of an HTML sentinel to
// produce its literal form rather than being converted to "<" or ">").
constexpr char HTML_ESCAPE_CHAR   = '\\';

} // namespace Tokens
