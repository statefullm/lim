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
    constexpr const char* PARAM_END_ESC = "<\\/parameter>";

    constexpr const char* DOUBLE_OPEN = "</</";
}
