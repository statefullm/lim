#pragma once

namespace Tokens {
    // --- Model ChatML / System Tokens ---
    constexpr const char* TURN_START = "<|im_start|>";
    constexpr const char* TURN_END   = "<|im_end|>";
    constexpr const char* HEADER_START = "<|start_header|>";
    constexpr const char* HEADER_END   = "<|end_header|>";
    constexpr const char* EOT          = "<|eot|>";

    // --- XML Tool Calling Schema ---
    constexpr const char* FUNC_START = "<function=";
    constexpr const char* FUNC_END   = "</function>";
    
    // --- XML Parameter Schema ---
    constexpr const char* PARAM_START = "<parameter=";
    constexpr const char* PARAM_END   = "</parameter>";
    
    // --- Escaped Tokens (For protecting the C++ parser) ---
    constexpr const char* PARAM_END_ESC = "<\\parameter>";
}
