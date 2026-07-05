#include "session_utils.h"
#include "common.h"
#include "token_generator.h"
#include "output.h"
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

// Forward declarations for globals in main.cc
extern bool is_debug;
extern std::ofstream token_log;
extern bool honest_speed;

std::string html_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '&') out += "&amp;";
        else if (c == '<') out += "&lt;";
        else if (c == '>') out += "&gt;";
        else if (c == '"') out += "&quot;";
        else out += c;
    }
    return out;
}

void log_tokens(const std::string& label, const std::vector<llama_token>& toks, llama_context* ctx) {
    if (!is_debug || !token_log.is_open()) return;
    for (llama_token t : toks) {
        std::string piece = common_token_to_piece(ctx, t);
        token_log << label << " " << t << " \"" << escape_token_piece(piece) << "\"\n";
    }
    token_log.flush();
}

void strip_tags(std::string& str, const std::vector<std::string>& tags) {
    for (const auto& tag : tags) {
        size_t p;
        while ((p = str.find(tag)) != string::npos) {
            str.erase(p, tag.length());
        }
    }
}

void diag_speed(int n_past, int n_ctx, int t_count, double elapsed, double decode_time) {
    if (t_count <= 0 || elapsed <= 0.0) return;
    double context_percent = (n_past / (double)n_ctx) * 100.0;

    // Pick denominator based on honest_speed global
    double denom = elapsed;  // default: wall clock ("honest")
    if (!honest_speed && decode_time > 0.0) {
        denom = decode_time;
    }

    // Send to browser status bar (compact: no labels)
    if (should_output_to_browser()) {
        std::ostringstream oss2;
        oss2 << round_int(t_count / denom) << " t/s | " << n_past << " (" << (int)context_percent << "%)";
        stream_speed(oss2.str());
    }
}

void diag_restore(const std::string& path, int token_count) {
    diag("Restoring session from " + path + "... (" + std::to_string(token_count) + " tokens)", "\033[35m");
}

