#include "session_utils.h"
#include "common.h"
#include "token_generator.h"
#include <fstream>
#include <string>
#include <vector>

using namespace std;

// Forward declarations for globals in main.cc
extern bool is_debug;
extern std::ofstream token_log;

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

