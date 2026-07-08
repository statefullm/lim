#include "session_utils.h"
#include "common.h"
#include "token_generator.h"
#include "output.h"
#include "tokens.h"
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

// Forward declarations
extern bool is_debug;
extern std::ofstream token_log;
extern std::ofstream tps_log;
extern bool honest_speed;

vector<llama_token> build_user_assistant_turn(llama_context* ctx, const string& input);

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

    double speed = t_count / denom;
    int speed_rounded = round_int(speed);

    // Write to TPS log file
    tps_log << n_past << " " << std::fixed << std::setprecision(3) << speed << "\n";

    // Send to browser status bar (compact: no labels)
    if (should_output_to_browser()) {
        std::ostringstream oss2;
        oss2 << speed_rounded << " t/s | " << n_past << " (" << (int)context_percent << "%)";
        stream_speed(oss2.str());
    }
}

void diag_restore(const std::string& path, int token_count) {
    diag("Restoring session from " + path + "... (" + std::to_string(token_count) + " tokens)", "\033[35m");
}

// Check git HEAD against saved SHA. If mismatched, warns the user and
// injects a message into restored_tokens (and decodes it via batch/n_past).
// Returns true if there was a mismatch.
bool check_git_head_on_restore(const std::string& save_path, const std::string& saved_sha,
                                llama_context* ctx, llama_batch& batch, int& n_past,
                                std::vector<llama_token>& restored_tokens) {
    if (saved_sha.empty()) return false;

    FILE* pipe = popen("git rev-parse HEAD 2>/dev/null", "r");
    std::string current_sha;
    if (pipe) {
        char buf[48];
        if (fgets(buf, sizeof(buf), pipe)) {
            current_sha = buf;
            while (!current_sha.empty() && (current_sha.back() == '\n' || current_sha.back() == '\r')) current_sha.pop_back();
        }
        pclose(pipe);
    }

    if (current_sha.empty() || saved_sha == current_sha) return false;

    std::string short_saved = saved_sha.substr(0, 7);
    std::string short_current = current_sha.substr(0, 7);
    diag("Git HEAD mismatch: session was at " + short_saved + ", currently at " + short_current, "\033[33m");

    // Inform the LLM about code changes.
    auto git_msg = build_user_assistant_turn(ctx,
        "Note: Git HEAD has changed since this session was saved (was " + short_saved + ", now " + short_current + "). Code or configuration may have been modified.");
    batch.n_tokens = 0;
    for (size_t i = 0; i < git_msg.size(); i++) {
        common_batch_add(batch, git_msg[i], n_past++, {0}, (i == git_msg.size() - 1));
    }
    restored_tokens.insert(restored_tokens.end(), git_msg.begin(), git_msg.end());
    return true;
}

void diag_session_restored(int session_num, size_t n_tokens, int n_ctx, const std::string& git_short) {
    string msg = "Session #" + std::to_string(session_num) + " restored: " + std::to_string(n_tokens) + " tokens loaded " + context_pct(n_tokens, n_ctx);
    if (!git_short.empty()) msg += " (git: " + git_short + ")";
    diag(msg, "\033[32m");
}

