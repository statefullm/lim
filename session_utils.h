#ifndef SESSION_UTILS_H
#define SESSION_UTILS_H

#include "llama.h"
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

// Default context window size (KV-cache token capacity).
// Override at runtime with LIM_CTX.
static constexpr int LIM_DEFAULT_CTX = 262144;

// HTML escape special characters
std::string html_escape(const std::string& s);

// HTML-escape text for browser display, converting literal newlines to <br>
std::string html_escape_for_browser(const std::string& s);

// Log a batch of tokens with a label (no-op if debug disabled or token_log not open)
void log_tokens(const std::string& label, const std::vector<llama_token>& toks, llama_context* ctx);

// Open the chat/token/TPS log files for the given session index.
// Returns false if the chat log could not be opened.
bool open_session_logs(int idx);

// Strip all occurrences of given tags from a string
void strip_tags(std::string& str, const std::vector<std::string>& tags);

// Format the compact speed + context string shown in the browser status bar:
// "123 t/s | 45678 (51%)". Shared by the turn-end diagnostic and the
// mid-generation status update.
inline std::string format_speed_ctx(int speed_tps, int n_past, int n_ctx) {
  int pct = (n_ctx > 0) ? (int)((n_past / (double)n_ctx) * 100.0) : 0;
  return std::to_string(speed_tps) + " t/s | " + std::to_string(n_past) + " (" + std::to_string(pct) + "%)";
}

// Speed/context diagnostic. Always writes the TPS log line and (when browser
// output is on) updates the browser status bar. With to_stdout, also prints
// the bracketed line to stdout and the chat log -- used for the once-per-turn
// diagnostic at the >>> prompt; mid-chain / mid-generation updates pass false.
// The denominator is the sample+sync window (matching llama-cli's
// "Generation: X t/s") unless honest_speed, in which case it is the full
// wall-clock elapsed time.
void diag_speed(int n_past, int n_ctx, int t_count, double elapsed, double decode_time = 0.0, bool to_stdout = false);

// Print a restore diagnostic message
void diag_restore(const std::string& path, int token_count);

// Strip an optional trailing " --checkpoints" flag from a /load-style
// argument (shared by CLI restore validation in main.cc and /load parsing
// in session.cc so the two can never drift apart).
// Returns true if the flag was present and removed.
bool strip_checkpoints_flag(std::string& arg);

// Announce a new session number to the terminal and chat log
// (used at startup, /clear, and /reincarnate).
void announce_new_session(int session_num);

// Get the current git HEAD SHA (40 hex chars), or "" if not in a repo / git missing.
std::string get_git_head_sha();

// Check git HEAD against saved SHA; inject warning if mismatched.
// The note is decoded into the KV cache and appended to restored_tokens so
// both stay consistent (skipped if it does not fit in the remaining context).
// n_batch bounds the decode chunks. Returns true if there was a mismatch.
bool check_git_head_on_restore(const std::string& save_path, const std::string& saved_sha,
                               llama_context* ctx, llama_batch& batch, int& n_past,
                               std::vector<llama_token>& restored_tokens, int n_batch);

// Print a session-restored diagnostic (with optional git SHA)
void diag_session_restored(int session_num, size_t n_tokens, int n_ctx, const std::string& git_short = "");

// Format the context percentage string: "(49%)"
inline std::string context_pct(size_t n_tokens, int n_ctx) {
  if (n_ctx <= 0) return "";
  int pct = (int)((n_tokens * 100) / n_ctx);
  return "(" + std::to_string(pct) + "%)";
}

inline int round_int(double d) { return (int)(d + 0.5); }

#endif // SESSION_UTILS_H

