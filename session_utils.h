#ifndef SESSION_UTILS_H
#define SESSION_UTILS_H

#include "llama.h"
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

// HTML escape special characters
std::string html_escape(const std::string& s);

// Log a batch of tokens with a label (no-op if debug disabled or token_log not open)
void log_tokens(const std::string& label, const std::vector<llama_token>& toks, llama_context* ctx);

// Strip all occurrences of given tags from a string
void strip_tags(std::string& str, const std::vector<std::string>& tags);

// Print speed/context diagnostic to stdout (used during auto-continue chains)
// When gen_wall_time > 0 and honest_speed is false, uses wall-clock generation time
// (first-token decode start to last-token decode end), matching llama-cli's "Generation t/s".
void diag_speed(int n_past, int n_ctx, int t_count, double elapsed, double decode_time = 0.0);

// Print a restore diagnostic message
void diag_restore(const std::string& path, int token_count, bool from_cache);

inline int round_int(double d) { return (int)(d + 0.5); }

#endif // SESSION_UTILS_H

