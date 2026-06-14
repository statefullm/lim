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
void diag_speed(int n_past, int n_ctx, int t_count, double elapsed);

#endif // SESSION_UTILS_H

