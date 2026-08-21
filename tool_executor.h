#ifndef TOOL_EXECUTOR_H
#define TOOL_EXECUTOR_H

#include "session.h"
#include "tools.h"
#include "tokens.h"
#include "token_generator.h"
#include <string>
#include <vector>
#include <functional>

class ToolExecutor {
public:
  struct Result {
    bool should_auto_continue = false;
    bool was_interrupted = false;
    // When true, the caller should feed a correction prompt and generate once.
    // The error message to include is in correction_error_msg.
    bool needs_correction = false;
  };

  static Result execute(
    SessionState& state,
    std::string& generated_text,
    const std::string& full_generated,
    size_t tool_start,
    size_t tool_end,
    bool was_mid_tool_call,
    std::function<std::vector<llama_token>(std::string)> tokenize,
    std::function<bool(const std::vector<llama_token>&)> feed_tokens,
    llama_context* ctx,
    int& n_past,
    const llama_context_params& cparams,
    int& g_auto_continue_depth,
    int max_auto_continue,
    bool allow_continue_resume
    );
};

#endif // TOOL_EXECUTOR_H

