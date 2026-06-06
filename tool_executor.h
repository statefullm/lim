#ifndef TOOL_EXECUTOR_H
#define TOOL_EXECUTOR_H

#include "session.h"
#include "tools.h"
#include "tokens.h"
#include "loop_detector.h"
#include "token_generator.h"
#include <string>
#include <vector>
#include <functional>

using namespace std;
using namespace Tokens;

class ToolExecutor {
public:
    struct Result {
        bool should_auto_continue = false;
        bool was_interrupted = false;
        bool context_exhausted = false;
    };

    static Result execute(
        SessionState& state,
        string& generated_text,
        const string& full_generated,
        size_t tool_start,
        size_t tool_end,
        bool was_mid_tool_call,
        function<vector<llama_token>(string)> tokenize,
        function<bool(const vector<llama_token>&)> feed_tokens,
        llama_context* ctx,
        int& n_past,
        const llama_context_params& cparams,
        int& g_auto_continue_depth,
        int max_auto_continue,
        bool allow_continue_resume
    );
};

#endif // TOOL_EXECUTOR_H

