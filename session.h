#ifndef SESSION_H
#define SESSION_H

#include "llama.h"
#include "common.h"
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <functional>
#include <chrono>
#include "loop_detector.h"

using namespace std;

// Run the main chat session loop.
// Returns true to continue, false when user types quit/exit.
bool run_chat_session(
    // LLM context and model (not owned)
    llama_context* ctx,
    const llama_vocab* vocab,
    llama_sampler* smpl,
    llama_batch& batch,
    int& n_past,
    const llama_context_params& cparams,

    // System prompt tokens
    const vector<llama_token>& system_tokens,

    // Configuration
    bool use_dummy_thought,

    // Session state (passed by reference so caller can observe changes)
    bool& auto_continue,
    bool& reincarnate_mode,
    bool& prev_was_interrupted,
    bool& first_turn_done,
    int& last_t_count,
    double& last_elapsed,
    int& last_n_past,

    // Loop prevention state
    set<string>& clean_files,
    LoopDetector& loop_guard,
    int& invalid_tool_strikes
);

#endif // SESSION_H
