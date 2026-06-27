#ifndef SESSION_H
#define SESSION_H

#include "llama.h"
#include "common.h"
#include <string>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <functional>
#include <chrono>
#include "loop_detector.h"
#include "filesystem.h"
using namespace std;

// Forward declaration for INITIAL_CWD from main.cc
extern string INITIAL_CWD;

struct SessionState {
    bool auto_continue = false;
    bool reincarnate_mode = false;
    bool prev_was_interrupted = false;
    bool first_turn_done = false;
    int last_t_count = 0;
    double last_elapsed = 0.0;
    double last_decode_time = 0.0;  // Sum of per-token decode intervals (for benchmark-style speed)
    int last_n_past = 0;
    map<string, string> file_cache;  // path -> content hash (for cache validation)
    LoopDetector loop_guard;
    int invalid_tool_strikes = 0;
    // Internal state (was static inside the function)
    int auto_continue_depth_val = 0;
    bool tool_interrupt_pending = false;
    string partial_tool_text;
    // All tokens fed into context, for save/restore
    vector<llama_token> all_context_tokens;
    // Token positions and prompt text at each prompt return, for partial restore
    vector<PromptCheckpoint> prompt_checkpoints;
    // Log file index (set by main.cc), so save files match chat log numbering
    int log_index = 0;
};

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

    // Session state (consolidated into a single struct)
    SessionState& state
);

#endif // SESSION_H
