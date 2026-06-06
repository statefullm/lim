#include "model.h"
#include "filesystem.h"
#include "tokens.h"
#include <iostream>
#include <string>

using namespace std;
using namespace Tokens;

// --- Model Detection and Chat Template Selection ---

// Function to get the appropriate chat template for a model type
string get_chat_template(ModelType model_type) {
    switch (model_type) {
        case ModelType::CHATML:
            return "chatml";
        case ModelType::LLAMA3:
            return "llama3";
        default:
            return "chatml";
    }
}

// Function to detect model type based on vocabulary tokens
ModelType detect_model_type(const llama_vocab * vocab) {
    int32_t n_tokens = llama_vocab_n_tokens(vocab);

    bool has_im_start = false;
    bool has_im_end = false;
    bool has_reasoning_start = false;
    bool has_reasoning_end = false;

    for (llama_token i = 0; i < n_tokens; i++) {
        const char* token_text = llama_vocab_get_text(vocab, i);
        if (token_text == nullptr) continue;

        string text(token_text);

        if (text.find(Tokens::TURN_START) != string::npos) has_im_start = true;
        if (text.find(Tokens::TURN_END) != string::npos) has_im_end = true;
        if (text.find(THINK_START) != string::npos) has_reasoning_start = true;
        if (text.find(THINK_END) != string::npos) has_reasoning_end = true;
    }

    if (has_reasoning_start && has_reasoning_end) return ModelType::CHATML;
    if (has_im_start && has_im_end) return ModelType::CHATML;

    return ModelType::CHATML;
}

// Forward declaration for diagnostic helper defined in main.cc
extern void diag(const string& msg, const char* color);

bool handle_llama_decode_error(llama_context *ctx, llama_batch batch, const char* error_msg, bool should_break) {
    int ret = llama_decode(ctx, batch);
    if (ret < -1) {
        diag(error_msg, "\033[31m");
        return false;
    } else if (ret == -1) {
        diag("Invalid input batch: " + std::string(error_msg), "\033[31m");
        return false;
    } else if (ret == 1 || ret == 2) {
        diag(ret == 1 ? error_msg : "Aborted", "\033[31m");
        if (should_break) return false;
        return true;
    }
    return true;
}

// Sync n_past with the actual KV cache position after a decode.
// Per llama.cpp API: on errors/aborts, partially-decoded ubatches may remain
// in the cache, and on ret==1 the cache is fully restored.  Blindly incrementing
// n_past before llama_decode() therefore drifts out of sync.  This helper
// queries the real cache position and corrects n_past accordingly.
void sync_n_past(llama_context *ctx, int &n_past) {
    llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (max_pos >= 0) {
        n_past = (int)(max_pos + 1);  // next position to write
    }
}

// --- Log Callbacks ---
// is_debug is defined in main.cc

void dummy_log_callback(enum ggml_log_level level, const char * text, void * user_data) {}

void custom_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    cerr << text;
}
