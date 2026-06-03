#ifndef MODEL_H
#define MODEL_H

#include "llama.h"
#include <string>

using namespace std;

// --- Model Detection and Chat Template Selection ---
enum class ModelType {
    UNKNOWN,
    CHATML,      // Standard ChatML format (Nemotron, Mistral, etc.)
    LLAMA3       // Llama 3 format
};

ModelType detect_model_type(const llama_vocab * vocab);
string get_chat_template(ModelType model_type);

// --- Decode Error Handling ---
bool handle_llama_decode_error(llama_context *ctx, llama_batch batch, const char* error_msg = "KV Cache Exhausted. Type 'clear' to reset.", bool should_break = true);

// Sync n_past with the actual KV cache position after a decode.
void sync_n_past(llama_context *ctx, int &n_past);

// --- Log Callbacks ---
extern bool first_prompt_displayed;
// is_debug is declared extern in filesystem.h and defined in main.cc

void dummy_log_callback(enum ggml_log_level level, const char * text, void * user_data);
void custom_log_callback(enum ggml_log_level level, const char * text, void * user_data);

#endif // MODEL_H
