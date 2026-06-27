#ifndef MODEL_H
#define MODEL_H

#include "llama.h"
#include <string>
#include <vector>

using namespace std;

// --- Model Detection and Chat Template Selection ---

// Mirror of llama.cpp's internal llm_chat_template enum (from llama/src/llama-chat.h).
// We only need the families that lim supports for incremental KV-cache appending.
enum class ModelType {
    UNKNOWN,
    CHATML,       // Standard ChatML (Nemotron, Mistral, etc.)
    LLAMA3,       // Llama 3 / 3.1 / 3.2 / 3.3
    PHI_3,        // Phi-3 mini / medium
    PHI_4,        // Phi-4
    MISTRAL_V7,   // Mistral Instruct v7 (system via [SYSTEM_PROMPT])
    MISTRAL_V1,   // Mistral Instruct v1 ([INST] ... [/INST])
    MISTRAL_V3,   // Mistral Instruct v3
    GEMMA,        // Gemma / Gemma-2 IT
    ZEPHYR,       // Zephyr
    VICUNA,       // Vicuna 1.1 / 1.5
    DEEPSEEK_3,   // DeepSeek v3 / v3.1
    GRANITE,      // Granite 3.x
    LLAMA2,       // Llama 2 ([INST] ... [/INST])
};

// A template fragment: human-readable text + pre-computed token IDs.
// Text is used for display/logging; tokens are fed into the KV cache.
struct ModelFragment {
    string text;
    vector<llama_token> tokens;
};

// Pre-computed model-specific turn delimiters, populated at startup by
// asking llama.cpp (llama_chat_apply_template) what the correct tokens are.
struct ModelTokens {
    ModelType type = ModelType::UNKNOWN;

    // Turn boundaries for each role (include role label + separator).
    // E.g. for ChatML: "user\n" / "\n"
    //      for Llama3: "<|start_header_id|>user<|end_header_id|>\n\n" / "<|eot_id|>"
    ModelFragment system_turn_start;
    ModelFragment user_turn_start;
    ModelFragment assistant_turn_start;

    // Turn end / EOT marker (may be empty for templates that rely on EOS).
    ModelFragment turn_end;

    // Whether turn_end tokens are meaningful (empty = template uses EOS only).
    bool has_explicit_turn_end() const { return !turn_end.tokens.empty(); }
};

// Global model tokens, initialized after model load.
extern ModelTokens g_model_tokens;

// Detect model type from GGUF metadata by asking llama.cpp for its Jinja
// chat template, then running llama.cpp's own llm_chat_detect_template().
ModelType detect_model_type(const llama_model *model);

// Populate g_model_tokens by:
//   1. Getting the built-in template name via llama.cpp detection
//   2. Applying it to minimal test messages via llama_chat_apply_template()
//   3. Tokenizing the resulting delimiter strings through llama.cpp
// Must be called after model and context are loaded.
void init_model_tokens(llama_context *ctx, const llama_model *model);

string get_chat_template_name(ModelType model_type);

// Generate the complete reserved-token escape contract for the system prompt.
// Covers PARAM_END (always active) plus model-specific turn delimiters.
// Returns a string with explicit token list, rules, and examples.
string generate_turn_escape_contract();

// --- Convenience helpers for message construction (token-vector based) ---

// Build system prompt tokens: BOS + system_turn_start + content + turn_end
vector<llama_token> build_system_prompt_tokens(llama_context *ctx, const string &content);

// Build user turn + assistant prefill as token vector:
//   user_turn_start + content + turn_end + "\n" + assistant_turn_start
vector<llama_token> build_user_assistant_turn(llama_context *ctx, const string &user_content);

// Build user turn only (no assistant prefill):
//   user_turn_start + content + turn_end + "\n"
vector<llama_token> build_user_turn_only(llama_context *ctx, const string &user_content);

// Build a tool result injection:
//   user_turn_start + "[Tool Result]\n" + content + turn_end + "\n" + assistant_turn_start
vector<llama_token> build_tool_result_turn(llama_context *ctx, const string &tool_output);

// Build forced-close tokens for EOG recovery / loop detection:
//   "\n" + turn_end + "\n"  (FUNC_END is added separately since it's lim protocol)
vector<llama_token> build_forced_close_tokens(llama_context *ctx);

// --- Decode Error Handling ---
bool handle_llama_decode_error(llama_context *ctx, llama_batch batch, const char* error_msg = "KV Cache Exhausted. Type 'clear' to reset.", bool should_break = true);
void sync_n_past(llama_context *ctx, int &n_past);

// --- Log Callbacks ---
extern bool first_prompt_displayed;
void dummy_log_callback(enum ggml_log_level level, const char * text, void * user_data);
void custom_log_callback(enum ggml_log_level level, const char * text, void * user_data);

#endif // MODEL_H

