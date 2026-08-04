#include "llama-lora-train.h"

#include "llama-model-saver.h"
#include "llama-arch.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: check if a tensor name matches any of the given regex patterns
// ---------------------------------------------------------------------------
static bool name_matches(const std::string& name,
                         const std::vector<std::regex>& patterns) {
    for (const auto& pat : patterns) {
        if (std::regex_search(name, pat)) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Helper: check if a tensor should be skipped (norms, biases, embeddings)
// ---------------------------------------------------------------------------
static bool should_skip_tensor(const std::string& name) {
    if (name.find("norm") != std::string::npos) return true;
    if (name.find("bias") != std::string::npos) return true;
    if (name.find("embd") != std::string::npos) return true;
    if (name.find("rope") != std::string::npos)  return true;
    // Skip output / cls tensors that are not weight matrices
    if (name.find("cls") != std::string::npos)   return true;
    return false;
}

// ---------------------------------------------------------------------------
// Helper: get or create a ggml_context for a given backend buffer type
// ---------------------------------------------------------------------------
static ggml_context* ctx_for_buft(
        ggml_backend_buffer_type_t buft,
        std::map<ggml_backend_buffer_type_t, ggml_context*>& ctx_map,
        std::vector<ggml_context_ptr>& ctxs) {
    auto it = ctx_map.find(buft);
    if (it == ctx_map.end()) {
        ggml_init_params params = {
            /*.mem_size   =*/ 1024 * 1024,  // initial size; tensors allocated separately
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context* buft_ctx = ggml_init(params);
        if (!buft_ctx) {
            fprintf(stderr, "llama-lora-train: failed to create ggml context for buffer type\n");
            return nullptr;
        }
        ctx_map[buft] = buft_ctx;
        ctxs.emplace_back(buft_ctx);
        return buft_ctx;
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// llama_lora_trainable::init
// ---------------------------------------------------------------------------
void llama_lora_trainable::init(const llama_model& model, int32_t rank, float alpha,
                                 const std::vector<std::string>& target_patterns,
                                 ggml_backend_sched_t sched) {
    this->rank  = rank;
    this->alpha = alpha;
    this->model = &model;

    // Compile regex patterns for target modules
    std::vector<std::regex> patterns;
    for (const auto& p : target_patterns) {
        try {
            patterns.emplace_back(p);
        } catch (const std::regex_error& e) {
            fprintf(stderr, "llama-lora-train: invalid regex pattern '%s': %s\n",
                    p.c_str(), e.what());
        }
    }

    // Collect all matching tensors and create A/B pairs
    auto* impl = dynamic_cast<const llama_model_base*>(&model);
    GGML_ASSERT(impl && "model must be a llama_model_base");

    // Map from buffer type -> context for tensor creation
    std::map<ggml_backend_buffer_type_t, ggml_context*> ctx_map;

    // First pass: create tensors in the appropriate contexts (no_alloc)
    for (const auto& [name, tensor] : impl->tensors_by_name) {
        // Check if name matches any target pattern
        if (!name_matches(name, patterns)) continue;

        // Skip non-weight tensors
        if (should_skip_tensor(name)) continue;

        // Skip recurrent layers -- their ops (e.g., GATED_DELTA_NET) have no backward pass
        {
            size_t pos = name.find("blk.");
            if (pos != std::string::npos) {
                uint32_t layer_id = std::stoi(name.substr(pos + 4));
                if (model.hparams.is_recr(layer_id)) {
                    continue;
                }
            }
        }

        // Get backend buffer type for this tensor (place A/B on same device)
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);

        ggml_context* ctx = ctx_for_buft(buft, ctx_map, ctxs);
        if (!ctx) {
            fprintf(stderr, "llama-lora-train: failed to create context for tensor '%s'\n", name.c_str());
            continue;
        }

        // ggml_mul_mat requires t0->ne[0] == t1->ne[0] (shared inner dimension).
        // For weight w with shape [inner_dim, out_dim], input cur is [inner_dim, n_tokens].
        // LoRA: B @ (A @ cur) where A is [inner_dim, rank], B is [rank, out_dim].
        int64_t inner_dim = tensor->ne[0];
        int64_t out_dim   = tensor->ne[1];

        ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner_dim, rank);
        ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out_dim);

        std::string name_a = name + ".lora_a";
        std::string name_b = name + ".lora_b";
        ggml_set_name(A, name_a.c_str());
        ggml_set_name(B, name_b.c_str());

        ab_map[name] = {A, B};
    }

        // Second pass: allocate buffers (one per backend buffer type)
    for (auto& it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context* ctx_dev = it.second;
        ggml_backend_buffer_ptr buf{ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft)};
        if (!buf) {
            fprintf(stderr, "llama-lora-train: failed to allocate buffer for LoRA tensors\n");
            return;
        }
        // Zero-initialize all tensor memory (A starts at zero)
        ggml_backend_buffer_clear(buf.get(), 0);
        bufs.emplace_back(std::move(buf));
    }

    // Third pass: initialize B matrices to random N(0, 1/sqrt(rank))
    // A matrices are already zero from buffer clear above
    std::mt19937 rng(42);  // fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(rank)));

    for (auto& [name, ab] : ab_map) {
        ggml_tensor* B = ab.second;
        if (!B) continue;

        // Get B's data pointer and fill with random values
        size_t n_elements = ggml_nelements(B);
        std::vector<float> b_data(n_elements);
        for (size_t i = 0; i < n_elements; ++i) {
            b_data[i] = dist(rng);
        }
        ggml_backend_tensor_set(B, b_data.data(), 0, n_elements * sizeof(float));

        fprintf(stderr, "llama-lora-train: initialized LoRA for '%s' [%ld x %ld], rank=%d\n",
                name.c_str(), ab.first->ne[0], B->ne[1], rank);
    }

    fprintf(stderr, "llama-lora-train: created %zu trainable LoRA pairs (rank=%d, alpha=%.1f)\n",
            ab_map.size(), rank, alpha);
}

// ---------------------------------------------------------------------------
// llama_lora_trainable::get_weight
// ---------------------------------------------------------------------------
const std::pair<ggml_tensor*, ggml_tensor*>* llama_lora_trainable::get_weight(ggml_tensor* w) const {
    if (!w || !w->name) return nullptr;

    auto it = ab_map.find(w->name);
    if (it != ab_map.end() && it->second.first && it->second.second) {
        return &(it->second);
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// llama_lora_trainable::save_adapter
// ---------------------------------------------------------------------------
void llama_lora_trainable::save_adapter(const char* path, const llama_model& model) {
    fprintf(stderr, "llama-lora-train: saving adapter to '%s'...\n", path);

    // Create a GGUF context for the adapter
    gguf_context* gguf_ctx = gguf_init_empty();

    // Set required metadata keys
    const LLM_KV llm_kv(model.arch);

    // general.type = "adapter"
    gguf_set_val_str(gguf_ctx, llm_kv(LLM_KV_GENERAL_TYPE).c_str(), "adapter");

    // general.architecture
    gguf_set_val_str(gguf_ctx, llm_kv(LLM_KV_GENERAL_ARCHITECTURE).c_str(),
                     llm_arch_name(model.arch));

    // adapter.type = "lora"
    gguf_set_val_str(gguf_ctx, llm_kv(LLM_KV_ADAPTER_TYPE).c_str(), "lora");

    // adapter.lora.alpha
    gguf_set_val_f32(gguf_ctx, llm_kv(LLM_KV_ADAPTER_LORA_ALPHA).c_str(), alpha);

    // Copy relevant model metadata (architecture-specific keys)
    // We need the architecture keys so the adapter can be matched to the model
    const llama_hparams& hparams = model.hparams;

    // Set essential architecture hyperparameters
    gguf_set_val_i32(gguf_ctx, llm_kv(LLM_KV_ATTENTION_HEAD_COUNT).c_str(),
                     static_cast<int32_t>(hparams.n_head()));
    gguf_set_val_i32(gguf_ctx, llm_kv(LLM_KV_CONTEXT_LENGTH).c_str(),
                     static_cast<int32_t>(hparams.n_ctx_train > 0 ? hparams.n_ctx_train : 4096));

    // Add the LoRA A and B tensors to the GGUF file
    for (const auto& [name, ab] : ab_map) {
        ggml_tensor* A = ab.first;
        ggml_tensor* B = ab.second;

        if (!A || !B) continue;

        // Add tensors: <base_name>.lora_a and <base_name>.lora_b
        gguf_add_tensor(gguf_ctx, A);
        gguf_add_tensor(gguf_ctx, B);
    }

    // Write the GGUF file
    size_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    if (!gguf_write_to_file(gguf_ctx, path, false)) {  // false = write meta + tensor data
        fprintf(stderr, "llama-lora-train: failed to write adapter to '%s'\n", path);
    }
    gguf_free(gguf_ctx);

    fprintf(stderr, "llama-lora-train: saved adapter with %zu tensors to '%s'\n",
            n_tensors, path);
}

