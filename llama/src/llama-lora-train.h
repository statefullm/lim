#pragma once

#include "llama-adapter.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

// Forward declarations
struct ggml_tensor;
struct ggml_backend_sched;
using ggml_backend_sched_t = struct ggml_backend_sched *;
struct llama_model;

/// Trainable LoRA adapter for fine-tuning.
///
/// Mirrors llama_adapter_lora but with trainable F32 tensors (A and B matrices).
/// The base model weights stay quantized and frozen; only the small LoRA
/// A/B matrices are updated during training.
struct llama_lora_trainable {
    // Map from base tensor name -> (A, B) pair
    // A shape: [input_dim, rank]   (column-major: ne[0]=input_dim, ne[1]=rank)
    // B shape: [rank, output_dim]  (column-major: ne[0]=rank,     ne[1]=output_dim)
    std::unordered_map<std::string, std::pair<ggml_tensor*, ggml_tensor*>> ab_map;

    // GPU contexts and buffers (one per backend buffer type)
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;           // LoRA alpha scaling factor
    int32_t rank;          // LoRA rank r

    // Pointer to base model (for tensor shape lookup)
    const llama_model* model = nullptr;

    llama_lora_trainable() = default;
    ~llama_lora_trainable() = default;

    // Constructor: initialize zero A matrices and random B matrices
    // for all target weight tensors in the model that match the given patterns.
    // Tensors are placed on the same backend buffer type as their base weight.
    void init(const llama_model& model, int32_t rank, float alpha,
              const std::vector<std::string>& target_patterns,
              ggml_backend_sched_t sched);

    // Lookup: given a base weight tensor, return its LoRA A/B pair if it exists
    const std::pair<ggml_tensor*, ggml_tensor*>* get_weight(ggml_tensor* w) const;

    // Save trained adapter as GGUF file compatible with llama_adapter_lora_init
    void save_adapter(const char* path, const llama_model& model);
};

using llama_lora_trainable_ptr = std::unique_ptr<llama_lora_trainable>;

