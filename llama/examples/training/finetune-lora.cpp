#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-lora-train.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <regex>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

// ---------------------------------------------------------------------------
// Helper: split a semicolon-separated pattern string into individual patterns
// ---------------------------------------------------------------------------
static std::vector<std::string> split_pattern(const std::string& s) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t pos;
    while ((pos = s.find(';', start)) != std::string::npos) {
        std::string token = s.substr(start, pos - start);
        if (!token.empty()) {
            result.push_back(token);
        }
        start = pos + 1;
    }
    std::string last = s.substr(start);
    if (!last.empty()) {
        result.push_back(last);
    }
    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    // LoRA-specific parameters
    int32_t lora_rank   = 64;
    float   lora_alpha  = 128.0f;
    bool    static_graph = false;
    std::string target_modules =
        ".*attn_q.weight;"
        ".*attn_k.weight;"
        ".*attn_v.weight;"
        ".*attn_output.weight;"
        ".*ffn_gate.weight;"
        ".*ffn_down.weight;"
        ".*ffn_up.weight";

    common_init();

    // Strip LoRA-specific arguments from argv before passing to common_params_parse
    // so they don't get rejected as unknown flags
    std::vector<const char*> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((strcmp(arg, "--lora-rank") == 0 && i + 1 < argc)) {
            lora_rank = std::atoi(argv[++i]);
        } else if ((strcmp(arg, "--lora-alpha") == 0 && i + 1 < argc)) {
            lora_alpha = std::atof(argv[++i]);
        } else if ((strcmp(arg, "--target-modules") == 0 && i + 1 < argc)) {
            target_modules = argv[++i];
        } else if (strcmp(arg, "--static-graph") == 0) {
            static_graph = true;
        } else {
            filtered_argv.push_back(arg);
        }
    }
    int fargc = filtered_argv.size();

    if (!common_params_parse(fargc, const_cast<char**>(filtered_argv.data()), params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (lora_rank <= 0) {
        LOG_ERR("lora-rank must be > 0\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // Forward llama.cpp internal log messages only when debugging (LIM_DEBUG=1).
    // During normal training, suppressing the callback avoids per-op overhead.
    if (getenv("LIM_DEBUG")) {
        llama_log_set([](ggml_log_level level, const char * text, void * /*ud*/) {
            common_log_add(common_log_main(), level, "%s", text);
        }, nullptr);
    }

    // Force training-specific settings
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    // Bypass common_init_from_params and follow LIM's original pattern:
    // just set n_gpu_layers high and let the loader place everything on GPU.
    // The fitter (common_fit_params) overestimates memory use because it
    // assumes F32 KV-cache types and full context utilization, which causes
    // it to unnecessarily reduce GPU layers for training workloads where the
    // actual cache is tiny.

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers;  // -1 or large value = all layers on GPU
    mparams.main_gpu     = params.main_gpu;
    mparams.split_mode   = params.split_mode;
    mparams.load_mode    = LLAMA_LOAD_MODE_NONE; // required for training

    auto cparams = llama_context_default_params();
    cparams.n_ctx       = params.n_ctx > 0 ? params.n_ctx : 2048;
    // Hybrid models (Qwen3.5/3.6): the fused GDN kernel has a chunked
    // backward implementation (gdn_back.cu) that parallelizes across token
    // chunks using grid sync, and a sequential forward kernel
    // (gated_delta_net.cu). With -ub 1, transformer MUL_MAT layers keep
    // GPU busy between GDN calls. Larger ubatch sizes benefit from the
    // chunked backward kernel but are limited by the sequential forward.
    const int default_ubatch = 1;
    cparams.n_batch     = params.n_batch > 0 ? params.n_batch : cparams.n_ctx;
    cparams.n_ubatch    = params.n_ubatch > 0 ? params.n_ubatch : default_ubatch;
    cparams.type_k      = params.cache_type_k;
    cparams.type_v      = params.cache_type_v;
    cparams.flash_attn_type = params.flash_attn_type;

    // Use all physical cores for graph construction and CPU-side ops.
    // Graph rebuild is the bottleneck for hybrid models with decomposed GDN.
    {
        FILE* fp = popen("nproc --all", "r");
        if (fp) {
            int ncores = 0;
            fscanf(fp, "%d", &ncores);
            pclose(fp);
            if (ncores > 0) {
                cparams.n_threads = ncores;
                cparams.n_threads_batch = ncores;
            }
        }
    }

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("%s: unable to create context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    // Print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    // Create trainable LoRA
    auto lora = std::make_unique<llama_lora_trainable>();
    auto target_patterns = split_pattern(target_modules);

    LOG_INF("Initializing LoRA: rank=%d, alpha=%.1f, target patterns:\n", lora_rank, lora_alpha);
    for (const auto& pat : target_patterns) {
        LOG_INF("  %s\n", pat.c_str());
    }

    lora->init(*model, lora_rank, lora_alpha, target_patterns, ctx->get_sched());

    // Store trainable LoRA in the context so the graph builder can access it
    llama_set_trainable_loras(ctx, std::move(lora));

    // Tokenize training data and create dataset
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    // Initialize optimizer -- this registers LoRA A/B tensors as trainable params
    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        LOG_INF("\n=== Epoch %d / %d ===\n", lr.epoch + 1, lr.epochs);
        if (static_graph) {
            llama_opt_epoch_static(ctx, dataset, result_train, result_eval, idata_split,
                                   ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        } else {
            llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        }
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    // Save the trained LoRA adapter as a GGUF file
    llama_save_trainable_adapter(ctx, params.out_file.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}

