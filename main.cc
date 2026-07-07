#include "llama.h"
#include "common.h"
#include "fit.h"
#include "filesystem.h"
#include "network.h"
#include "tokens.h"
#include "output.h"
#include "signals.h"
#include "server.h"
#include "model.h"
#include "session_utils.h"
#include "token_generator.h"
#include "session.h"
#include "taskset.h"
#include <readline/readline.h>
#include <readline/history.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <clocale>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>

// --- Global State ---
bool is_debug = false;
ofstream chat_log;
ofstream token_log;
ofstream tps_log;
string INITIAL_CWD;

// LIM_HONEST_SPEED: how the t/s diagnostic is computed.
//   0 (default): benchmark-style -- tokens / sample+sync window (first to last token),
//                 covering N sampling ops + (N-1) decode cycles. Matches llama-cli.
//   1: "honest" speed -- tokens / total wall clock time (includes all CPU overhead).
bool honest_speed = false;  // default: benchmark-style

// LIM_SPEED_INTERVAL: how often (in tokens) to update the speed diagnostic.
int speed_update_interval = 100;

// LIM_EXEC_TRUNCATION: max bytes of exec_shell output before truncation.
size_t exec_truncation_limit = 32768;  // default: 32KB

static void diag_impl(const string& formatted_line, const string& msg) {
    // Diagnostic messages (session status, errors, etc.) always go to the
    // terminal regardless of LIM_OUTPUT mode.
    cout << formatted_line << "\n";
    consoleMarkNewline(true);
    cout.flush();
    if (chat_log.is_open()) {
        chat_log << "[" << msg << "]" << "\n\n";
        chat_log.flush();
    }
}

void diag(const string& msg, const char* color) {
    diag_impl(string(color) + msg + "\033[0m", msg);
}

// Model path -- set in main(), read by session.cc for V1 cache writes.
std::string g_model_path;

int main(int argc, char ** argv) {
    setlocale(LC_ALL, "");

    // Read the required username from LIM_AI_USER env var (default: "ai")
    // set user from env var
    const char* ai_user_env = getenv("LIM_AI_USER");
    const char* required_user = ai_user_env && ai_user_env[0] ? ai_user_env : "ai";

    uid_t uid = getuid();
    struct passwd *pw = getpwuid(uid);
    if (pw == nullptr || strcmp(pw->pw_name, required_user) != 0) {
        cerr << "Error: This program must be run as user '" << required_user << "'" << endl;
        return 1;
    }

    // HOME is initialized at global scope via g_homeInit

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        ofstream cwd_file(HOME+"/.cwd");
        if (cwd_file.is_open()) {
            cwd_file << cwd << endl;
            cwd_file.close();
        }
        // Save initial directory for clear command to restore
        INITIAL_CWD = cwd;
    }

    umask(0002);
    atexit([]() {
        cout << "\033[0m";  // Reset terminal colors on exit
        NetworkTools::cleanup_services();
        cleanup_lim_server();
    });

    setup_signals();

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <model_path> [--checkpoints] [restore_file]" << endl;
        return 1;
    }

    g_model_path = argv[1];


    // --checkpoints can appear before or after the save path.
    bool restore_from_file = false;
    bool show_checkpoints = false;
    string restore_path;
    string restore_path_abs;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--checkpoints") == 0) {
            show_checkpoints = true;
        } else {
            if (!restore_path.empty()) {
                cerr << "Error: Unexpected argument: " << argv[i] << endl;
                return 1;
            }
            restore_path = argv[i];
            // Match /save behavior: append .save if not already present
            if (restore_path.size() < std::strlen(SAVE_EXT) || restore_path.compare(restore_path.size() - std::strlen(SAVE_EXT), std::strlen(SAVE_EXT), SAVE_EXT) != 0) {
                restore_path += SAVE_EXT;
            }
            // Prepend LIM_SAVE_DIR to relative paths
            if (!restore_path.empty() && restore_path[0] != '/') {
                restore_path = LIM_SAVE_DIR + "/" + restore_path;
            }
        }
    }
    restore_from_file = !restore_path.empty();
    if (restore_from_file) {
        // Validate the save file exists before loading the model.
        struct stat st;
        if (stat(restore_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            cerr << "Error: Save file not found: " << restore_path << endl;
            return 1;
        }

        // Resolve to absolute path for cache key consistency
        char abs_buf[4096];
        if (realpath(restore_path.c_str(), abs_buf)) {
            restore_path_abs = abs_buf;
        } else {
            restore_path_abs = restore_path;
        }
    }

    // Sampling parameters: all controlled via LIM_* environment variables
    float temp = 0.7f;
    float top_p = 0.8f;
    int32_t top_k = 20;
    float min_p = 0.0f;
    float penalty_present = 1.5f;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    bool use_dummy_thought = false;
    {
        const char* env;
        if ((env = getenv("LIM_TEMP")) != nullptr) temp = atof(env);
        if ((env = getenv("LIM_TOP_P")) != nullptr) top_p = atof(env);
        if ((env = getenv("LIM_TOP_K")) != nullptr) top_k = atoi(env);
        if ((env = getenv("LIM_MIN_P")) != nullptr) min_p = atof(env);
        if ((env = getenv("LIM_PENALTY_PRESENT")) != nullptr) penalty_present = atof(env);
        if ((env = getenv("LIM_PENALTY_REPEAT")) != nullptr) penalty_repeat = atof(env);
        if ((env = getenv("LIM_PENALTY_FREQ")) != nullptr) penalty_freq = atof(env);
        if ((env = getenv("LIM_SEED")) != nullptr) seed = (uint32_t)strtoul(env, nullptr, 10);

        // LIM_THINKING: set to 0 to suppress thinking blocks for faster throughput.
        // Not recommended for math or complex reasoning tasks.
        if ((env = getenv("LIM_THINKING")) != nullptr) {
            use_dummy_thought = (atoi(env) == 0);
        }
    }

    const char* debug_env = getenv("LIM_DEBUG");
    if (debug_env != nullptr && strcmp(debug_env, "1") == 0) {
        is_debug = true;
    }

    // LIM_HONEST_SPEED: 1 = honest wall-clock, 0 = benchmark-style (default)
    {
        const char* env = getenv("LIM_HONEST_SPEED");
        if (env != nullptr && strlen(env) > 0) {
            honest_speed = (atoi(env) != 0);
        }
    }

    // LIM_SPEED_INTERVAL: tokens between speed diagnostic updates (default 100)
    {
        const char* env = getenv("LIM_SPEED_INTERVAL");
        if (env != nullptr && strlen(env) > 0) {
            int val = atoi(env);
            if (val > 0) speed_update_interval = val;
        }
    }

    // LIM_EXEC_TRUNCATION: max bytes of exec_shell output before truncation (default 32768)
    {
        const char* env = getenv("LIM_EXEC_TRUNCATION");
        if (env != nullptr && strlen(env) > 0) {
            long val = strtol(env, nullptr, 10);
            if (val > 0) exec_truncation_limit = static_cast<size_t>(val);
        }
    }

    if (!is_debug) {
        llama_log_set(dummy_log_callback, nullptr);
    } else {
        llama_log_set(custom_log_callback, nullptr);
    }

    mkdir(LIM_LOG_DIR.c_str(), 0775);
    int log_index = 1;
    string log_file_name;
    while (true) {
        log_file_name = LIM_LOG_DIR + "/" + to_string(log_index);
        ifstream check_file(log_file_name.c_str());
        if (!check_file.good()) break;
        log_index++;
    }
    chat_log.open(log_file_name, ios::app);
    if (!chat_log.is_open()) {
        cerr << "Error: Failed to open log file. The directory isn't writeable by user ai." << endl;
        return 1;
    }

    // Open token log file (for LIM_DEBUG=1)
    string token_log_name = LIM_LOG_DIR + "/" + to_string(log_index) + ".tokens";
    if (is_debug) {
        token_log.open(token_log_name, ios::trunc);
        if (token_log.is_open()) {
            token_log << "# Token Log - Session #" << log_index << "\n";
            token_log << "# Format: FEED <label> <token_id> \"<escaped_piece>\"   (tokens fed into context)\n";
            token_log << "#         <seq> <token_id> \"<escaped_piece>\"                     (tokens generated by LLM)\n";
            token_log << "# Escaping: \\n = newline, \\r = carriage return, \\t = tab\n";
        }
    }

    // Open TPS log file (context + tokens/sec diagnostic)
    {
        string tps_log_name = LIM_LOG_DIR + "/" + to_string(log_index) + ".tps";
        tps_log.open(tps_log_name, ios::trunc);
        if (tps_log.is_open()) {
            tps_log << "# TPS Log - Session #" << log_index << "\n";
            tps_log << "# Format: <context_tokens> <tokens_per_second>\n";
        }
    }

    // Initialize the fast stream pipe
    init_output_stream();

    // Start limServer.py if browser output is enabled
    if (should_output_to_browser()) {
        start_lim_server_if_needed();
        // Wait for the server only if we just started it. If it was pre-existing
        // (g_lim_server_pid == -2), it's already listening -- no marker to wait for.
        if (g_lim_server_pid > 0 && !wait_for_server_ready()) {
            log_diagnostic("WARNING: limServer did not become ready. Browser output may fail.", true);
        }
    }

    auto log_entry = [&](const string& role, const string& text) {
        if (chat_log.is_open()) {
            chat_log << "=== " << role << " ===\n" << text << "\n\n";
            chat_log.flush();
        }
    };

    diag("Session #" + to_string(log_index) + " started: type /help to see a list of commands", "\033[35m");
    if (is_debug) Taskset::log_core_detection(std::cerr);
    log_entry("SYSTEM", "Starting LLM Controller Session (#" + to_string(log_index) + ")");

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    auto mparams = llama_model_default_params();
    // Allow overriding model params with LIM_* environment variables
    bool gpu_layers_explicit = false;
    {
        const char* env;
        if ((env = getenv("LIM_GPU_LAYERS")) != nullptr) {
            int val = atoi(env);
            mparams.n_gpu_layers = val;
            // Treat -1 as "not set" -- it's the default and should trigger auto-fit.
            gpu_layers_explicit = (val != -1);
        } else {
            mparams.n_gpu_layers = -1; // -1 means "all layers" (auto-fit)
        }
        if ((env = getenv("LIM_USE_MMAP")) != nullptr) {
            mparams.use_mmap = atoi(env) != 0;
        } else {
            mparams.use_mmap = false;
        }
        if ((env = getenv("LIM_USE_MLOCK")) != nullptr) {
            mparams.use_mlock = atoi(env) != 0;
        } else {
            mparams.use_mlock = true;
        }
    }

    auto cparams = llama_context_default_params();
    // Allow overriding context params with LIM_* environment variables
    bool ctx_explicit = false;
    {
        const char* env;
        if ((env = getenv("LIM_CTX")) != nullptr) {
            cparams.n_ctx = atoi(env);
            ctx_explicit = true;
        } else {
            cparams.n_ctx = 262144;
        }
        if ((env = getenv("LIM_BATCH")) != nullptr) {
            cparams.n_batch = atoi(env);
        } else {
            cparams.n_batch = 2048;
        }
        if ((env = getenv("LIM_UBATCH")) != nullptr) {
            cparams.n_ubatch = atoi(env);
        } else {
            cparams.n_ubatch = 512;
        }
        if ((env = getenv("LIM_THREADS")) != nullptr) {
            cparams.n_threads = atoi(env);
        } else {
            cparams.n_threads = Taskset::p_core_thread_count();
        }
        if ((env = getenv("LIM_THREADS_BATCH")) != nullptr) {
            cparams.n_threads_batch = atoi(env);
        } else {
            cparams.n_threads_batch = Taskset::p_core_thread_count();
        }
    }

    // KV-cache types: override via LIM_CACHE_TYPE_K / LIM_CACHE_TYPE_V
    // Accepted values: F16, Q4_0, Q5_0, Q5_1, Q8_0, Q8_1 (default Q8_0)
    auto parse_kv_type = [](const char* env, ggml_type fallback) -> ggml_type {
        if (!env || !env[0]) return fallback;
        if (strcmp(env, "F16") == 0) return GGML_TYPE_F16;
        if (strcmp(env, "Q4_0") == 0) return GGML_TYPE_Q4_0;
        if (strcmp(env, "Q5_0") == 0) return GGML_TYPE_Q5_0;
        if (strcmp(env, "Q5_1") == 0) return GGML_TYPE_Q5_1;
        if (strcmp(env, "Q8_0") == 0) return GGML_TYPE_Q8_0;
        if (strcmp(env, "Q8_1") == 0) return GGML_TYPE_Q8_1;
        cerr << "Warning: unknown LIM_CACHE_TYPE value '" << env << "', using default." << endl;
        return fallback;
    };
    cparams.type_k = parse_kv_type(getenv("LIM_CACHE_TYPE_K"), GGML_TYPE_Q8_0);
    cparams.type_v = parse_kv_type(getenv("LIM_CACHE_TYPE_V"), GGML_TYPE_Q8_0);

    // Check whether the model file exists before doing anything with it.
    {
        struct stat st;
        if (stat(argv[1], &st) != 0) {
            diag("Model file not found: " + string(argv[1]), "\033[31m");
            return 1;
        }
    }

    // Vectors must live through model loading since mparams holds pointers into them.
    const size_t ndevs = llama_max_devices();
    std::vector<float> tensor_split(ndevs, 0.0f);
    const size_t n_overrides = llama_max_tensor_buft_overrides();
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides(n_overrides, {nullptr, nullptr});

    // Use common_fit_params for GPU layer offloading (matches llama-cli behavior).
    // For MoE models this enables partial layer offloading when the full model
    // doesn't fit in VRAM. Always go through the fitter rather than trying a
    // raw load first, since a failed load can leave GPU state that reduces
    // available memory for subsequent fitting attempts.
    if (!gpu_layers_explicit && mparams.n_gpu_layers < 0) {
        std::vector<size_t> margins(ndevs, 1024 * 1024 * 1024);
        mparams.tensor_split = tensor_split.data();
        mparams.tensor_buft_overrides = tensor_buft_overrides.data();

        uint32_t n_ctx_min = ctx_explicit ? cparams.n_ctx : 8192;

        common_params_fit_status fit_status = common_fit_params(
            argv[1], &mparams, &cparams,
            tensor_split.data(),
            tensor_buft_overrides.data(),
            margins.data(),
            n_ctx_min,
            GGML_LOG_LEVEL_ERROR);

        if (fit_status == COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            // Defer the success message until after model load, when we know total layers.
        } else if (fit_status == COMMON_PARAMS_FIT_STATUS_FAILURE) {
            diag("Warning: could not fully fit model to device memory, using fallback parameters", "\033[33m");
        } else {
            diag("Error during model fitting, proceeding with default parameters", "\033[31m");
        }
    }

    llama_model * model = llama_model_load_from_file(argv[1], mparams);

    if (!model) {
        // Model file exists but loading failed -- likely OOM on GPU or corrupt file.
        struct stat st;
        if (stat(argv[1], &st) == 0) {
            double size_gb = static_cast<double>(st.st_size) / (1024.0 * 1024.0 * 1024.0);
            char sz_buf[32];
            snprintf(sz_buf, sizeof(sz_buf), "%.1f GB", size_gb);
            string device = (mparams.n_gpu_layers > 0) ? "GPU" : "CPU";
            diag("Failed to load model: " + string(argv[1]) + " (" + device + ", " + sz_buf + ")", "\033[31m");
        } else {
            diag("Failed to load model: " + string(argv[1]), "\033[31m");
        }
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Write model metadata to TPS log header
    if (tps_log.is_open()) {
        tps_log << "# Model: " << argv[1] << "\n";
        tps_log << "# Context limit: " << cparams.n_ctx << "\n";
        tps_log << "# GPU layers: " << mparams.n_gpu_layers << "\n";
        tps_log << "# Temperature: " << temp << "\n";
        tps_log << "# Top_p: " << top_p << "\n";
        tps_log << "# Top_k: " << top_k << "\n";
        tps_log << "# Min_p: " << min_p << "\n";
        tps_log << "# Penalty present: " << penalty_present << "\n";
        tps_log << "# Penalty repeat: " << penalty_repeat << "\n";
        tps_log << "# Penalty freq: " << penalty_freq << "\n";
    }

    // Report GPU layer offload with total layer count.
    if (mparams.n_gpu_layers >= 0) {
        int32_t n_layers = llama_model_n_layer(model);
        // Cap at actual layer count: auto-fit may set n_gpu_layers to n_layers+1
        // to include the output layer, but we report only transformer blocks.
        int32_t gpu_layers = std::min(mparams.n_gpu_layers, n_layers);

        // Detect partial offloading: tensor_buft_overrides were populated by
        // common_fit_params when MoE expert weights couldn't fit in VRAM. Each
        // non-null entry corresponds to one layer with some tensors on CPU.
        int32_t partial_layers = 0;
        for (size_t i = 0; i < n_overrides && tensor_buft_overrides[i].pattern != nullptr; i++) {
            partial_layers++;
        }

        if (partial_layers > 0) {
            diag("Model loaded: " + to_string(gpu_layers) + "/" + to_string(n_layers) +
                 " layers on GPU, " + to_string(partial_layers) + " with MoE experts on CPU", "\033[32m");
        } else {
            diag("Model loaded: " + to_string(gpu_layers) + "/" + to_string(n_layers) +
                 " layers on GPU", "\033[32m");
        }
    }

    // Apply remaining context params that fit_params shouldn't touch
    cparams.flash_attn_type = (llama_flash_attn_type)1;
    cparams.offload_kqv = true;

    // Recurrent state snapshots disabled - undo uses the checkpoint mechanism
    // (rs_checkpoint_save/restore) which is independent of n_rs_seq.
    cparams.n_rs_seq = 0;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        int32_t n_layers = llama_model_n_layer(model);
        int32_t gpu_layers = std::min(mparams.n_gpu_layers, n_layers);
        diag("Failed to initialize model context: " + to_string(gpu_layers) +
             "/" + to_string(n_layers) + " layers on GPU. The model may be too large for available device memory.", "\033[31m");
        return 1;
    }

    // Always load and tokenize the system prompt.
    // This is needed even during restore so that `system_tokens` holds only the
    // actual system prompt (not the full conversation).  clear_context() uses
    // system_tokens to re-seed the KV cache after a wipe, so it must be correct.
    string system_prompt;

    bool prompt_file_exists = false;
    {
        string config_prompt_path = LIM_CONFIG_DIR + "/prompt";
        string legacy_prompt_path = HOME + "/prompt";
        ifstream prompt_file(config_prompt_path);
        if (!prompt_file.is_open()) {
            // Backward compatibility: fall back to ~/prompt
            prompt_file.open(legacy_prompt_path);
        }
        if (prompt_file.is_open()) {
            stringstream buffer;
            buffer << prompt_file.rdbuf();
            system_prompt = buffer.str();
            prompt_file.close();
            prompt_file_exists = true;
        }
    }

    // Only append cwd and date if a system prompt file was found.
    // If ~/.config/lim/prompt is missing, leave system_prompt empty for unbiased comparison.
    if (prompt_file_exists) {
        char current_cwd[1024];
        if (getcwd(current_cwd, sizeof(current_cwd)) != nullptr) {
            system_prompt += "\n\nCurrent working directory: " + string(current_cwd) + "\n";
        }

        time_t now = time(nullptr);
        struct tm tm_buf;
        if (localtime_r(&now, &tm_buf)) {
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S %Z", &tm_buf);
            system_prompt += "Current date and time: " + string(time_str) + "\n";
        }
    }

    // Initialize model-specific turn delimiters by asking llama.cpp for the correct tokens.
    init_model_tokens(ctx, model);

    // Optionally append the reserved-token escape contract to the system prompt.
    // Controlled by env var LIM_ESCAPE_CONTRACT (default 0 = hidden, 1 = included).
    {
        const char* env = getenv("LIM_ESCAPE_CONTRACT");
        int include_contract = 0; // default: hidden from prompt (still functional in code)
        if (env) include_contract = atoi(env);
        if (include_contract) {
            system_prompt += "\n\n" + generate_turn_escape_contract();
        }
    }

    // Build system prompt using model-type-aware token vectors (BOS + system turn).
    // If no prompt file was found, skip the system turn entirely to match llama-cli's -sys "" behavior.
    vector<llama_token> system_tokens;
    if (!system_prompt.empty()) {
        system_tokens = build_system_prompt_tokens(ctx, system_prompt);
    }
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(lparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, penalty_repeat, penalty_freq, penalty_present));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
    int n_past = 0;

    if (!restore_from_file) {
        // Normal startup: feed system prompt tokens into KV cache
        batch.n_tokens = 0;
        for (size_t i = 0; i < (int)system_tokens.size(); i++) {
            common_batch_add(batch, system_tokens[i], n_past++, {0}, (i == (int)system_tokens.size() - 1));
            if (is_debug && token_log.is_open()) {
                string piece = common_token_to_piece(ctx, system_tokens[i]);
                token_log << "FEED SYSTEM_PROMPT_INIT " << system_tokens[i] << " \"" << escape_token_piece(piece) << "\"\n";
                token_log.flush();
            }
        }

        if (!system_tokens.empty() && !handle_llama_decode_error(ctx, batch)) return 1;
    } else {
        // Restore from save file.
        //   Header: "LIM_SAVE_V3 git_sha=<sha> n_tokens=<N> n_checkpoints=<M>\n<token_ids_as_int32><checkpoint_offsets_as_int32>"
        vector<llama_token> restored_tokens;
        vector<PromptCheckpoint> restored_checkpoints;
        string saved_sha;
        int saved_session = -1;
        int n_restored = 0;
        bool used_v2 = false;
        bool cache_hit = false;

        // Try compact token save
        if (read_token_save(restore_path, restored_tokens)) {
            // Parse git SHA and session from the header by reading just the first line
            FILE* fp = fopen(restore_path.c_str(), "rb");
            if (fp) {
                char head[128];
                size_t n = fread(head, 1, sizeof(head) - 1, fp);
                fclose(fp);
                head[n] = '\0';
                string header_line(head);
                size_t nl = header_line.find('\n');
                if (nl != string::npos) header_line.resize(nl);
                size_t sha_pos = header_line.find("git_sha=");
                if (sha_pos != string::npos) {
                    string raw = header_line.substr(sha_pos + 8);
                    size_t sp = raw.find(' ');
                    saved_sha = (sp != string::npos) ? raw.substr(0, sp) : raw;
                }
                size_t sess_pos = header_line.find("session=");
                if (sess_pos != string::npos) {
                    string raw = header_line.substr(sess_pos + 8);
                    size_t sp = raw.find(' ');
                    try { saved_session = std::stoi((sp != string::npos) ? raw.substr(0, sp) : raw); } catch (...) { saved_session = -1; }
                }
            }

            // Read checkpoint offsets from V3 save file
            restored_checkpoints = read_checkpoint_offsets(restore_path);

            // Try instant restore from V1 cache before slow token decode
            bool cache_hit_local = false;
            if (!restore_path_abs.empty() && !show_checkpoints) {
                cache_hit_local = try_load_v1_cache(restore_path_abs, restored_tokens, argv[1], ctx);
                if (cache_hit_local && is_debug) {
                    std::string key = get_cache_dir() + "/"; // just to trigger dir creation check
                    diag("Restore from cache.", "\033[35m");
                }
            }
            cache_hit = cache_hit_local;

            if (cache_hit) {
                n_past = (int)llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
                n_restored = n_past;
                used_v2 = true;
                diag_restore(restore_path, n_restored);
            } else {
                used_v2 = true;

                // If checkpoints exist, offer partial restore via readline history navigation
                int restore_limit = (int)restored_tokens.size();
                if (!restored_checkpoints.empty()) {
                    size_t num_cps = restored_checkpoints.size();
                    diag("Save contains " + to_string(num_cps) + " prompt checkpoint" + (num_cps != 1 ? "s" : "") + ".", "\033[35m");
                    diag("Up/down arrows to navigate, Enter to confirm.", "\033[37m");

                    using_history();
                    clear_history();

                    // Add checkpoints oldest-to-newest.
                    // Pressing up from empty line shows the most recent checkpoint first.
                    for (const auto& cp : restored_checkpoints) {
                        string label = cp.prompt.empty() ? "(empty)" : cp.prompt;
                        string entry = label + " (" + to_string(cp.n_past) + " tokens)";
                        add_history(entry.c_str());
                    }

                    char* line = readline("Restore> ");
                    // If Ctrl+C was pressed during readline, stop_generation will be set.
                    // Treat this as a cancellation regardless of what readline returned.
                    if (stop_generation) {
                        if (line) free(line);
                        n_restored = -1; // sentinel: indicates cancelled restore
                        stop_generation = 0;
                    } else if (!line) {
                        // Ctrl+D (EOF) -- skip decode, start fresh session
                        n_restored = -1; // sentinel: indicates cancelled restore
                    } else {
                        string input = line;
                        free(line);
                        if (input == "/quit" || input == "/exit") {
                            cerr << "\nExiting." << endl;
                            llama_free(ctx);
                            llama_model_free(model);
                            llama_backend_free();
                            return 0;
                        }
                        try {
                            // Match against checkpoint display strings.
                            for (const auto& cp : restored_checkpoints) {
                                string label = cp.prompt.empty() ? "(empty)" : cp.prompt;
                                string expected = label + " (" + to_string(cp.n_past) + " tokens)";
                                if (input == expected) {
                                    restore_limit = cp.n_past;
                                    break;
                                }
                            }
                        } catch (...) {}
                    }

                    if (restore_limit < (int)restored_tokens.size()) {
                        diag("Restoring to checkpoint: " + to_string(restore_limit) + " tokens", "\033[35m");
                    }
                }

                if (n_restored != -1) {
                    n_restored = restore_limit;
                    diag_restore(restore_path, n_restored);

                    // Re-decode all tokens through the model to rebuild the KV cache.
                // This is deterministic: same tokens + same model = identical KV cache.
                // Decode in n_batch-sized chunks to stay within llama.cpp's batch limit.
                auto restore_start = chrono::high_resolution_clock::now();
                size_t cp_restore_idx = 0; // index into restored_checkpoints
                for (int i = 0; i < n_restored; i += (int)cparams.n_batch) {
                    int chunk = std::min((int)cparams.n_batch, n_restored - i);
                    batch.n_tokens = 0;
                    for (int j = 0; j < chunk; j++) {
                        common_batch_add(batch, restored_tokens[i + j], n_past++, {0}, (i + j == n_restored - 1));
                    }
                    if (!handle_llama_decode_error(ctx, batch, "KV Cache Exhausted during restore. Type '/clear' to reset.", false)) {
                        sync_n_past(ctx, n_past);
                        cerr << "Error: Failed to decode tokens during restore" << endl;
                        return 1;
                    }
                    // After each chunk, check if we've crossed any checkpoint positions
                    // and save the recurrent state for instant undo support.
                    while (cp_restore_idx < restored_checkpoints.size() &&
                           restored_checkpoints[cp_restore_idx].n_past <= n_past) {
                        llama_memory_rs_checkpoint_save(llama_get_memory(ctx), 0);
                        cp_restore_idx++;
                    }
                }
                sync_n_past(ctx, n_past);

                auto restore_end = chrono::high_resolution_clock::now();
                double restore_elapsed = chrono::duration<double>(restore_end - restore_start).count();
                double restore_speed = (restore_elapsed > 0) ? n_restored / restore_elapsed : 0;
                diag("KV cache regenerated: " + to_string(n_restored) + " tokens at " +
                     std::to_string((int)restore_speed) + " t/s (" +
                     std::to_string((int)restore_elapsed) + "s)", "\033[35m");

                // Auto-write V1 cache for instant future restores (full restore only, skip with --checkpoints)
                if (!show_checkpoints && !restore_path_abs.empty() && restore_limit == (int)restored_tokens.size()) {
                    if (is_debug) {
                        diag("Save to cache.", "\033[35m");
                    }
                    write_v1_cache(restore_path_abs, restored_tokens, argv[1], ctx, "");
                }
                }
            }

        } else {
            cerr << "Error: Save file has an invalid format: " << restore_path << endl;
            return 1;
        }

        // Trim restored_tokens to the actual count (skip if cancelled)
        if (n_restored >= 0) {
            restored_tokens.resize(n_restored);

            // Keep only checkpoints within the restored range.
            restored_checkpoints.erase(
                std::remove_if(restored_checkpoints.begin(), restored_checkpoints.end(),
                    [n_restored](const PromptCheckpoint& cp) { return cp.n_past > n_restored; }),
                restored_checkpoints.end());
        }

        // If restore was cancelled (Ctrl+C/Ctrl+D at checkpoint prompt), start fresh.
        if (n_restored == -1) {
            cerr << "Restore cancelled. Starting fresh session." << endl;
            log_entry("SYSTEM", "Restore cancelled, starting fresh session");
            SessionState state;
            state.all_context_tokens = system_tokens;
            state.log_index = log_index;

            batch.n_tokens = 0;
            n_past = 0;
            for (size_t i = 0; i < (int)system_tokens.size(); i++) {
                common_batch_add(batch, system_tokens[i], n_past++, {0}, (i == (int)system_tokens.size() - 1));
            }
            if (!system_tokens.empty() && !handle_llama_decode_error(ctx, batch)) return 1;

            bool result = run_chat_session(
                ctx, vocab, smpl, batch, n_past, cparams,
                system_tokens, use_dummy_thought,
                state
            );

            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }

        // Show git HEAD status if a SHA was recorded in the save file.
        if (!saved_sha.empty()) {
            FILE* pipe = popen("git rev-parse HEAD 2>/dev/null", "r");
            string current_sha;
            if (pipe) {
                char buf[48];
                if (fgets(buf, sizeof(buf), pipe)) {
                    current_sha = buf;
                    while (!current_sha.empty() && (current_sha.back() == '\n' || current_sha.back() == '\r')) current_sha.pop_back();
                }
                pclose(pipe);
            }
            string short_saved = saved_sha.substr(0, 7);
            if (!current_sha.empty()) {
                string short_current = current_sha.substr(0, 7);
                if (saved_sha == current_sha) {
                    diag_session_restored(saved_session, restored_tokens.size(), (int)cparams.n_ctx, short_saved);
                } else {
                    diag_session_restored(saved_session, restored_tokens.size(), (int)cparams.n_ctx);
                    diag("Git HEAD mismatch: session was at " + short_saved + ", currently at " + short_current, "\033[33m");
                    // Inform the LLM about code changes so it can adapt.
                    {
                        vector<llama_token> git_msg = build_user_assistant_turn(ctx,
                            "Note: Git HEAD has changed since this session was saved (was " + short_saved + ", now " + short_current + "). Code or configuration may have been modified.");
                        restored_tokens.insert(restored_tokens.end(), git_msg.begin(), git_msg.end());
                        batch.n_tokens = 0;
                        for (size_t i = 0; i < (int)git_msg.size(); i++) {
                            common_batch_add(batch, git_msg[i], n_past++, {0}, (i == (int)git_msg.size() - 1));
                        }
                        if (!handle_llama_decode_error(ctx, batch)) return 1;
                    }
                }
            } else {
                diag_session_restored(saved_session, restored_tokens.size(), (int)cparams.n_ctx);
            }
        } else {
            diag_session_restored(saved_session, restored_tokens.size(), (int)cparams.n_ctx);
        }
        log_entry("SYSTEM", "Restored session from " + restore_path);

        // Session state with full conversation history for save/restore tracking.
        // system_tokens remains as the real system prompt only (for clear_context).
        // Filtered checkpoints (up to restore point) are preserved so they're
        // retained in future autosaves.
        SessionState state;
        state.all_context_tokens = restored_tokens;
        state.prompt_checkpoints = restored_checkpoints;
        // After a fast restore, the recurrent checkpoint stack is empty.
        // The prompt_checkpoints list has entries from the save file that have
        // no corresponding stack entries.  Record this offset so undo can
        // translate between the two index spaces.
        // Also save one live checkpoint at the current tail position so that
        // undoing back to the restore boundary can use rs_checkpoint_restore
        // for an instant undo without re-decode.
        state.checkpoint_stack_offset = cache_hit ? (int)restored_checkpoints.size() : 0;
        if (cache_hit && !restored_checkpoints.empty()) {
            llama_memory_rs_checkpoint_save(llama_get_memory(ctx), 0);
        }
        state.log_index = log_index;

        // --- Run the main chat session loop ---
        bool result = run_chat_session(
            ctx, vocab, smpl, batch, n_past, cparams,
            system_tokens, use_dummy_thought,
            state
        );

        // Cleanup
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    // Session state (normal startup)
    SessionState state;
    state.all_context_tokens = system_tokens;
    state.log_index = log_index;

    // --- Run the main chat session loop ---
    bool result = run_chat_session(
        ctx, vocab, smpl, batch, n_past, cparams,
        system_tokens, use_dummy_thought,
        state
    );

    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
