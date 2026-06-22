#include "llama.h"
#include "common.h"
#include "filesystem.h"
#include "network.h"
#include "tokens.h"
#include "output.h"
#include "signals.h"
#include "server.h"
#include "model.h"
#include "session.h"
#include "taskset.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <set>
#include <clocale>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>

// --- Helper to escape token piece strings for token log ---
static string escape_token_piece(const string& s) {
    string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c == '"') out += "\\\"";
        else out += c;
    }
    return out;
}

// --- Global State ---
bool is_debug = false;
ofstream chat_log;
ofstream token_log;
string INITIAL_CWD;

static void diag_impl(const string& formatted_line, const string& msg) {
    // Diagnostic messages (session status, errors, etc.) always go to the
    // terminal regardless of LLLM_OUTPUT mode.
    cout << formatted_line << "\n";
    cout.flush();
    if (chat_log.is_open()) {
        chat_log << "[" << msg << "]" << "\n\n";
        chat_log.flush();
    }
}

void diag(const string& msg, const char* color) {
    diag_impl(string(color) + "[" + msg + "]\033[0m", msg);
}

// Model path -- set in main(), read by session.cc for V1 cache writes.
std::string g_model_path;

int main(int argc, char ** argv) {
    setlocale(LC_ALL, "");

    // Read the required username from LLLM_AI_USER env var (default: "ai")
    const char* ai_user_env = getenv("LLLM_AI_USER");
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
        cleanup_lllm_server();
    });

    setup_signals();

    if (argc < 2 || argc > 3) {
        cerr << "Usage: " << argv[0] << " <model_path> [restore_file]" << endl;
        return 1;
    }

    g_model_path = argv[1];

    // Check if we are restoring from a save file
    bool restore_from_file = (argc == 3);
    string restore_path;
    string restore_path_abs;
    if (restore_from_file) {
        restore_path = argv[2];
        // Match /save behavior: append .save if not already present
        if (restore_path.size() < 5 || restore_path.substr(restore_path.size() - 5) != ".save") {
            restore_path += ".save";
        }
        // Resolve to absolute path for cache key consistency
        char abs_buf[4096];
        if (realpath(restore_path.c_str(), abs_buf)) {
            restore_path_abs = abs_buf;
        } else {
            restore_path_abs = restore_path;
        }
    }

    // Sampling parameters: all controlled via LLLM_* environment variables
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
        if ((env = getenv("LLLM_TEMP")) != nullptr) temp = atof(env);
        if ((env = getenv("LLLM_TOP_P")) != nullptr) top_p = atof(env);
        if ((env = getenv("LLLM_TOP_K")) != nullptr) top_k = atoi(env);
        if ((env = getenv("LLLM_MIN_P")) != nullptr) min_p = atof(env);
        if ((env = getenv("LLLM_PENALTY_PRESENT")) != nullptr) penalty_present = atof(env);
        if ((env = getenv("LLLM_PENALTY_REPEAT")) != nullptr) penalty_repeat = atof(env);
        if ((env = getenv("LLLM_PENALTY_FREQ")) != nullptr) penalty_freq = atof(env);
        if ((env = getenv("LLLM_SEED")) != nullptr) seed = (uint32_t)strtoul(env, nullptr, 10);

        // LLLM_THINKING: set to 0 to suppress thinking blocks for faster throughput.
        // Not recommended for math or complex reasoning tasks.
        if ((env = getenv("LLLM_THINKING")) != nullptr) {
            use_dummy_thought = (atoi(env) == 0);
        }
    }

    const char* debug_env = getenv("LLLM_DEBUG");
    if (debug_env != nullptr && strcmp(debug_env, "1") == 0) {
        is_debug = true;
    }

    if (!is_debug) {
        llama_log_set(dummy_log_callback, nullptr);
    } else {
        llama_log_set(custom_log_callback, nullptr);
    }

    mkdir("log", 0775);
    int log_index = 1;
    string log_file_name;
    while (true) {
        log_file_name = "log/" + to_string(log_index);
        ifstream check_file(log_file_name.c_str());
        if (!check_file.good()) break;
        log_index++;
    }
    chat_log.open(log_file_name, ios::app);
    if (!chat_log.is_open()) {
        cerr << "Error: Failed to open log file. The directory isn't writeable by user ai." << endl;
        return 1;
    }

    // Open token log file (for LLLM_DEBUG=1)
    string token_log_name = "log/" + to_string(log_index) + ".tokens";
    if (is_debug) {
        token_log.open(token_log_name, ios::trunc);
        if (token_log.is_open()) {
            token_log << "# Token Log - Session #" << log_index << "\n";
            token_log << "# Format: FEED <label> <token_id> \"<escaped_piece>\"   (tokens fed into context)\n";
            token_log << "#         <seq> <token_id> \"<escaped_piece>\"                     (tokens generated by LLM)\n";
            token_log << "# Escaping: \\n = newline, \\r = carriage return, \\t = tab\n";
        }
    }

    // Initialize the fast stream pipe
    init_output_stream();

    // Start lllmServer.py if browser output is enabled
    if (should_output_to_browser()) {
        start_lllm_server_if_needed();
        // Wait for the server only if we just started it. If it was pre-existing
        // (g_lllm_server_pid == -2), it's already listening -- no marker to wait for.
        if (g_lllm_server_pid > 0 && !wait_for_server_ready()) {
            log_diagnostic("WARNING: lllmServer did not become ready. Browser output may fail.", true);
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
    // Allow overriding model params with LLLM_* environment variables
    {
        const char* env;
        if ((env = getenv("LLLM_GPU_LAYERS")) != nullptr) {
            mparams.n_gpu_layers = atoi(env);
        } else {
            mparams.n_gpu_layers = 999;
        }
        if ((env = getenv("LLLM_USE_MMAP")) != nullptr) {
            mparams.use_mmap = atoi(env) != 0;
        } else {
            mparams.use_mmap = false;
        }
        if ((env = getenv("LLLM_USE_MLOCK")) != nullptr) {
            mparams.use_mlock = atoi(env) != 0;
        } else {
            mparams.use_mlock = true;
        }
    }

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    ModelType model_type = detect_model_type(vocab);

    auto cparams = llama_context_default_params();
    // Allow overriding context params with LLLM_* environment variables
    {
        const char* env;
        if ((env = getenv("LLLM_CTX")) != nullptr) {
            cparams.n_ctx = atoi(env);
        } else {
            cparams.n_ctx = 262144;
        }
        if ((env = getenv("LLLM_BATCH")) != nullptr) {
            cparams.n_batch = atoi(env);
        } else {
            cparams.n_batch = 2048;
        }
        if ((env = getenv("LLLM_UBATCH")) != nullptr) {
            cparams.n_ubatch = atoi(env);
        } else {
            cparams.n_ubatch = 512;
        }
        if ((env = getenv("LLLM_THREADS")) != nullptr) {
            cparams.n_threads = atoi(env);
        } else {
            cparams.n_threads = 8;
        }
        if ((env = getenv("LLLM_THREADS_BATCH")) != nullptr) {
            cparams.n_threads_batch = atoi(env);
        } else {
            cparams.n_threads_batch = 8;
        }
    }
    cparams.flash_attn_type = (llama_flash_attn_type)1;
    cparams.offload_kqv = true;

    // KV-cache types: override via LLLM_TYPE_K / LLLM_TYPE_V
    // Accepted values: F16, Q8_0, Q4_0, Q5_0, Q5_1, Q8_1 (default Q8_0)
    auto parse_kv_type = [](const char* env, ggml_type fallback) -> ggml_type {
        if (!env || !env[0]) return fallback;
        if (strcmp(env, "F16") == 0) return GGML_TYPE_F16;
        if (strcmp(env, "Q4_0") == 0) return GGML_TYPE_Q4_0;
        if (strcmp(env, "Q5_0") == 0) return GGML_TYPE_Q5_0;
        if (strcmp(env, "Q5_1") == 0) return GGML_TYPE_Q5_1;
        if (strcmp(env, "Q8_0") == 0) return GGML_TYPE_Q8_0;
        if (strcmp(env, "Q8_1") == 0) return GGML_TYPE_Q8_1;
        cerr << "Warning: unknown LLLM_TYPE value '" << env << "', using default." << endl;
        return fallback;
    };
    cparams.type_k = parse_kv_type(getenv("LLLM_TYPE_K"), GGML_TYPE_Q8_0);
    cparams.type_v = parse_kv_type(getenv("LLLM_TYPE_V"), GGML_TYPE_Q8_0);

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    // Always load and tokenize the system prompt.
    // This is needed even during restore so that `system_tokens` holds only the
    // actual system prompt (not the full conversation).  clear_context() uses
    // system_tokens to re-seed the KV cache after a wipe, so it must be correct.
    string system_prompt;

    ifstream prompt_file(HOME+"/prompt");
    if (prompt_file.is_open()) {
        stringstream buffer;
        buffer << prompt_file.rdbuf();
        system_prompt = buffer.str();
        prompt_file.close();
    }

    // Append current working directory and date to system prompt
    {
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

    string formatted_system_prompt = string(Tokens::TURN_START) + "system\n" + system_prompt + Tokens::TURN_END + "\n";
    vector<llama_token> system_tokens = common_tokenize(ctx, formatted_system_prompt, true, true);
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

        if (!handle_llama_decode_error(ctx, batch)) return 1;
    } else {
        // Restore from save file.
        // V2 format: compact token sequence -- re-decode through model to rebuild KV cache.
        //   Header: "LLLM_SAVE_V2 git_sha=<sha> n_tokens=<N>\n<token_ids_as_int32>"
        vector<llama_token> restored_tokens;
        string saved_sha;
        int n_restored = 0;
        bool used_v2 = false;

        // Try V2 (compact token save) first
        if (read_token_save(restore_path, restored_tokens)) {
            // Parse git SHA from the header by reading just the first line
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
            }

            // Try instant restore from V1 cache before slow token decode
            bool cache_hit = false;
            if (!restore_path_abs.empty()) {
                cache_hit = try_load_v1_cache(restore_path_abs, argv[1], saved_sha, ctx);
                if (cache_hit && is_debug) {
                    std::string key = get_cache_dir() + "/"; // just to trigger dir creation check
                    diag("Restore from cache.", "\033[35m");
                }
            }

            auto log_restore = [&](const string& path, int count) {
                diag("Restoring session from " + path + "... (" + to_string(count) + " tokens)", "\033[35m");
            };

            if (cache_hit) {
                n_past = (int)llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
                n_restored = n_past;
                used_v2 = true;
                log_restore(restore_path, n_restored);
            } else {
                used_v2 = true;
                n_restored = (int)restored_tokens.size();
                log_restore(restore_path, n_restored);

                // Re-decode all tokens through the model to rebuild the KV cache.
                // This is deterministic: same tokens + same model = identical KV cache.
                // Decode in n_batch-sized chunks to stay within llama.cpp's batch limit.
                auto restore_start = chrono::high_resolution_clock::now();
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
                }
                sync_n_past(ctx, n_past);

                auto restore_end = chrono::high_resolution_clock::now();
                double restore_elapsed = chrono::duration<double>(restore_end - restore_start).count();
                double restore_speed = (restore_elapsed > 0) ? n_restored / restore_elapsed : 0;
                diag("KV cache regenerated: " + to_string(n_restored) + " tokens at " +
                     std::to_string((int)restore_speed) + " t/s (" +
                     std::to_string((int)restore_elapsed) + "s)", "\033[35m");

                // Auto-write V1 cache for instant future restores
                if (!restore_path_abs.empty()) {
                    if (is_debug) {
                        diag("Save to cache.", "\033[35m");
                    }
                    write_v1_cache(restore_path_abs, argv[1], saved_sha, ctx);
                }
            }
        } else {
            // V2 read failed -- not a supported save format
            cerr << "Error: Save file is not a valid token save: " << restore_path << endl;
            return 1;
        }

        // Trim restored_tokens to the actual count
        restored_tokens.resize(n_restored);

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
                    diag("Session restored: " + to_string(restored_tokens.size()) + " tokens loaded (git: " + short_saved + ")", "\033[32m");
                } else {
                    diag("Session restored: " + to_string(restored_tokens.size()) + " tokens loaded", "\033[32m");
                    diag("Git HEAD mismatch: session was at " + short_saved + ", currently at " + short_current, "\033[33m");
                }
            } else {
                diag("Session restored: " + to_string(restored_tokens.size()) + " tokens loaded", "\033[32m");
            }
        } else {
            diag("Session restored: " + to_string(restored_tokens.size()) + " tokens loaded", "\033[32m");
        }
        log_entry("SYSTEM", "Restored session from " + restore_path);

        // Session state with full conversation history for save/restore tracking.
        // system_tokens remains as the real system prompt only (for clear_context).
        SessionState state;
        state.all_context_tokens = restored_tokens;
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
