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
#include "mtp.h"
#include "gguf.h"
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

// Version is passed from the Makefile via -DLIM_VERSION="x.y.z"
#ifndef LIM_VERSION
#define LIM_VERSION "0.1.0"
#endif

using namespace std;

// --- Global State ---
bool is_debug = false;
// Deterministic (greedy) mode: temperature 0.  Set in main() after env
// parsing.  In this mode load_system_prompt_text() omits the wall-clock
// timestamp (the only session-varying prompt piece) so greedy runs are
// byte-reproducible -- required for Gate 1 token-identity testing.
bool g_deterministic_mode = false;
ofstream chat_log;
ofstream token_log;
ofstream tps_log;
string INITIAL_CWD;
// LIM_HONEST_SPEED: how the t/s diagnostic is computed.
//   0 (default): benchmark-style -- tokens / sample+sync window (first to last token),
//                 covering N sampling ops + (N-1) decode cycles. Matches llama-cli.
//   1: "honest" speed -- tokens / total wall clock time (includes all CPU overhead).
bool honest_speed = false;  // default: benchmark-style

// LIM_CHATBOT_MODE: benchmarking modes for comparing history handling
//   0 (default): LIM normal mode -- O(1) KV-cache append, no re-decode
//   1: Standard chatbot -- clear cache + re-decode full history each turn.
//      Re-feeds exact saved tokens (preserving per-turn BOS positions) plus new input.
//   2: Cache-aware prefix match -- KV-cache stays in memory, re-tokenizes full conversation
//      text and compares against cached prefix to find where to resume decoding.
//   Both modes force honest speed measurement so TPS includes re-decode overhead.
int chatbot_mode = 0;

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

// Built-in default phrasing for the dummy-thought stub (LIM_THINKING=0), used
// when LIM_DUMMY_THOUGHT is unset.
static const char* DEFAULT_DUMMY_THOUGHT = "The user wants a direct answer. I will output the requested data immediately without preamble.";
std::string g_dummy_thought_text;

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
    if (isatty(STDOUT_FILENO)) cout << "\033[0m";  // Reset terminal colors on exit
    NetworkTools::cleanup_services();
    cleanup_lim_server();
  });

  setup_signals();

  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <model_path> [--version] [restore_file] [--checkpoints]" << endl;
    return 1;
  }

  // Check for --version anywhere in args
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--version") == 0) {
      cout << LIM_VERSION << endl;
      return 0;
    }
  }

  g_model_path = argv[1];


  // Everything after the model path is the /load argument: joined verbatim
  // and injected as the first command, so the CLI behaves exactly like
  // in-session /load -- one save file, with an optional trailing
  // --checkpoints flag (e.g. "cats --checkpoints").
  bool restore_from_file = false;
  string restore_arg;
  for (int i = 2; i < argc; i++) {
    if (i > 2) restore_arg += " ";
    restore_arg += argv[i];
  }
  restore_from_file = !restore_arg.empty();
  if (restore_from_file) {
    // Validate the save file exists before loading the model.
    // Use the same parsing /load applies so the check matches: optional
    // trailing --checkpoints flag, then .save extension and LIM_SAVE_DIR.
    string arg = restore_arg;
    strip_checkpoints_flag(arg);
    string check_path = apply_save_dir(append_save_ext(arg));
    struct stat st;
    if (stat(check_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
      cerr << "Error: Save file not found: " << check_path << endl;
      return 1;
    }
  }

  // Sampling parameters: all controlled via LIM_* environment variables.
  // Variable and env names follow model-card / vLLM spelling (temperature,
  // presence_penalty, ...). Legacy shorter env names (LIM_TEMP, LIM_PENALTY_*)
  // are still accepted as fallbacks; the new name wins if both are set.
  float temperature = 0.7f;
  float top_p = 0.8f;
  int32_t top_k = 20;
  float min_p = 0.0f;
  float presence_penalty = 1.5f;
  float repetition_penalty = 1.0f;
  float frequency_penalty = 0.0f;
  uint32_t seed = LLAMA_DEFAULT_SEED;
  bool use_dummy_thought = false;
  {
    const char* env;
    // New names win over legacy aliases (LIM_TEMP, LIM_PENALTY_*) if both are set.
    if ((env = getenv("LIM_TEMPERATURE")) != nullptr) temperature = atof(env);
    else if ((env = getenv("LIM_TEMP")) != nullptr) temperature = atof(env);
    g_deterministic_mode = (temperature <= 0.0f);
    if ((env = getenv("LIM_TOP_P")) != nullptr) top_p = atof(env);
    if ((env = getenv("LIM_TOP_K")) != nullptr) top_k = atoi(env);
    if ((env = getenv("LIM_MIN_P")) != nullptr) min_p = atof(env);
    if ((env = getenv("LIM_PRESENCE_PENALTY")) != nullptr) presence_penalty = atof(env);
    else if ((env = getenv("LIM_PENALTY_PRESENT")) != nullptr) presence_penalty = atof(env);
    if ((env = getenv("LIM_REPETITION_PENALTY")) != nullptr) repetition_penalty = atof(env);
    else if ((env = getenv("LIM_PENALTY_REPEAT")) != nullptr) repetition_penalty = atof(env);
    if ((env = getenv("LIM_FREQUENCY_PENALTY")) != nullptr) frequency_penalty = atof(env);
    else if ((env = getenv("LIM_PENALTY_FREQ")) != nullptr) frequency_penalty = atof(env);
    if ((env = getenv("LIM_SEED")) != nullptr) seed = (uint32_t)strtoul(env, nullptr, 10);

    // LIM_THINKING: set to 0 to suppress thinking blocks for faster throughput.
    // Not recommended for math or complex reasoning tasks.
    if ((env = getenv("LIM_THINKING")) != nullptr) {
      use_dummy_thought = (atoi(env) == 0);
    }

    // LIM_DUMMY_THOUGHT: override the dummy-thought stub text (LIM_THINKING=0).
    // Unset -> built-in default phrasing. Explicitly empty -> empty thinking block
    // (Qwen 3.8's "no thinking" signal). Otherwise use the given string as-is.
    env = getenv("LIM_DUMMY_THOUGHT");
    g_dummy_thought_text = (env != nullptr) ? env : DEFAULT_DUMMY_THOUGHT;
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

  // LIM_CHATBOT_MODE: benchmarking modes for comparing history handling
  {
    const char* env = getenv("LIM_CHATBOT_MODE");
    if (env != nullptr && strlen(env) > 0) {
      int val = atoi(env);
      if (val >= 1 && val <= 2) chatbot_mode = val;
    }
  }

  // Chatbot modes force honest speed measurement so TPS includes re-decode overhead.
  if (chatbot_mode > 0 && !honest_speed) {
    honest_speed = true;
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
  mkdir(LIM_CACHE_DIR.c_str(), 0775);  int log_index = 1;
  string log_file_name;
  while (true) {
    log_file_name = LIM_LOG_DIR + "/" + to_string(log_index);
    ifstream check_file(log_file_name.c_str());
    if (!check_file.good()) break;
    log_index++;
  }
  // Open chat/token/TPS log files for this session.
  if (!open_session_logs(log_index)) {
    cerr << "Error: Failed to open log file. The directory isn't writeable by user ai." << endl;
    return 1;
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

  announce_new_session(log_index);  // also logs "Starting LLM Controller Session (#N)" to the chat log
  if (chatbot_mode == 1) {
    diag("Chatbot mode 1 enabled: full re-tokenize + re-decode each turn", "\033[33m");
  } else if (chatbot_mode == 2) {
    diag("Chatbot mode 2 enabled: cache-aware prefix match + delta decode each turn", "\033[33m");
  }
  if (is_debug) Taskset::log_core_detection(std::cerr);

  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  auto mparams = llama_model_default_params();

  // --- MTP speculative decoding (LIM_MTP=1) ---
  // Two modes:
  //   1. Embedded MTP: the main model contains nextn tensors -- set
  //      mparams.load_mtp and create the draft ctx from the same model.
  //   2. Sidecar MTP: a separate MTP-only GGUF provides the draft head.
  //      Set LIM_MTP_SIDECAR=<path>; the main model does not need to contain
  //      nextn tensors (e.g. Qwen3.8-Flash-Next quants that strip them).
  //
  // Detection uses the {arch}.nextn_predict_layers metadata key (present in
  // the GGUF header of every shard), the same mechanism llama.cpp uses.
  // Benchmark chatbot modes 1/2 are excluded.
  bool mtp_enabled = false;
  bool mtp_use_sidecar = false;
  bool mtp_sidecar_shared = false;
  int mtp_draft_len = 4;
  int mtp_draft_batch = 128;
  {
    const char* env = getenv("LIM_MTP");
    const bool wanted = (env != nullptr && atoi(env) == 1) && chatbot_mode == 0;
    if (wanted) {
      const char* env_d = getenv("LIM_MTP_DRAFT");
      if (env_d != nullptr && strlen(env_d) > 0) {
        int v = atoi(env_d);
        if (v >= 1 && v <= 32) mtp_draft_len = v;
      }
      const char* env_b = getenv("LIM_MTP_BATCH");
      if (env_b != nullptr && strlen(env_b) > 0) {
        int v = atoi(env_b);
        if (v >= 1 && v <= 512) mtp_draft_batch = v;
      }

      // Determine which GGUF to probe: sidecar if set, otherwise the main model.
      const char* sidecar_path = getenv("LIM_MTP_SIDECAR");
      const bool sidecar_set = (sidecar_path != nullptr && strlen(sidecar_path) > 0);
      const char* probe_path = sidecar_set ? sidecar_path : argv[1];

      struct gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ nullptr };
      struct gguf_context* gc = gguf_init_from_file(probe_path, gp);
      if (gc) {
        const int64_t arch_id = gguf_find_key(gc, "general.architecture");
        if (arch_id >= 0 && gguf_get_kv_type(gc, arch_id) == GGUF_TYPE_STRING) {
          const std::string arch = gguf_get_val_str(gc, arch_id);
          const int64_t nextn_id = gguf_find_key(gc, (arch + ".nextn_predict_layers").c_str());
          const bool has_mtp = (nextn_id >= 0 &&
                                gguf_get_kv_type(gc, nextn_id) == GGUF_TYPE_UINT32 &&
                                gguf_get_val_u32(gc, nextn_id) > 0);
          if (has_mtp && sidecar_set) {
            mtp_enabled = true;
            mtp_use_sidecar = true;
            // Detect shared-head sidecars (no own tok_embd; borrows from main).
            mtp_sidecar_shared = gguf_find_tensor(gc, "token_embd.weight") < 0;
          } else if (has_mtp) {
            mtp_enabled = true;
            mparams.load_mtp = true;
          } else if (sidecar_set) {
            diag("MTP: sidecar file has no nextn (MTP) head -- MTP disabled", "\033[33m");
          } else {
            diag("MTP: LIM_MTP=1 but model has no nextn (MTP) head -- MTP disabled", "\033[33m");
          }
        } else {
          diag("MTP: cannot read architecture from " + std::string(probe_path) + " -- MTP disabled", "\033[33m");
        }
      } else {
        diag("MTP: failed to open " + std::string(probe_path) + " -- MTP disabled", "\033[33m");
      }
      if (gc) gguf_free(gc);
    }
  }

  // Allow overriding model params with LIM_* environment variables
  bool gpu_layers_explicit = false;
  {
    const char* env, *env2;
    if ((env = getenv("LIM_GPU_LAYERS")) != nullptr) {
      int val = atoi(env);
      mparams.n_gpu_layers = val;
      // Treat -1 as "not set" -- it's the default and should trigger auto-fit.
      gpu_layers_explicit = (val != -1);
    } else {
      mparams.n_gpu_layers = -1; // -1 means "all layers" (auto-fit)
    }
    if ((env = getenv("LIM_USE_MMAP")) != nullptr && (env2 = getenv("LIM_USE_MLOCK")) != nullptr) {
      bool use_mmap = atoi(env) != 0;
      bool use_mlock = atoi(env2) != 0;
      if (use_mmap && use_mlock) mparams.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
      else if (use_mmap) mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
      else if (use_mlock) mparams.load_mode = LLAMA_LOAD_MODE_MLOCK;
      else mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    } else {
      // Defaults: mmap=off, mlock=on
      mparams.load_mode = LLAMA_LOAD_MODE_MLOCK;
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
      cparams.n_ctx = LIM_DEFAULT_CTX;
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

  // GPU layer offloading strategy:
  //   1. Try loading all layers on GPU directly (no margin, no fitter).
  //   2. If that fails (OOM), fall back to common_fit_params with a 1 GiB
  //      margin to find the maximum subset of layers that fits in VRAM.
  llama_model * model = nullptr;
  bool used_fitter = false;

  if (!gpu_layers_explicit && mparams.n_gpu_layers < 0) {
    // Step 1: attempt full GPU offload with zero margin.
    model = llama_model_load_from_file(argv[1], mparams);

    if (!model) {
      used_fitter = true;
      // Step 2: direct load failed -- use the fitter to find a workable split.
      // 1 GiB margin: accounts for CUDA context overhead and allocator
      // fragmentation not captured by the fitter's memory model.
      std::vector<size_t> margins(ndevs, (size_t)(1.0 * 1024 * 1024 * 1024));
      mparams.tensor_split = tensor_split.data();
      mparams.tensor_buft_overrides = tensor_buft_overrides.data();

      uint32_t n_ctx_min = cparams.n_ctx;

      // When MTP is enabled, account for the draft's VRAM cost so the fitter
      // doesn't fill VRAM and leave no room for the sidecar/draft context.
      //   - Embedded MTP (no sidecar): the draft context shares the main model's
      //     weights, so pass it as an `extra` model with shares_model=true. It
      //     loads fine standalone (same file).
      //   - Any sidecar (shared OR self-contained): MTP draft heads are rejected
      //     when loaded on their own (missing trunk tensors), so the fitter can't
      //     measure them. Skip the extra model and inflate the per-device margin
      //     by the sidecar file size plus an estimate of the draft KV/compute.
      common_fit_extra_model * extra_ptr = nullptr;
      llama_model_params mparams_mtp;
      llama_context_params cparams_mtp;
      if (mtp_enabled) {
        mparams_mtp = llama_model_default_params();
        mparams_mtp.n_gpu_layers = -1;
        mparams_mtp.load_mtp = true;
        mparams_mtp.load_mode = LLAMA_LOAD_MODE_NONE;
        cparams_mtp = cparams;
        cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cparams_mtp.n_rs_seq = 0;
        cparams_mtp.n_seq_max = 1;
        cparams_mtp.n_batch = std::min<int>(cparams.n_batch, mtp_draft_batch);
        cparams_mtp.n_ubatch = std::min<int>(cparams.n_ubatch, mtp_draft_batch);
        cparams_mtp.n_outputs_max = 2;
        cparams_mtp.n_outputs_max_per_seq = 2;

        if (!mtp_use_sidecar) {
          // Embedded MTP: same model file, draft shares weights. Measure as extra.
          const common_fit_extra_model extra_mtp = {
              /*.path_model   =*/ argv[1],
              /*.mparams      =*/ &mparams_mtp,
              /*.cparams      =*/ &cparams_mtp,
              /*.shares_model =*/ true,
          };
          extra_ptr = const_cast<common_fit_extra_model*>(&extra_mtp);
        } else {
          // Sidecar (shared or self-contained): cannot be measured standalone.
          // Reserve the sidecar file size (weights go to GPU) + draft KV/compute.
          size_t extra_vram = 0;
          struct stat st;
          if (stat(getenv("LIM_MTP_SIDECAR"), &st) == 0) {
            extra_vram = (size_t)st.st_size;  // draft head weights, all on GPU
          }
          // Draft KV + compute: one full-attention layer. ~300 MB per 100K tokens
          // at q8_0 (measured), plus ~4 MB per draft-batch row for compute buffers.
          extra_vram += (size_t)(cparams.n_ctx / 100000.0 * 300.0 * 1024.0 * 1024.0);
          extra_vram += (size_t)(mtp_draft_batch) * 4 * 1024 * 1024;
          for (size_t i = 0; i < margins.size(); i++) {
            margins[i] += extra_vram;
          }
        }
      }

      common_params_fit_status fit_status = common_fit_params(
        argv[1], &mparams, &cparams,
        tensor_split.data(),
        tensor_buft_overrides.data(),
        margins.data(),
        n_ctx_min,
        extra_ptr,
        GGML_LOG_LEVEL_ERROR);

      // On success the message is deferred until after model load, when we know total layers.
      if (fit_status == COMMON_PARAMS_FIT_STATUS_FAILURE) {
        diag("Warning: could not fully fit model to device memory, using fallback parameters", "\033[33m");
      } else if (fit_status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
        diag("Error during model fitting, proceeding with default parameters", "\033[31m");
      }

      model = llama_model_load_from_file(argv[1], mparams);
    }
  } else {
    model = llama_model_load_from_file(argv[1], mparams);
  }

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
    tps_log << "# Temperature: " << temperature << "\n";
    tps_log << "# Top_p: " << top_p << "\n";
    tps_log << "# Top_k: " << top_k << "\n";
    tps_log << "# Min_p: " << min_p << "\n";
    tps_log << "# Presence penalty: " << presence_penalty << "\n";
    tps_log << "# Repetition penalty: " << repetition_penalty << "\n";
    tps_log << "# Frequency penalty: " << frequency_penalty << "\n";
    tps_log << "# Chatbot mode: " << chatbot_mode << "\n";
    tps_log << "# MTP: " << (mtp_enabled
                       ? std::string("enabled (draft=") + std::to_string(mtp_draft_len) +
                         ", batch=" + std::to_string(mtp_draft_batch) +
                         (mtp_use_sidecar ? ", sidecar)" : ")")
                       : "disabled") << "\n";
    tps_log << "# Format: <context_tokens> <tokens_per_second>\n";
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

    const char* verb = used_fitter ? "fitted" : "loaded";
    if (partial_layers > 0) {
      diag(string("Model ") + verb + ": " + to_string(gpu_layers) + "/" + to_string(n_layers) +
           " layers on GPU, " + to_string(partial_layers) + " with MoE experts on CPU", "\033[32m");
    } else {
      diag(string("Model ") + verb + ": " + to_string(gpu_layers) + "/" + to_string(n_layers) +
           " layers on GPU", "\033[32m");
    }
  }

  // Apply remaining context params that fit_params shouldn't touch
  cparams.flash_attn_type = (llama_flash_attn_type)1;
  cparams.offload_kqv = true;

  // Recurrent state snapshots: the rs_checkpoint stack (turn boundaries,
  // /undo) works with n_rs_seq = 0.  MTP verification additionally needs the
  // per-token snapshot window (n_rs_seq = draft length) so a partially
  // accepted draft batch can be rolled back mid-turn on hybrid models.
  cparams.n_rs_seq = mtp_enabled ? (uint32_t)mtp_draft_len : 0;

  llama_context * ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    int32_t n_layers = llama_model_n_layer(model);
    int32_t gpu_layers = std::min(mparams.n_gpu_layers, n_layers);
    diag("Failed to initialize model context: " + to_string(gpu_layers) +
         "/" + to_string(n_layers) + " layers on GPU. The model may be too large for available device memory.", "\033[31m");
    return 1;
  }

  // Create the MTP draft context (before the system prompt feed, so the
  // mirror hook covers the initial prefill too).
  llama_model* model_mtp = nullptr;  // sidecar model (freed at exit)
  if (mtp_enabled) {
    if (mtp_use_sidecar) {
      const char* sidecar_path = getenv("LIM_MTP_SIDECAR");
      auto scparams = llama_model_default_params();
      scparams.n_gpu_layers = -1;    // tiny model, all on GPU
      scparams.load_mtp = true;
      scparams.load_mode = LLAMA_LOAD_MODE_NONE;  // tiny model, no need for mmap
      model_mtp = llama_model_load_from_file(sidecar_path, scparams);
      if (!model_mtp) {
        diag("MTP: failed to load sidecar '" + std::string(sidecar_path) + "' -- MTP disabled", "\033[33m");
        mtp_enabled = false;
      }
    }
    if (mtp_enabled) {
      llama_model* draft_model = mtp_use_sidecar ? model_mtp : model;
      g_mtp = MtpSpeculator::create(ctx, draft_model, cparams, mtp_draft_len, mtp_draft_batch);
      if (!g_mtp) {
        mtp_enabled = false;
        if (model_mtp) { llama_model_free(model_mtp); model_mtp = nullptr; }
      }
    }
  }

  // Always load and tokenize the system prompt.
  // This is needed even during restore so that `system_tokens` holds only the
  // actual system prompt (not the full conversation).  clear_context() uses
  // system_tokens to re-seed the KV cache after a wipe, so it must be correct.
  // If no prompt file exists, system_prompt stays empty (unbiased comparison).
  string system_prompt;
  load_system_prompt_text(system_prompt);

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
  int32_t n_vocab = llama_vocab_n_tokens(vocab);
  // Upstream C API argument order: (penalty_last_n, repeat, freq, present).
  llama_sampler_chain_add(smpl, llama_sampler_init_penalties(n_vocab, 64, repetition_penalty, frequency_penalty, presence_penalty));
  llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(temperature, 0.0f, 1.0f));
  llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

  llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
  int n_past = 0;

  // Feed system prompt tokens into KV cache.
  // (CLI restore: the injected /load command re-seeds the cache from the
  // save file, so the system prompt decode here is simply discarded.)
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

  // Session state. When started with a restore file, "/load <path>
  // [--checkpoints]" is injected as the first input and handled by the
  // normal /load path -- identical to in-session /clear + /load.
  SessionState state;
  state.all_context_tokens = system_tokens;
  // Benchmark modes 1/2: seed the canonical conversation text with exactly
  // what was just fed (an empty prompt means the empty/invalid text and the
  // first mode 1/2 turn reconstructs it from the tracker).
  if (chatbot_mode == 1 || chatbot_mode == 2) {
    state.conversation_text = build_system_turn_text(system_prompt);
  }
  state.log_index = log_index;
  if (restore_from_file) {
    state.cli_restore_path = restore_arg;
  }

  // --- Run the main chat session loop ---
  bool result = run_chat_session(
    ctx, vocab, smpl, batch, n_past, cparams,
    system_tokens, system_prompt, use_dummy_thought,
    state
    );

  // Cleanup (free the MTP draft context before the model it shares;
  // sidecar model is freed after the draft ctx but before the main model)
  if (g_mtp) {
    delete g_mtp;
    g_mtp = nullptr;
  }
  llama_free(ctx);
  if (model_mtp) llama_model_free(model_mtp);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
