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

// Version is passed from the Makefile via -DLIM_VERSION="x.y.z"
#ifndef LIM_VERSION
#define LIM_VERSION "0.1.0"
#endif

using namespace std;

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
    const string flag = " --checkpoints";
    if (arg.size() >= flag.size() &&
        arg.compare(arg.size() - flag.size(), flag.size(), flag) == 0) {
      arg.erase(arg.size() - flag.size());
    }
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

  auto log_entry = [&](const string& role, const string& text) {
    if (chat_log.is_open()) {
      chat_log << "=== " << role << " ===\n" << text << "\n\n";
      chat_log.flush();
    }
  };
  announce_new_session(log_index);
  if (chatbot_mode == 1) {
    diag("Chatbot mode 1 enabled: full re-tokenize + re-decode each turn", "\033[33m");
  } else if (chatbot_mode == 2) {
    diag("Chatbot mode 2 enabled: KV-cache save/restore each turn", "\033[33m");
  }
  if (is_debug) Taskset::log_core_detection(std::cerr);
  log_entry("SYSTEM", "Starting LLM Controller Session (#" + to_string(log_index) + ")");

  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  auto mparams = llama_model_default_params();
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

      common_params_fit_status fit_status = common_fit_params(
        argv[1], &mparams, &cparams,
        tensor_split.data(),
        tensor_buft_overrides.data(),
        margins.data(),
        n_ctx_min,
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
    ifstream prompt_file(config_prompt_path);
    if (prompt_file.is_open()) {
      stringstream buffer;
      buffer << prompt_file.rdbuf();
      system_prompt = buffer.str();
      prompt_file.close();
      prompt_file_exists = true;
    }

    // Load site-specific localprompt: check current directory first, then LIM_CONFIG_DIR.
    {
      string cwd_localprompt_path = "./localprompt";
      string config_localprompt_path = LIM_CONFIG_DIR + "/localprompt";
      ifstream localprompt_file(cwd_localprompt_path);
      if (!localprompt_file.is_open()) {
        localprompt_file.open(config_localprompt_path);
      }
      if (localprompt_file.is_open()) {
        stringstream buffer;
        buffer << localprompt_file.rdbuf();
        // Prepend (not append) so site-specific text -- e.g., a Qwen 3.8
        // reasoning-effort instruction placed in localprompt -- lands at the
        // start of the system message, matching where Qwen's own template puts it.
        system_prompt = buffer.str() + "\n" + system_prompt;
        localprompt_file.close();
      }
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
  state.log_index = log_index;
  if (restore_from_file) {
    state.cli_restore_path = restore_arg;
  }

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
