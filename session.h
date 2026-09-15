#ifndef SESSION_H
#define SESSION_H

#include "llama.h"
#include "common.h"

#define STR(name, ...) { name, (int)(sizeof(name) - 1), __VA_ARGS__ }
#include <string>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <functional>
#include <chrono>
#include "filesystem.h"

// Forward declaration for INITIAL_CWD from main.cc
extern std::string INITIAL_CWD;

struct SessionState {
  bool auto_continue = false;
  bool reincarnate_mode = false;
  bool reincarnate_first_turn = false;  // True for the first generation after reincarnate completes
  bool prev_was_interrupted = false;
  bool first_turn_done = false;
  int last_t_count = 0;
  double last_elapsed = 0.0;
  double last_decode_time = 0.0;  // Wall-clock generation time (first-token to last-token decode)
  double last_feed_time = 0.0;    // Time spent feeding/re-decoding tokens (chatbot mode)
  int last_n_past = 0;
  std::map<std::string, std::string> file_cache;  // path -> content hash (for cache validation)
  // Internal state (was static inside the function)
  int auto_continue_depth_val = 0;
  bool tool_interrupt_pending = false;
  std::string partial_tool_text;
  // All tokens fed into context, for save/restore
  std::vector<llama_token> all_context_tokens;
  // Benchmark modes only (LIM_CHATBOT_MODE=1/2; mode 0 never touches it): the
  // canonical text that, fed chunk-wise as done by the build_* helpers in
  // model.cc, reproduces all_context_tokens -- i.e. the fully templated
  // conversation (system turn + user/assistant turns, including the streamed
  // text of generated tokens) that a text-based chatbot or llama-server would
  // hold and re-tokenize on the next turn.  EMPTY means invalid: the next mode
  // 1/2 turn rebuilds it from the cached tokens (one-time detokenize
  // round-trip).  In-memory only; never written to save files.
  // Tool-result and correction-cycle feeds are NOT appended (tools don't occur
  // in the benchmark runs): if tools ever do run in these modes, the text is
  // simply shorter than the tracker -- mode 2 handles that as re-tokenization
  // drift (the anchored re-decode path rebuilds the text), and mode 1
  // re-tokenizes the text it holds (the model just can't see the tool results,
  // as a text chatbot whose client dropped them would).
  std::string conversation_text;
  // Token positions and prompt text at each prompt return, for partial restore
  std::vector<PromptCheckpoint> prompt_checkpoints;
  // Number of historical checkpoints not present in the live recurrent checkpoint
  // stack (e.g., after a fast restore from cache where only new prompts get saved).
  // Used to offset rs_checkpoint_restore/prune indices during undo.
  int checkpoint_stack_offset = 0;
  // Log file index (set by main.cc), so save files match chat log numbering
  int log_index = 0;
  // CLI restore: the /load argument (path plus optional trailing
  // --checkpoints) to inject as the first command (set by main.cc when
  // started with a restore argument). Empty = none.
  std::string cli_restore_path;
  // Path from the most recent search_file call, for edit_file path inference.
  std::string last_search_path;
  // Tool-correction checkpoint: saved at start of each generate_response()
  int tool_correction_n_past = 0;
  bool has_tool_correction_checkpoint = false;
  // True once this malformed call's single correction attempt has been
  // consumed.  Cleared on a validated correction injection and at prompt
  // return; a second malformed call with it set ejects to the prompt.
  bool correction_attempted_this_turn = false;
  // Stack index of the most recent tool-correction checkpoint (for pruning).
  int tool_correction_checkpoint_idx = -1;
  // Correction mode: set by tool_executor when a bad tool call needs retry.
  bool tool_correction_mode = false;
};

// Run the main chat session loop.
// Returns true to continue, false when user types quit/exit.
bool run_chat_session(
  // LLM context and model (not owned)
  llama_context* ctx,
  const llama_vocab* vocab,
  llama_sampler* smpl,
  llama_batch& batch,
  int& n_past,
  const llama_context_params& cparams,

  // System prompt tokens
  const std::vector<llama_token>& system_tokens,

  // System prompt text (same content the tokens were built from; keeps the
  // token/text pair in lockstep for conversation_text reconstruction in
  // benchmark modes 1/2)
  const std::string& system_prompt_text,

  // Configuration
  bool use_dummy_thought,

  // Session state (consolidated into a single struct)
  SessionState& state
  );

#endif // SESSION_H
