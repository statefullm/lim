#ifndef TOKEN_GENERATOR_H
#define TOKEN_GENERATOR_H

#include "llama.h"
#include "common.h"
#include <string>
#include <vector>

// Helper to escape token piece strings for token log
std::string escape_token_piece(const std::string& s);

// Helper functions for tool call detection (shared between session.cc and token_generator.cc)
size_t find_tool_end_robust(const std::string& text, size_t from_pos, bool* out_in_parameter = nullptr);
void repair_malformed_tool_end(std::string& text, size_t pos);
void _strip_think_and_tool_tags(std::string& str);

class TokenGenerator {
public:
  struct Result {
    std::string text;
    bool has_tool_call = false;
    size_t tool_start = std::string::npos;
    size_t tool_end = std::string::npos;
    bool was_interrupted = false;
    int token_count = 0;
    bool early_exit = false;  // context exhaustion or decode error (not normal EOG)
    double decode_time = 0.0;  // Sum of per-token decode intervals (seconds)
  };

  TokenGenerator(llama_context* ctx, const llama_vocab* vocab,
                 llama_sampler* smpl, llama_batch& batch,
                 int& n_past, const llama_context_params& cparams,
                 double turn_timeout_sec, bool was_mid_tool_call,
                 int last_n_past,
                 std::vector<llama_token>* out_tokens = nullptr,
                 double feed_time = 0.0,
                 bool is_reincarnating = false,
                 bool is_auto_continue = false);

  Result generate();

private:
  llama_context* ctx_;
  const llama_vocab* vocab_;
  llama_sampler* smpl_;
  llama_batch& batch_;
  int& n_past_;
  const llama_context_params& cparams_;
  double turn_timeout_sec_;
  double feed_time_;  // Time spent feeding/re-decoding tokens before generation

  // Internal state
  std::string generated_text_;
  std::string unprinted_text_;
  std::string full_response_;
  size_t print_pos_;
  bool in_tool_call_stream_;
  bool in_parameter_;
  size_t tool_start_;
  size_t tool_end_;
  bool trigger_tool_execution_;
  size_t func_search_pos_;
  bool had_eog_recovery_;
  bool context_warned_this_turn_;
  bool in_thinking_block_;
  size_t think_start_;
  size_t think_end_;
  std::string think_buffer_;
  bool think_buffering_;
  int t_count_;
  int last_n_past_;
  bool was_mid_tool_call_;
  bool is_reincarnating_;
  bool is_auto_continue_;
  std::vector<llama_token>* out_tokens_;  // If non-null, each sampled token is appended here

  // Silent-loop detector: count tokens generated outside parameters while
  // inside a tool call stream (FUNC_START found, FUNC_END not yet).
  int tool_call_outside_param_count_ = 0;
  // Whether we've seen at least one PARAM_START in the current tool call.
  // Tokens between FUNC_START and the first PARAM_START are not counted,
  // since that region is just the function name / preamble.
  bool tool_call_seen_parameter_ = false;
  // True when the token sampled this iteration came from EOG resampling.
  // Those tokens are spurious artifacts and should not count toward the
  // silent-loop detector.
  bool eog_recovered_this_token_ = false;
};

#endif // TOKEN_GENERATOR_H
