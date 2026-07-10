#include "model.h"
#include "common.h"
#include "chat.h"
#include "filesystem.h"
#include "parsers.h"
#include "tokens.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <cstring>

using namespace std;

// Global model tokens (declared extern in model.h)
ModelTokens g_model_tokens;

// --- llama.cpp internal symbols, exported from libllama.so ---
// These are not in the public llama.h header but are compiled into the shared library.
// We declare them to avoid duplicating llama.cpp's own detection/template logic.

// enum llm_chat_template is defined in llama/src/llama-chat.h;
// we mirror the values we need here (must stay in sync with upstream).
enum llm_chat_template {
  LLM_CHAT_TEMPLATE_UNKNOWN = 0,
  LLM_CHAT_TEMPLATE_CHATML,
  LLM_CHAT_TEMPLATE_LLAMA_2,
  LLM_CHAT_TEMPLATE_LLAMA_2_SYS,
  LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,
  LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,
  LLM_CHAT_TEMPLATE_MISTRAL_V1,
  LLM_CHAT_TEMPLATE_MISTRAL_V3,
  LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN,
  LLM_CHAT_TEMPLATE_MISTRAL_V7,
  LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN,
  LLM_CHAT_TEMPLATE_PHI_3,
  LLM_CHAT_TEMPLATE_PHI_4,
  LLM_CHAT_TEMPLATE_FALCON_3,
  LLM_CHAT_TEMPLATE_ZEPHYR,
  LLM_CHAT_TEMPLATE_MONARCH,
  LLM_CHAT_TEMPLATE_GEMMA,
  LLM_CHAT_TEMPLATE_ORION,
  LLM_CHAT_TEMPLATE_OPENCHAT,
  LLM_CHAT_TEMPLATE_VICUNA,
  LLM_CHAT_TEMPLATE_VICUNA_ORCA,
  LLM_CHAT_TEMPLATE_DEEPSEEK,
  LLM_CHAT_TEMPLATE_DEEPSEEK_2,
  LLM_CHAT_TEMPLATE_DEEPSEEK_3,
  LLM_CHAT_TEMPLATE_DEEPSEEK_OCR,
  LLM_CHAT_TEMPLATE_COMMAND_R,
  LLM_CHAT_TEMPLATE_LLAMA_3,
  LLM_CHAT_TEMPLATE_CHATGLM_3,
  LLM_CHAT_TEMPLATE_CHATGLM_4,
  LLM_CHAT_TEMPLATE_GLMEDGE,
  LLM_CHAT_TEMPLATE_MINICPM,
  LLM_CHAT_TEMPLATE_EXAONE_3,
  LLM_CHAT_TEMPLATE_EXAONE_4,
  LLM_CHAT_TEMPLATE_EXAONE_MOE,
  LLM_CHAT_TEMPLATE_RWKV_WORLD,
  LLM_CHAT_TEMPLATE_GRANITE_3_X,
  LLM_CHAT_TEMPLATE_GRANITE_4_0,
  LLM_CHAT_TEMPLATE_GRANITE_4_1,
  LLM_CHAT_TEMPLATE_GIGACHAT,
  LLM_CHAT_TEMPLATE_MEGREZ,
  LLM_CHAT_TEMPLATE_YANDEX,
  LLM_CHAT_TEMPLATE_BAILING,
  LLM_CHAT_TEMPLATE_BAILING_THINK,
  LLM_CHAT_TEMPLATE_BAILING2,
  LLM_CHAT_TEMPLATE_LLAMA4,
  LLM_CHAT_TEMPLATE_SMOLVLM,
  LLM_CHAT_TEMPLATE_DOTS1,
  LLM_CHAT_TEMPLATE_HUNYUAN_MOE,
  LLM_CHAT_TEMPLATE_OPENAI_MOE,
  LLM_CHAT_TEMPLATE_HUNYUAN_DENSE,
  LLM_CHAT_TEMPLATE_HUNYUAN_VL,
  LLM_CHAT_TEMPLATE_KIMI_K2,
  LLM_CHAT_TEMPLATE_SEED_OSS,
  LLM_CHAT_TEMPLATE_GROK_2,
  LLM_CHAT_TEMPLATE_PANGU_EMBED,
  LLM_CHAT_TEMPLATE_SOLAR_OPEN,
};

// llama.cpp's own template detector: takes a Jinja template string (or built-in name)
// and returns the matching llm_chat_template enum.
llm_chat_template llm_chat_detect_template(const std::string &tmpl);


// --- Model Detection and Chat Template Selection ---

// Map llama.cpp's internal enum to our ModelType.
static ModelType map_llm_template(llm_chat_template tmpl) {
  switch (tmpl) {
    case LLM_CHAT_TEMPLATE_CHATML:            return ModelType::CHATML;
    case LLM_CHAT_TEMPLATE_LLAMA_3:           return ModelType::LLAMA3;
    case LLM_CHAT_TEMPLATE_PHI_3:             return ModelType::PHI_3;
    case LLM_CHAT_TEMPLATE_PHI_4:             return ModelType::PHI_4;
    case LLM_CHAT_TEMPLATE_MISTRAL_V7:
    case LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN: return ModelType::MISTRAL_V7;
    case LLM_CHAT_TEMPLATE_MISTRAL_V1:        return ModelType::MISTRAL_V1;
    case LLM_CHAT_TEMPLATE_MISTRAL_V3:
    case LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN: return ModelType::MISTRAL_V3;
    case LLM_CHAT_TEMPLATE_GEMMA:             return ModelType::GEMMA;
    case LLM_CHAT_TEMPLATE_ZEPHYR:            return ModelType::ZEPHYR;
    case LLM_CHAT_TEMPLATE_VICUNA:
    case LLM_CHAT_TEMPLATE_VICUNA_ORCA:       return ModelType::VICUNA;
    case LLM_CHAT_TEMPLATE_DEEPSEEK_3:        return ModelType::DEEPSEEK_3;
    case LLM_CHAT_TEMPLATE_GRANITE_3_X:
    case LLM_CHAT_TEMPLATE_GRANITE_4_0:
    case LLM_CHAT_TEMPLATE_GRANITE_4_1:       return ModelType::GRANITE;
    case LLM_CHAT_TEMPLATE_LLAMA_2:
    case LLM_CHAT_TEMPLATE_LLAMA_2_SYS:
    case LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS:
    case LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP: return ModelType::LLAMA2;
    default:                                  return ModelType::UNKNOWN;
  }
}

// Map our ModelType back to the built-in template name string that
// llama_chat_apply_template() accepts.
string get_chat_template_name(ModelType model_type) {
  switch (model_type) {
    case ModelType::CHATML:     return "chatml";
    case ModelType::LLAMA3:     return "llama3";
    case ModelType::PHI_3:      return "phi3";
    case ModelType::PHI_4:      return "phi4";
    case ModelType::MISTRAL_V7: return "mistral-v7";
    case ModelType::MISTRAL_V1: return "mistral-v1";
    case ModelType::MISTRAL_V3: return "mistral-v3";
    case ModelType::GEMMA:      return "gemma";
    case ModelType::ZEPHYR:     return "zephyr";
    case ModelType::VICUNA:     return "vicuna";
    case ModelType::DEEPSEEK_3: return "deepseek3";
    case ModelType::GRANITE:    return "granite";
    case ModelType::LLAMA2:     return "llama2";
    default:                    return "chatml";  // fallback
  }
}

// Detect model type from GGUF metadata.
// Strategy: read the Jinja chat template stored in the GGUF file, then pass it
// to llama.cpp's own llm_chat_detect_template() which pattern-matches against
// all known families. This is exactly what llama-cli does internally.
ModelType detect_model_type(const llama_model *model) {
  const char *tmpl_str = llama_model_chat_template(model, nullptr);
  if (!tmpl_str || tmpl_str[0] == '\0') {
    // No template metadata -- fall back to ChatML (most common default)
    return ModelType::CHATML;
  }

  string tmpl(tmpl_str);
  llm_chat_template detected = llm_chat_detect_template(tmpl);
  ModelType mt = map_llm_template(detected);

  if (mt == ModelType::UNKNOWN) {
    // Detection failed -- try to use the template string directly with
    // llama_chat_apply_template. If it's already a built-in name, that works.
    // Otherwise fall back to ChatML.
    const char *builtin_names[64];
    int n_builtins = llama_chat_builtin_templates(builtin_names, 64);
    for (int i = 0; i < n_builtins; i++) {
      if (tmpl == builtin_names[i]) {
        mt = map_llm_template(llm_chat_detect_template(tmpl));
        break;
      }
    }
    if (mt == ModelType::UNKNOWN) {
      mt = ModelType::CHATML;  // safe fallback
    }
  }

  return mt;
}


// --- Tokenization helpers ---

static vector<llama_token> tok(llama_context *ctx, const string &s) {
  if (s.empty()) return {};
  return common_tokenize(ctx, s, false, true);
}

// Trim a string from the front, returning the removed prefix.
// Useful for extracting delimiters by comparing template outputs.
static string strip_prefix(string &s, const string &prefix) {
  if (s.rfind(prefix, 0) == 0) {
    s.erase(0, prefix.size());
    return prefix;
  }
  return "";
}

// Trim a string from the back, returning the removed suffix.
static string strip_suffix(string &s, const string &suffix) {
  if (!suffix.empty() && s.size() >= suffix.size() &&
      s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0) {
    string removed = s.substr(s.size() - suffix.size());
    s.resize(s.size() - suffix.size());
    return removed;
  }
  return "";
}


// Helper to escape control characters for debug logging
static string escape_for_log(const string &s) {
  string out;
  out.reserve(s.size() * 2);
  for (char c : s) {
    if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else out += c;
  }
  return out;
}


// --- init_model_tokens: ask llama.cpp for the correct tokens ---

void init_model_tokens(llama_context *ctx, const llama_model *model) {
  // Get the actual Jinja template string from GGUF metadata.
  // This is what the model itself says it uses -- more reliable than detection.
  const char *tmpl_str = llama_model_chat_template(model, nullptr);
  if (!tmpl_str || tmpl_str[0] == '\0') {
    tmpl_str = "chatml";  // safe fallback
  }

  // Detect the template family for logging and ModelType tracking.
  string tmpl(tmpl_str);
  llm_chat_template detected = llm_chat_detect_template(tmpl);
  g_model_tokens.type = map_llm_template(detected);

  char buf[8192];

  // Strategy: apply the template to carefully crafted test messages and diff
  // the outputs to isolate each delimiter fragment.

  // --- 1. Extract user_turn_start ---
  llama_chat_message msg_user = {"user", "_XTEST_X"};
  int32_t n = llama_chat_apply_template(tmpl_str, &msg_user, 1, false, buf, sizeof(buf));
  string user_no_ass(n > 0 ? buf : "");

  // Everything before our sentinel is the user turn start.
  string sentinel = "_XTEST_X";
  size_t cp = user_no_ass.find(sentinel);
  string user_turn_start_str = (cp != string::npos) ? user_no_ass.substr(0, cp) : "";

  // --- 2. Extract assistant_turn_start and turn_end ---
  // Apply with add_ass=true to get the suffix added after user content.
  n = llama_chat_apply_template(tmpl_str, &msg_user, 1, true, buf, sizeof(buf));
  string user_with_ass(n > 0 ? buf : "");

  // The tail is: everything after "_XTEST_X" in the add_ass=true output.
  string tail_after_content = "";
  if (cp != string::npos && user_with_ass.size() >= cp + sentinel.size()) {
    tail_after_content = user_with_ass.substr(cp + sentinel.size());
  }
  // tail_after_content = turn_end + assistant_turn_start

  // Now get just the assistant prefix by applying template to an assistant-only message.
  llama_chat_message msg_ass = {"assistant", "_A_"};
  n = llama_chat_apply_template(tmpl_str, &msg_ass, 1, false, buf, sizeof(buf));
  string ass_only(n > 0 ? buf : "");

  size_t ap = ass_only.find("_A_");
  string assistant_turn_start_str = (ap != string::npos) ? ass_only.substr(0, ap) : "";

  // turn_end = tail_after_content with assistant_turn_start stripped from the end.
  string turn_end_str;
  if (!assistant_turn_start_str.empty() &&
      tail_after_content.size() >= assistant_turn_start_str.size()) {
    size_t offset = tail_after_content.size() - assistant_turn_start_str.size();
    string candidate = tail_after_content.substr(offset);
    if (candidate == assistant_turn_start_str) {
      turn_end_str = tail_after_content.substr(0, offset);
    } else {
      // assistant_turn_start not at the end -- try finding it in the middle.
      size_t pos = tail_after_content.find(assistant_turn_start_str);
      if (pos != string::npos) {
        turn_end_str = tail_after_content.substr(0, pos);
      } else {
        // Fallback: entire tail is turn_end (no assistant prefix found).
        turn_end_str = tail_after_content;
      }
    }
  } else if (!tail_after_content.empty()) {
    // No assistant prefix detected -- entire tail is turn_end.
    turn_end_str = tail_after_content;
  }

  // --- 3. Extract system_turn_start ---
  llama_chat_message msg_sys = {"system", "_S_"};
  n = llama_chat_apply_template(tmpl_str, &msg_sys, 1, false, buf, sizeof(buf));
  string sys_only(n > 0 ? buf : "");

  size_t sp = sys_only.find("_S_");
  string system_turn_start_str = (sp != string::npos) ? sys_only.substr(0, sp) : user_turn_start_str;

  // --- Store results ---
  g_model_tokens.system_turn_start.text  = system_turn_start_str;
  g_model_tokens.system_turn_start.tokens = tok(ctx, system_turn_start_str);

  g_model_tokens.user_turn_start.text  = user_turn_start_str;
  g_model_tokens.user_turn_start.tokens = tok(ctx, user_turn_start_str);

  g_model_tokens.assistant_turn_start.text  = assistant_turn_start_str;
  g_model_tokens.assistant_turn_start.tokens = tok(ctx, assistant_turn_start_str);

  g_model_tokens.turn_end.text  = turn_end_str;
  g_model_tokens.turn_end.tokens = tok(ctx, turn_end_str);

  // Set thinking tags by querying the model's chat template.
  // common_chat_templates_apply populates thinking_start_tag and
  // thinking_end_tag from the Jinja template via its autoparser,
  // so we get the correct tags for any model without hardcoding.
  {
    auto tmpls = common_chat_templates_init(model, tmpl_str);
    if (tmpls) {
      common_chat_templates_inputs inputs;
      inputs.messages.push_back({{"user", "_X_"}});
      inputs.add_generation_prompt = true;
      inputs.use_jinja = true;

      try {
        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        if (!params.thinking_start_tag.empty()) {
          g_model_tokens.think_start = params.thinking_start_tag;
        }
        if (!params.thinking_end_tag.empty()) {
          g_model_tokens.think_end = params.thinking_end_tag;
        }
      } catch (...) {
        // Template application failed -- leave tags empty.
        // Thinking block detection will simply never match.
      }
    }
  }

  // Log what we detected (useful for debugging)
  if (is_debug) {
    cerr << "[model] Detected template: " << get_chat_template_name(g_model_tokens.type) << "\n";
    cerr << "[model] system_turn_start  = \"" << escape_for_log(system_turn_start_str) << "\"\n";
    cerr << "[model] user_turn_start    = \"" << escape_for_log(user_turn_start_str) << "\"\n";
    cerr << "[model] assistant_turn_start = \"" << escape_for_log(assistant_turn_start_str) << "\"\n";
    cerr << "[model] turn_end           = \"" << escape_for_log(turn_end_str) << "\"\n";
    cerr << "[model] think_start      = \"" << escape_for_log(g_model_tokens.think_start) << "\"\n";
    cerr << "[model] think_end        = \"" << escape_for_log(g_model_tokens.think_end) << "\"\n";
  }
}

// Generate the complete reserved-token escape contract for the system prompt.
// Covers PARAM_END (always active) plus model-specific turn delimiters.
string generate_turn_escape_contract() {
  // Helper: escape control characters for display in prompt.
  // Also breaks up known special token strings so the tokenizer doesn't
  // interpret them as structural boundaries when this contract is fed to the model.
  auto display = [](const string &t) -> string {
    string out;
    for (char c : t) {
      if (c == '\n') out += "\\n";
      else if (c == '\r') out += "\\r";
      else if (c == '\t') out += "\\t";
      else if (c < 32 && c > 0) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\x%02x", (unsigned char)c);
        out += buf;
      } else {
        out += c;
      }
    }
    // Break up special token strings so the tokenizer won't recognize them.
    // Insert a zero-width space (U+200B) after '<|' to prevent tokenization.
    string broken = out;
    size_t pos = 0;
    while ((pos = broken.find("<|", pos)) != string::npos) {
      broken.insert(pos + 2, "\xe2\x80\x8b"); // UTF-8 for U+200B
      pos += 4;
    }
    return broken;
  };

  // Helper: produce escaped form of a token (one '\' after first char).
  auto esc = [](const string &t) -> string {
    if (t.size() < 2) return t;
    return t.substr(0, 1) + "\\" + t.substr(1);
  };

  // Helper: produce double-escaped form.
  auto esc2 = [&esc](const string &t) -> string {
    return esc(esc(t));
  };

  // Collect all active reserved tokens with labels.
  struct TokenDesc { string label; string text; };
  vector<TokenDesc> tokens;

  // PARAM_END is always active.
  tokens.push_back(TokenDesc{"parameter closing tag", Tokens::PARAM_END});

  // Extract base turn tokens from model markers (strip role labels and newlines).
  auto add_base_tokens = [&](const string &marker) {
    size_t pos = 0;
    while ((pos = marker.find("<|", pos)) != string::npos) {
      size_t end = marker.find("|>", pos + 2);
      if (end != string::npos) {
        string token = marker.substr(pos, end - pos + 2);
        bool seen = false;
        for (auto &td : tokens) { if (td.text == token) { seen = true; break; } }
        if (!seen) {
          tokens.push_back(TokenDesc{"turn token", token});
        }
        pos = end + 2;
      } else break;
    }
  };

  if (!g_model_tokens.user_turn_start.text.empty()) add_base_tokens(g_model_tokens.user_turn_start.text);
  if (!g_model_tokens.assistant_turn_start.text.empty()) add_base_tokens(g_model_tokens.assistant_turn_start.text);
  if (!g_model_tokens.system_turn_start.text.empty()) add_base_tokens(g_model_tokens.system_turn_start.text);
  if (!g_model_tokens.turn_end.text.empty()) add_base_tokens(g_model_tokens.turn_end.text);

  string contract = "RESERVED TOKEN ESCAPING CONTRACT:\n";
  contract += "The following strings are reserved structural markers and can never\n";
  contract += "appear literally in tool calls or tool responses. When they appear\n";
  contract += "in file content, they are escaped with one additional backslash after\n";
  contract += "the first character.\n\n";

  for (auto &td : tokens) {
    contract += "- " + display(td.text) + " (" + td.label + ")\n";
  }

  contract += "\n1. SYSTEM -> LLM: Any reserved tokens in file content arrive escaped\n";
  contract += "   (one extra backslash after the first character). You will NEVER see\n";
  contract += "   unescaped reserved tokens in tool responses.\n\n";
  contract += "2. LLM -> SYSTEM: If you want to write text containing a reserved token,\n";
  contract += "   use the escaped form with one extra backslash after the first character.\n";
  contract += "   The system removes that backslash before writing to disk; any other\n";
  contract += "   backslashes are preserved as-is.\n\n";
  contract += "Examples (on disk -> in response):\n";

  for (auto &td : tokens) {
    contract += "  " + display(td.text) + " -> " + display(esc(td.text)) + "\n";
    contract += "  " + display(esc(td.text)) + " -> " + display(esc2(td.text)) + "\n";
  }

  contract += "\nActionable shortcut: whatever form you see in a tool response, use that exact\n";
  contract += "same text when writing it back. The system handles both directions symmetrically.\n";

  return contract;
}

vector<llama_token> build_system_prompt_tokens(llama_context *ctx, const string &content) {
  // Build full string then tokenize in one pass with add_bos=true.
  // Critical: some models (e.g., Qwen3.6) share BOS and <|im_start|> token IDs,
  // so manually prepending BOS would produce a duplicate. common_tokenize(add_bos=true)
  // handles this correctly by letting the tokenizer decide.
  string msg = g_model_tokens.system_turn_start.text + content;
  if (g_model_tokens.has_explicit_turn_end())
    msg += g_model_tokens.turn_end.text;
  return common_tokenize(ctx, msg, true, true);
}

vector<llama_token> build_user_assistant_turn(llama_context *ctx, const string &user_content) {
  string msg = build_user_assistant_turn_text(user_content);
  return common_tokenize(ctx, msg, false, true);
}

string build_user_assistant_turn_text(const string &user_content) {
  string msg = g_model_tokens.user_turn_start.text + user_content;
  if (g_model_tokens.has_explicit_turn_end())
    msg += g_model_tokens.turn_end.text;
  msg += g_model_tokens.assistant_turn_start.text;
  return msg;
}

vector<llama_token> build_user_turn_only(llama_context *ctx, const string &user_content) {
  string msg = g_model_tokens.user_turn_start.text + user_content;
  if (g_model_tokens.has_explicit_turn_end())
    msg += g_model_tokens.turn_end.text;
  return common_tokenize(ctx, msg, false, true);
}

vector<llama_token> build_tool_result_turn(llama_context *ctx, const string &tool_output) {
  // Escape model turn tokens in the tool output so they don't get
  // misinterpreted as structural boundaries during tokenization.
  string escaped_output = tool_output;
  escape_turn_tags(escaped_output);

  string msg = g_model_tokens.user_turn_start.text + "[Tool Result]\n" + escaped_output;
  if (g_model_tokens.has_explicit_turn_end())
    msg += g_model_tokens.turn_end.text;
  msg += g_model_tokens.assistant_turn_start.text;
  return common_tokenize(ctx, msg, false, true);
}

vector<llama_token> build_forced_close_tokens(llama_context *ctx) {
  vector<llama_token> result;

  // Leading newline
  auto nl = tok(ctx, "\n");
  result.insert(result.end(), nl.begin(), nl.end());

  // turn_end
  if (g_model_tokens.has_explicit_turn_end()) {
    result.insert(result.end(), g_model_tokens.turn_end.tokens.begin(),
                  g_model_tokens.turn_end.tokens.end());
  }

  // Trailing newline
  nl = tok(ctx, "\n");
  result.insert(result.end(), nl.begin(), nl.end());

  return result;
}


// --- Decode Error Handling ---

// Forward declaration for diagnostic helper defined in main.cc
extern void diag(const string& msg, const char* color);

bool handle_llama_decode_error(llama_context *ctx, llama_batch batch, const char* error_msg, bool should_break) {
  int ret = llama_decode(ctx, batch);
  if (ret < -1) {
    diag(error_msg, "\033[31m");
    return false;
  } else if (ret == -1) {
    diag("Invalid input batch: " + std::string(error_msg), "\033[31m");
    return false;
  } else if (ret == 1 || ret == 2) {
    diag(ret == 1 ? error_msg : "Aborted", "\033[31m");
    if (should_break) return false;
    return true;
  }
  return true;
}

void sync_n_past(llama_context *ctx, int &n_past) {
  llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
  if (max_pos >= 0) {
    n_past = (int)(max_pos + 1);
  }
}

// --- Log Callbacks ---

void dummy_log_callback(enum ggml_log_level level, const char * text, void * user_data) {}

void custom_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
  cerr << text;
}

