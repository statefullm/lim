#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include "llama.h"
#include <string>
#include <vector>
#include <map>
#include <cstdint>

// Fast file fingerprint (mtime:size) for cache validation without reading content.
std::string file_fingerprint(const std::string& path);

// Write a save file with a one-line text header containing the git SHA,
// followed by the raw llama state.  The header format is:
//     LIM_SAVE_v1 git_sha=<40-hex>\n<raw-state-bytes>
// Returns true on success.  Overwrites any existing file at save_path.
bool write_llm_save(const std::string& save_path, const uint8_t* state_data, size_t state_size);

// Read the header from an LLLM save file.
// On success returns the git SHA and sets *header_size to the byte count of
// the header line (including trailing newline).  The raw llama state begins
// at offset *header_size in the file.  Returns "" for unrecognized / old-style
// files (callers should fall back to llama_state_load_file directly).
std::string read_llm_save_header(const std::string& save_path, size_t* header_size);

// --- Compact token-based save/restore ---
// Save format: LIM_SAVE_V3 git_sha=<40-hex> n_tokens=<N> n_checkpoints=<M>\n<token_ids_as_int32_le><checkpoint_entries>
// Each checkpoint entry: <n_past as int32_le><prompt_len as uint16_le><prompt_bytes>
bool read_token_save(const std::string& save_path, std::vector<llama_token>& tokens);

struct PromptCheckpoint {
    int n_past;
    std::string prompt;
};

// V3 variant: includes prompt-return checkpoints for partial restore.
bool write_token_save_v3(const std::string& save_path, const std::vector<llama_token>& tokens,
                         const std::vector<PromptCheckpoint>& checkpoints);
bool read_token_save(const std::string& save_path, std::vector<llama_token>& tokens);
// Returns the prompt-return checkpoints embedded in a V3 save file.
std::vector<PromptCheckpoint> read_checkpoint_offsets(const std::string& save_path);

// V1 cache: after a slow restore, auto-save the rebuilt KV cache for instant
// future restores.  Cache lives in <cwd>/.cache/<name>-<hash>.
// The hash is content-based (SHA-256 of token data + model filename), so it
// survives save file renames and moves.  The name suffix is purely informational.
static constexpr const char* SAVE_EXT = ".save";

bool try_load_v1_cache(const std::string& v2_path, const std::vector<llama_token>& tokens,
                       const std::string& model_path, struct llama_context* ctx);
// Compute the content-based cache hash for a token sequence and model.
std::string cache_hash_for_save(const std::vector<llama_token>& tokens,
                                const std::string& model_path);
// After a successful restore or named save, persist the current KV cache.
// old_hash: if non-empty, the cache entry matching this hash is deleted first
// (it corresponds to the previous content of the save file).
// Returns true on success (or if an equivalent cache entry already exists).
bool write_v1_cache(const std::string& v2_path, const std::vector<llama_token>& tokens,
                    const std::string& model_path, struct llama_context* ctx,
                    const std::string& old_hash = "");
std::string get_cache_dir();
#include <functional>

class FileSystemTools {
public:
  FileSystemTools();

  // Execute a shell command. Callbacks fire as output arrives, allowing
  // real-time streaming. on_open is called once before any output, on_chunk
  // per read(), and on_close after the command finishes (with the full result).
  std::string exec_shell(const std::string& command,
                         std::function<void()> on_open = nullptr,
                         std::function<void(const std::string&)> on_chunk = nullptr,
                         std::function<void(const std::string&)> on_close = nullptr);

  std::vector<std::map<std::string, std::string>> read_files(const std::vector<std::string>& paths);

  std::map<std::string, std::string> write_file(const std::string& path, const std::string& content);

  std::map<std::string, std::string> edit_file(const std::string& path, const std::string& old_str, const std::string& new_str);

  // Search for text with optional line range.
  // Returns map with keys: "content" (LLM-facing result), "match_count",
  // "actual_start", "actual_end" (for line-range mode), "error" (non-empty on failure).
  std::map<std::string, std::string> search_file(const std::string& path, const std::string& text, const std::string& begin_str = "", const std::string& end_str = "");

private:
  std::string _get_fullpath(const std::string& path);
};

// --- Consolidated Diagnostic Logging Function ---
// This function handles all diagnostic output to both session and log file
// Parameters:
//   - message: The diagnostic message to output
//   - logOnly: If true, only write to log file (not to session)
//   - debugOnly: If true, only output when LIM_DEBUG=1 environment variable is set
//   - tag: Optional tag to prepend to the message (e.g., "[Edit]")
void log_diagnostic(const std::string& message, bool logOnly = false, bool debugOnly = false,
                    const std::string& tag = "");

// Styled wrapper: outputs tool call diagnostics with .tool-label colors in the browser.
// Writes to chat_log, styled HTML to browser pipe, and plain text to stdout.
void log_tool_diagnostic(const std::string& message, bool debugOnly = false,
                         const std::string& tag = "");



// Global debug flag - declared in main.cc and used by other modules
extern bool is_debug;

// exec_shell truncation limit (in bytes) - declared in main.cc, configurable via LIM_EXEC_TRUNCATION
extern size_t exec_truncation_limit;

// Chat log file stream - declared in main.cc and used for all tool diagnostics
extern std::ofstream chat_log;

// Output mode control functions - declared in output.h
#include "output.h"

#endif // FILESYSTEM_H
