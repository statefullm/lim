#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>
#include <vector>
#include <map>

// Fast file fingerprint (mtime:size) for cache validation without reading content.
std::string file_fingerprint(const std::string& path);
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
  static const std::string HOME;
};

// --- Consolidated Diagnostic Logging Function ---
// This function handles all diagnostic output to both session and log file
// Parameters:
//   - message: The diagnostic message to output
//   - logOnly: If true, only write to log file (not to session)
//   - debugOnly: If true, only output when LLLM_DEBUG=1 environment variable is set
//   - tag: Optional tag to prepend to the message (e.g., "[Edit]")
void log_diagnostic(const std::string& message, bool logOnly = false, bool debugOnly = false,
                    const std::string& tag = "");

// Styled wrapper: outputs tool call diagnostics with .tool-label colors in the browser.
// Writes to chat_log, styled HTML to browser pipe, and plain text to stdout.
void log_tool_diagnostic(const std::string& message, bool debugOnly = false,
                         const std::string& tag = "");



// Global debug flag - declared in main.cc and used by other modules
extern bool is_debug;

// Chat log file stream - declared in main.cc and used for all tool diagnostics
extern std::ofstream chat_log;

// Output mode control functions - declared in output.h
#include "output.h"

#endif // FILESYSTEM_H
