#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>
#include <vector>
#include <map>

class FileSystemTools {
public:
  FileSystemTools();

  std::string exec_shell(const std::string& command);

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
