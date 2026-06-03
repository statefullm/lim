#include "filesystem.h"
#include "output.h"
#include "parsers.h"
#include "tokens.h"
#include "network.h"  // For process_pdf_with_docling and base64_encode
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pwd.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits.h>

using namespace std;
using namespace Tokens;

// Wrapper: outputs tool diagnostics with .tool-label styling in the browser.
// Writes to chat_log, styled HTML to browser pipe, and plain text to stdout.
void log_tool_diagnostic(const string& message, bool debugOnly /* = false */,
                         const string& tag /* = "" */) {
    if (!chat_log.is_open()) return;

    string final_message = tag.empty() ? message : tag + " " + message;

    if (!debugOnly || is_debug) {
        // Write to chat_log
        chat_log << final_message << "\n";
        chat_log.flush();

        // Styled HTML to browser pipe (uses .tool-label colors from viewer.html)
        if (should_output_to_browser()) {
            if (pipe_fd < 0) {
                const char* FIFO_PATH = "/tmp/lllm.fifo";
                pipe_fd = open(FIFO_PATH, O_RDWR | O_NONBLOCK);
            }
            if (pipe_fd >= 0) {
                string safe;
                for (char c : final_message) {
                    if (c == '&') safe += "&amp;";
                    else if (c == '<') safe += "&lt;";
                    else if (c == '>') safe += "&gt;";
                    else safe += c;
                }
                string html = "<div class='tool-label'>" + safe + "</div>";
                pipe_write(&SEG_HTML, 1);
                pipe_write(html.c_str(), html.length());
            }
        }

        // Plain text to stdout
        if (should_output_to_stdout()) {
            cout << final_message << endl;
            fflush(stdout);
        }
    }
}

// --- Consolidated Diagnostic Logging Function ---
void log_diagnostic(const string& message, bool logOnly /* = false */, bool debugOnly /* = false */,
                    const string& tag /* = "" */) {
    if (!chat_log.is_open()) {
        cerr << "[ERROR] log file is not open; cannot write diagnostic message\n";
        return;
    }

    // Prepend tag if provided
    string final_message = tag.empty() ? message : tag + " " + message;

    if (!logOnly) {
        if (!debugOnly || is_debug) {
            if (should_output_to_browser()) {
                // Output to browser via FIFO pipe
                if (pipe_fd < 0) {
                    // Try to initialize the pipe if not already done
                    const char* FIFO_PATH = "/tmp/lllm.fifo";
                    pipe_fd = open(FIFO_PATH, O_RDWR | O_NONBLOCK);
                }
                if (pipe_fd >= 0) {
                  string browser_msg = final_message + "\n";
                    ssize_t res = write(pipe_fd, browser_msg.c_str(), browser_msg.length());

                    // If write fails with EAGAIN or EWOULDBLOCK, the pipe buffer is full
                    if (res < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                        close(pipe_fd);
                        pipe_fd = -1;
                    }
                }
            }
            if (should_output_to_stdout()) {
                // Output to stdout (when browser mode is off, or in combined mode 3)
                cout << final_message << endl;
                fflush(stdout);
            }
        }
    }

    chat_log << final_message << "\n";
    chat_log.flush();
}

// --- Helper to escape tags from disk so they don't break the LLM's XML parser ---
// Recursive backslash scheme: for <\//parameter> with N backslashes between < and /,
// produce <\//parameter> with N+1 backslashes sent to the LLM.
static void escape_parameter_tags(std::string& str) {
    size_t start_pos = 0;
    while (start_pos < str.length()) {
        size_t lt_pos = str.find('<', start_pos);
        if (lt_pos == string::npos) break;

        size_t scan = lt_pos + 1;
        int num_backslashes = 0;
        while (scan < str.length() && str[scan] == '\\') {
            num_backslashes++;
            scan++;
        }

        string suffix = "/parameter>";
        bool match = true;
        for (size_t k = 0; k < suffix.length(); k++) {
            if (scan + k >= str.length() || str[scan + k] != suffix[k]) {
                match = false;
                break;
            }
        }

        if (match) {
            size_t token_start = lt_pos;
            size_t token_end = scan + suffix.length();

            // Replace with N+1 backslashes
            string replacement = "<";
            for (int b = 0; b <= num_backslashes; b++) {
                replacement += '\\';
            }
            replacement += "/parameter>";

            str.replace(token_start, token_end - token_start, replacement);
            start_pos = token_start + replacement.length();
        } else {
            start_pos = lt_pos + 1;
        }
    }
}

// --- Helper to unescape tags from LLM input before writing to disk ---
// Recursive backslash scheme: for <\//parameter> with N backslashes between < and /,
// produce <\//parameter> with N-1 backslashes on disk (inverse of escape).
static void unescape_parameter_tags(std::string& str) {
    size_t start_pos = 0;
    while (start_pos < str.length()) {
        size_t lt_pos = str.find('<', start_pos);
        if (lt_pos == string::npos) break;

        size_t scan = lt_pos + 1;
        int num_backslashes = 0;
        while (scan < str.length() && str[scan] == '\\') {
            num_backslashes++;
            scan++;
        }

        string suffix = "/parameter>";
        bool match = true;
        for (size_t k = 0; k < suffix.length(); k++) {
            if (scan + k >= str.length() || str[scan + k] != suffix[k]) {
                match = false;
                break;
            }
        }

        if (match) {
            size_t token_start = lt_pos;
            size_t token_end = scan + suffix.length();

            // Replace with N-1 backslashes (minimum 0)
            string replacement = "<";
            int new_bs = (num_backslashes > 0) ? (num_backslashes - 1) : 0;
            for (int b = 0; b < new_bs; b++) {
                replacement += '\\';
            }
            replacement += "/parameter>";

            str.replace(token_start, token_end - token_start, replacement);
            start_pos = token_start + replacement.length();
        } else {
            start_pos = lt_pos + 1;
        }
    }
}

const string FileSystemTools::HOME = "/home/ai";

// Test edit - filesystem tools verification
FileSystemTools::FileSystemTools() {}

string FileSystemTools::_get_fullpath(const string& path) {
  string home_cwd = HOME + "/.cwd";
  ifstream cwd_file(home_cwd);
  string cwd_str;

  if (cwd_file.is_open()) {
    getline(cwd_file, cwd_str);
    cwd_file.close();
  } else {
    cwd_str = HOME;
  }

  string fullpath;
  if (path.empty() || path[0] != '/') {
    fullpath = cwd_str + "/" + path;
  } else {
    fullpath = path;
  }

  char result[PATH_MAX];
  if (realpath(fullpath.c_str(), result) != nullptr) {
    return string(result);
  }
  return fullpath;
}

string FileSystemTools::exec_shell(const string& command) {
  // Build human-readable function call syntax (truncate for display)
  string cmd_str = "\"" + (command.length() > 80 ? command.substr(0, 77) + "..." : command) + "\"";

  // Output function call to both stdout and logfile
  log_tool_diagnostic("exec_shell(" + cmd_str + ")");

  string safe_cmd = command;
  size_t pos = 0;
  while ((pos = safe_cmd.find("'", pos)) != string::npos) {
    safe_cmd.replace(pos, 1, "'\\''");
    pos += 4;
  }

  // Capture exit code in a temp file to avoid collision with command output.
  // Mixing exit-code markers into stdout/stderr breaks when commands produce
  // text containing the marker (e.g., nested bash -c wrappers).
  string exit_code_file = "/tmp/lllm_exitcode_XXXXXX";
  int fd = mkstemp(const_cast<char*>(exit_code_file.c_str()));
  if (fd >= 0) close(fd); // Close so bash can write to it

  string full_command = "bash -c 'cd \"$(cat ~/.cwd 2>/dev/null || echo " + HOME + ")\" && " + safe_cmd + " ; echo \"$?\" > \"" + exit_code_file + "\"' 2>&1";

  string result;
  FILE* pipe = popen(full_command.c_str(), "r");
  if (!pipe) {
    unlink(exit_code_file.c_str());
    return "Error: Failed to spawn shell process. The exec_shell tool may be temporarily unavailable.";
  }

  char buffer[1024];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }

  int pipe_status = pclose(pipe);

  // Read exit code from temp file -- guaranteed to be the actual command exit code.
  int exit_code = -1;
  {
    ifstream ec_in(exit_code_file.c_str());
    string ec_str;
    if (ec_in >> ec_str) {
      exit_code = atoi(ec_str.c_str());
    }
  }
  unlink(exit_code_file.c_str());

  // Fallback: if we couldn't read the temp file, try pipe status
  if (exit_code == -1 && WIFEXITED(pipe_status)) {
    exit_code = WEXITSTATUS(pipe_status);
  }

  if (!result.empty()) {
    chat_log << result << "\n\n";
    chat_log.flush();
  } else {
    chat_log << "[Command executed with no output]\n\n";
    chat_log.flush();
  }

  // Build the final result with exit code information for better LLM diagnostics
  if (result.empty()) {
    if (exit_code == 0) {
      result = "[Command exited with code 0 and produced no output]\n";
    } else if (exit_code > 0) {
      result = "[Command exited with code " + to_string(exit_code) + " and produced no output. The command may have failed silently -- check your syntax or try a different approach.]\n";
    } else {
      result = "[Command produced no output; exit code could not be determined]\n";
    }
  } else {
    // Always include exit code so the LLM can distinguish success from failure
    if (exit_code > 0) {
      result = "[Exit code: " + to_string(exit_code) + "]\n" + result;
    }
  }
  return result;
}

string FileSystemTools::search_file(const string& path, const string& text, int begin_line, int end_line) {
  // Build human-readable function call syntax (truncate for display)
  string path_str = "\"" + (path.length() > 50 ? path.substr(0, 47) + "..." : path) + "\"";

  // Build the function call label (logged at end with match count for text searches)
  string search_label = "search_file(" + path_str;
  if (begin_line > 0 && end_line >= begin_line) {
    search_label += ", lines " + to_string(begin_line) + "-" + to_string(end_line);
  }
  search_label += ")";

  // If LLLM_DEBUG=1, also output full text (truncated in stdout only)
  if (is_debug) {
    string text_str = "\"" + (text.length() > 80 ? text.substr(0, 77) + "..." : text) + "\"";
    log_diagnostic("TEXT: " + text_str);
  }

  // Always output full text to logfile (no truncation)
  log_diagnostic("TEXT: \"" + text + "\"", true /* logOnly */);

  // Unescape search text (LLM provides escaped content per prompt instructions)
  string unescaped_text = text;
  unescape_parameter_tags(unescaped_text);

  string fullpath = _get_fullpath(path);
  ifstream in_file(fullpath);
  if (!in_file.is_open()) {
    return "Error: Failed to open file for reading: " + fullpath;
  }

  stringstream buffer;
  buffer << in_file.rdbuf();
  string content = buffer.str();
  in_file.close();

  // Line range reading mode: if both begin and end are provided, return those lines directly
  if (begin_line > 0 && end_line >= begin_line) {
    vector<string> lines;
    string line;
    stringstream ss(content);
    while (getline(ss, line)) {
      lines.push_back(line);
    }

    // Adjust to 0-based indexing and clamp to valid range
    int start = max(0, begin_line - 1);
    int end = min((int)lines.size() - 1, end_line - 1);

    if (start >= (int)lines.size()) {
      return "Error: Line " + to_string(begin_line) + " is beyond the end of file (" + to_string(lines.size()) + " lines).";
    }

    string result = "Lines " + to_string(start + 1) + "-" + to_string(end + 1) + " of " + path + ":\n";
    for (int i = start; i <= end; i++) {
      result += lines[i] + "\n";
    }

    log_tool_diagnostic(search_label);
    escape_parameter_tags(result);
    return result;
  }

  bool search_with_newlines = (unescaped_text.find('\n') != string::npos);
  string result = "";
  int match_count = 0;
  int context = 5;

  if (search_with_newlines) {
    size_t pos = 0;
    while ((pos = content.find(unescaped_text, pos)) != string::npos) {
      match_count++;
      if (match_count > 10) {
          result += "... (Truncated after 10 matches)\n";
          break;
      }

      size_t start_pos = 0;
      int start_line = 1, end_line = 1;
      for (size_t i = 0; i < pos; i++) {
        if (content[i] == '\n') start_line++;
      }

      size_t end_pos = pos + unescaped_text.length();
      for (size_t i = 0; i < end_pos && i < content.length(); i++) {
        if (content[i] == '\n') end_line++;
      }

      result += "--- Match " + to_string(match_count) + " (Lines " + to_string(start_line) + "-" + to_string(end_line) + ") ---\n";

      size_t ctx_start = pos;
      size_t ctx_end = end_pos;

      int lines_before = 0;
      while (ctx_start > 0 && lines_before < context) {
        ctx_start--;
        if (content[ctx_start] == '\n') lines_before++;
      }

      int lines_after = 0;
      while (ctx_end < content.length() && lines_after < context) {
        if (content[ctx_end] == '\n') lines_after++;
        ctx_end++;
      }

      result += content.substr(ctx_start, ctx_end - ctx_start);
      pos = end_pos;
    }
  } else {
    vector<string> lines;
    string line;
    ifstream line_file(fullpath);
    if (!line_file.is_open()) {
      return "Error: Failed to reopen file for line-by-line reading: " + fullpath;
    }
    while (getline(line_file, line)) {
      lines.push_back(line);
    }
    line_file.close();

    int i = 0;
    while (i < (int)lines.size()) {
      if (lines[i].find(unescaped_text) != string::npos) {
        match_count++;
        if (match_count > 10) {
          result += "... (Truncated after 10 matches)\n";
          break;
        }
        int start = max(0, i - context);
        int end = min((int)lines.size() - 1, i + context);
        result += "--- Match " + to_string(match_count) + " (Lines " + to_string(start + 1) + "-" + to_string(end + 1) + ") ---\n";
        for (int j = start; j <= end; j++) {
          result += lines[j] + "\n";
        }
        result += "\n";
        i = end;
      }
      i++;
    }
  }

  if (match_count == 0) {
    log_tool_diagnostic(search_label + ": No occurrences found");
    return "No occurrences found for text.";
  }

  // Log function call with match count on one line
  log_tool_diagnostic(search_label + ": " + to_string(match_count) + " match(es)");

  escape_parameter_tags(result); // Escape any literal XML tags before sending to LLM
  return result;
}

vector<map<string, string>> FileSystemTools::read_files(const vector<string>& paths) {
  // Output function call to both stdout and logfile for each file
  for (const auto& path : paths) {
    string path_str = "\"" + (path.length() > 50 ? path.substr(0, 47) + "..." : path) + "\"";
    log_tool_diagnostic("read_file(" + path_str + ")");
  }

  vector<map<string, string>> results;

  // Check if any file is a PDF to determine if we need Docling
  bool needs_pdf_processing = false;
  for (const auto& path : paths) {
    string ext = path;
    size_t last_dot = path.rfind('.');
    if (last_dot != string::npos) {
      ext = path.substr(last_dot);
      transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
    }
    if (ext == ".pdf" || ext == ".PDF") {
      needs_pdf_processing = true;
      break;
    }
  }

  // Only start Docling if we have PDF files to process
  NetworkTools net;
  if (needs_pdf_processing) {
    net.start_docling_if_needed();
  }

  for (const auto& path : paths) {
    map<string, string> result;
    result["path"] = path;
    result["content"] = "";
    result["error"] = "";

    // Check if this is a URL
    bool is_url = (path.find("http://") == 0 || path.find("https://") == 0);

    // Check if this is a PDF file
    string ext = path;
    size_t last_dot = path.rfind('.');
    if (last_dot != string::npos) {
      ext = path.substr(last_dot);
      transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
    }

    bool is_pdf = (ext == ".pdf" || ext == ".PDF");

    if (is_pdf) {
      // Unified PDF handling - local files read directly, URLs use NetworkTools
      cerr << "Processing PDF: " + path << endl;

      bool is_url = (path.find("http://") == 0 || path.find("https://") == 0);

      if (is_url) {
        // Remote URL - use NetworkTools fetch mechanism
        NetworkTools net;
        vector<string> url_list = {path};
        vector<map<string, string>> network_results = net.fetch_urls(url_list);

        if (!network_results.empty()) {
          result["content"] = network_results[0]["content"];
          result["error"] = network_results[0]["error"];

          if (result["error"].empty() && !result["content"].empty()) {
            cerr << "Successfully processed PDF: " + path + " (" + to_string(result["content"].length()) + " bytes)" << endl;
          } else {
            cerr << "PDF processing failed: " + path << endl;
          }
        } else {
          result["error"] = "[No results from network fetch]";
          cerr << "Network fetch returned empty results" << endl;
        }
      } else {
        // Local PDF file - read directly and process with Docling
        ifstream in_file(_get_fullpath(path));
        if (!in_file.is_open()) {
          result["error"] = "Failed to open local PDF file for reading: " + path;
          cerr << "Error: Failed to open local PDF file" << endl;
          results.push_back(result);
          continue;
        }

        stringstream buffer;
        buffer << in_file.rdbuf();
        string pdf_binary = buffer.str();
        in_file.close();

        // Process with Docling using existing NetworkTools member function
        NetworkTools net;
        string content = net.process_pdf_with_docling(pdf_binary);

        if (content.find("[Docling Error") != string::npos ||
            content.find("[Curl Init Failed]") != string::npos) {
          result["error"] = content;
          cerr << "PDF processing failed: " + path << endl;
        } else {
          result["content"] = NetworkTools::limit_context_size(content);
          cerr << "Successfully processed PDF: " + path + " (" + to_string(result["content"].length()) + " bytes)" << endl;
        }
      }
    } else if (is_url) {
      // Remote non-PDF URL - use NetworkTools fetch_urls mechanism
      log_diagnostic("Fetching remote URL: " + path, true /* logOnly */);

      NetworkTools net;
      vector<string> url_list = {path};
      vector<map<string, string>> network_results = net.fetch_urls(url_list);

      if (!network_results.empty()) {
        result["content"] = network_results[0]["content"];
        result["error"] = network_results[0]["error"];

        if (result["error"].empty() && !result["content"].empty()) {
          cerr <<  "Successfully fetched remote URL: " + path + " (" + to_string(result["content"].length()) + " bytes)" << endl;
        } else {
          cerr << "Remote URL fetch failed: " + path << endl;
        }
      } else {
        result["error"] = "[No results from network fetch]";
        cerr << "Network fetch returned empty results" << endl;
      }
    } else {
      // Regular local text file handling
      ifstream in_file(_get_fullpath(path));
      if (!in_file.is_open()) {
        result["error"] = "Failed to open file for reading: " + path;
        cerr << "Error: Failed to open file: " << endl;
        results.push_back(result);
        continue;
      }

      stringstream buffer;
      buffer << in_file.rdbuf();
      string content = buffer.str();

      // Log success without content - only file size, never the actual content
      // Use logOnly=true to ensure content is never shown on stdout or in logfile
      log_diagnostic("Successfully read: " + path + " (size=" + to_string(content.length()) + " bytes)", true /* logOnly */);

      escape_parameter_tags(content); // Escape any literal XML tags before sending to LLM

      result["status"] = "success";
      result["content"] = content;
    }

    results.push_back(result);
  }

  return results;
}

map<string, string> FileSystemTools::write_file(const string& path, const string& content) {
  // Build human-readable function call syntax (truncate for display)
  string path_str = "\"" + (path.length() > 50 ? path.substr(0, 47) + "..." : path) + "\"";

  // Output the tool function call to both stdout and logfile
  log_tool_diagnostic("write_file(" + path_str + ")");

  string fullpath = _get_fullpath(path);

  // Unescape PARAM_END tokens before writing to disk (LLM provides escaped content)
  string writable_content = content;
  unescape_parameter_tags(writable_content);

  ofstream out_file(fullpath);
  if (!out_file.is_open()) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to open file for writing: " + path;
    log_diagnostic("Error: Failed to open file for writing", true /* logOnly */);
    return result;
  }

  out_file << writable_content;
  bool write_ok = !out_file.fail();  // Capture stream state BEFORE close() clears it
  out_file.flush(); // Ensure data is written to disk immediately
  out_file.close();

  map<string, string> result;
  if (!write_ok) {
    result["status"] = "error";
    result["error"] = "Write failed for file: " + path;
    log_diagnostic("Error: Write failed for file", true /* logOnly */);
    return result;
  }
  result["status"] = "success";
  result["bytes"] = to_string(writable_content.length());
  // Log success without content - use logOnly=true to ensure content is never shown
  log_diagnostic("Successfully written: " + path + " (size=" + to_string(writable_content.length()) + " bytes)", true /* logOnly */);
  return result;
}

map<string, string> FileSystemTools::edit_file(const string& path, const string& old_str, const string& new_str) {
  // Build human-readable function call syntax (truncate for display)
  string path_str = "\"" + (path.length() > 50 ? path.substr(0, 47) + "..." : path) + "\"";

  // Build the function call label (logged at end with change count)
  string edit_label = "edit_file(" + path_str + ")";

  // If LLLM_DEBUG=1, also output OLD and NEW text (truncated in stdout only)
  if (is_debug) {
    string old_str_trunc = "\"" + (old_str.length() > 80 ? old_str.substr(0, 77) + "..." : old_str) + "\"";
    string new_str_trunc = "\"" + (new_str.length() > 80 ? new_str.substr(0, 77) + "..." : new_str) + "\"";
    log_diagnostic("OLD_TEXT: " + old_str_trunc);
    log_diagnostic("NEW_TEXT: " + new_str_trunc);
  }

  // Always output full OLD and NEW text to logfile (no truncation)
  log_diagnostic("OLD_TEXT: \"" + old_str + "\"", true /* logOnly */);
  log_diagnostic("NEW_TEXT: \"" + new_str + "\"", true /* logOnly */);

  // Unescape PARAM_END (LLM provides escaped content per prompt instructions)
  string unescaped_old = old_str;
  string unescaped_new = new_str;
  unescape_parameter_tags(unescaped_old);
  unescape_parameter_tags(unescaped_new);

  string fullpath = _get_fullpath(path);
  ifstream in_file(fullpath);
  if (!in_file.is_open()) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to open file for reading: " + path;
    log_diagnostic("Error: Failed to open file for reading", true /* logOnly */);
    return result;
  }

  stringstream buffer;
  buffer << in_file.rdbuf();
  string content = buffer.str();
  in_file.close();

  size_t pos = 0;
  int changes_count = 0;

  if (unescaped_old.empty()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "OLD_TEXT cannot be empty";
      log_diagnostic("Error: OLD_TEXT is empty", true /* logOnly */);
      return result;
  }

  while ((pos = content.find(unescaped_old, pos)) != string::npos) {
    content.replace(pos, unescaped_old.length(), unescaped_new);
    pos += unescaped_new.length();
    changes_count++;
  }

  if (changes_count == 0) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "String not found in file. Ensure the OLD_TEXT exactly matches the file contents, including all whitespace and newlines.";
    log_diagnostic("Error: String not found in file", true /* logOnly */);
    return result;
  }

  if (content.empty()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "CRITICAL ERROR: Content is empty - refusing to write empty file";
      log_diagnostic("Error: Content is empty after edit", true /* logOnly */);
      return result;
  }

  ofstream out_file(fullpath);
  if (!out_file.is_open()) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to open file for writing after edit";
    log_diagnostic("Error: Failed to open file for writing after edit", true /* logOnly */);
    return result;
  }

  out_file << content;
  out_file.flush(); // Ensure data is written to disk immediately
  out_file.close();

  map<string, string> result;
  if (!out_file) {
    result["status"] = "error";
    result["error"] = "Write failed after edit for file: " + path;
    log_diagnostic("Error: Write failed after edit", true /* logOnly */);
    return result;
  }
  result["status"] = "updated";
  result["changes"] = "Successfully applied replacement (" + to_string(changes_count) + " total occurrences modified).";
  log_tool_diagnostic(edit_label + ": " + to_string(changes_count) + " change(s)");
  return result;
}
