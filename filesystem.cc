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
#include <sys/wait.h>
#include <poll.h>
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

const string FileSystemTools::HOME = "/home/ai";

// Helper: extract the file extension (lowercased) from a path, or empty string.
static string _file_ext(const string& path) {
  size_t dot = path.rfind('.');
  if (dot == string::npos) return "";
  string ext = path.substr(dot);
  transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
  return ext;
}

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

string FileSystemTools::exec_shell(const string& command, function<void()> on_open,
                                   function<void(const string&)> on_chunk,
                                   function<void(const string&)> on_close) {
  // Build human-readable function call syntax
  string cmd_str = "\"" + command + "\"";

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

  string full_command = "stdbuf -oL -eL bash -c 'source ~/.bashrc 2>/dev/null; cd \"$(cat ~/.cwd 2>/dev/null || echo " + HOME + ")\" && " + safe_cmd + " ; echo \"$?\" > \"" + exit_code_file + "\"' 2>&1";

  // Use fork + pipe to stream output as it becomes available.
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    unlink(exit_code_file.c_str());
    return "Error: Failed to create pipe for shell execution.";
  }

  // Maximize the pipe buffer to prevent EPIPE.
  fcntl(pipefd[0], F_SETPIPE_SZ, 1048576);
  pid_t pid = fork();
  if (pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    unlink(exit_code_file.c_str());
    return "Error: Failed to spawn shell process. The exec_shell tool may be temporarily unavailable.";
  }

  if (pid == 0) {
    // Child process: redirect stdout/stderr to the pipe, then exec.
    close(pipefd[0]); // Close read end
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);

    execl("/bin/bash", "bash", "-c", full_command.c_str(), nullptr);
    // If execl fails, write error and exit
    fprintf(stderr, "Error: Failed to exec bash\n");
    _exit(127);
  }

  // Parent process: read from pipe incrementally.
  close(pipefd[1]); // Close write end

  // Make the read end non-blocking so we can detect EOF while streaming.
  int flags = fcntl(pipefd[0], F_GETFL, 0);
  fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

  string result;

  // Limit output to avoid overwhelming LLM context and browser display.
  // All three consumers (LLM result, browser stream, chat_log) see the same bounded output.
  static constexpr size_t MAX_OUTPUT_SIZE = 32768;  // 32KB
  bool truncated = false;

  // Signal that streaming is about to begin.
  if (on_open) on_open();

  struct pollfd pfd;
  pfd.fd = pipefd[0];
  pfd.events = POLLIN;

  while (true) {
    // Block until the pipe has data or the child exits.
    int ready = poll(&pfd, 1, -1);  // -1 = block indefinitely
    if (ready < 0) {
      // Interrupted by signal - retry.
      if (errno == EINTR) continue;
      break;  // Unexpected error, bail out.
    }

    if (pfd.revents & (POLLIN | POLLHUP)) {
      char buffer[65536];
      ssize_t n = read(pipefd[0], buffer, sizeof(buffer) - 1);

      if (n > 0) {
        buffer[n] = '\0';

        // Keep draining the child pipe to prevent EPIPE, but only
        // forward data to consumers until we hit the limit.
        if (!truncated) {
          size_t remaining = MAX_OUTPUT_SIZE - result.size();
          if (static_cast<size_t>(n) > remaining) {
            // This chunk would exceed the limit - clip it.
            result += std::string(buffer, remaining);
            if (on_chunk) on_chunk(std::string(buffer, remaining));
            if (chat_log.is_open()) { chat_log << std::string(buffer, remaining); chat_log.flush(); }

            // Inject truncation marker for all consumers.
            string trunc_msg = "\n[Output truncated at " + std::to_string(MAX_OUTPUT_SIZE) + " bytes]\n";
            result += trunc_msg;
            if (on_chunk) on_chunk(trunc_msg);
            if (chat_log.is_open()) { chat_log << trunc_msg; chat_log.flush(); }
            truncated = true;
          } else {
            result += buffer;
            if (on_chunk) on_chunk(buffer);
            if (chat_log.is_open()) { chat_log << buffer; chat_log.flush(); }
          }
        }
        // If already truncated, just discard - child is still being drained.
      } else {
        // n == 0 or n < 0: EOF / pipe closed.
        break;
      }
    }

    // Check if child has exited regardless of poll events.
    int wstatus = 0;
    pid_t waited = waitpid(pid, &wstatus, WNOHANG);
    if (waited > 0) {
      // Drain any remaining data in the pipe (prevent EPIPE).
      if (pfd.revents & (POLLIN | POLLHUP)) {
        char buffer[65536];
        ssize_t n = read(pipefd[0], buffer, sizeof(buffer) - 1);
        // Discard - child is exiting, just drain.
      }
      break;
    }
    if (waited == -1) break;  // Child gone.
  }

  close(pipefd[0]);

  // Signal that streaming is complete, passing the accumulated result.
  if (on_close) on_close(result);

  // Wait for child to finish and get exit status.
  int pipe_status = 0;
  waitpid(pid, &pipe_status, 0);

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

  // Build the final result with exit code information for better LLM diagnostics
  if (result.empty()) {
    chat_log << "[Command executed with no output]\n\n";
    chat_log.flush();

    if (exit_code == 0) {
      result = "[Command exited with code 0 and produced no output]\n";
    } else if (exit_code > 0) {
      result = "[Command exited with code " + to_string(exit_code) + " and produced no output. The command may have failed silently -- check your syntax or try a different approach.]\n";
    } else {
      result = "[Command produced no output; exit code could not be determined]\n";
    }
  } else {
    chat_log << "\n\n";
    chat_log.flush();

    // Always include exit code so the LLM can distinguish success from failure
    if (exit_code > 0) {
      result = "[Exit code: " + to_string(exit_code) + "]\n" + result;
    }
  }
  return result;
}

map<string, string> FileSystemTools::search_file(const string& path, const string& text, const string& begin_str, const string& end_str) {
  map<string, string> out;
  out["content"] = "";
  out["match_count"] = "0";
  out["actual_start"] = "0";
  out["actual_end"] = "0";
  out["error"] = "";

  // Build human-readable function call syntax
  string path_str = "\"" + path + "\"";

  // Build the function call label (logged at end with match count for text searches)
  string search_label = "search_file(" + path_str;
  if (!begin_str.empty() || !end_str.empty()) {
    search_label += ", lines " + begin_str + "-" + end_str;
  }
  search_label += ")";

  // Parse and validate begin/end -- parse as integers first, then validate.
  int begin_line = 0, end_line = 0;
  if (!begin_str.empty()) {
    char* endptr = nullptr;
    long val = strtol(begin_str.c_str(), &endptr, 10);
    if (*endptr != '\0' || begin_str.empty() || val < 1) {
      out["error"] = "Error: 'begin' must be a positive integer.";
      log_tool_diagnostic(search_label);
      out["display"] = "Search file: " + path + ": " + out["error"];
      return out;
    }
    begin_line = static_cast<int>(val);
  }
  if (!end_str.empty()) {
    char* endptr = nullptr;
    long val = strtol(end_str.c_str(), &endptr, 10);
    if (*endptr != '\0' || end_str.empty() || val < 1) {
      out["error"] = "Error: 'end' must be a positive integer.";
      log_tool_diagnostic(search_label);
      out["display"] = "Search file: " + path + ": " + out["error"];
      return out;
    }
    end_line = static_cast<int>(val);
  }

  // Validate range: begin must not exceed end, whenever both are provided.
  if (begin_line >= 1 && end_line >= 1 && begin_line > end_line) {
    out["error"] = "Invalid range: 'begin' (" + to_string(begin_line) + ") is greater than 'end' (" + to_string(end_line) + ").";
    log_tool_diagnostic(search_label);
    out["display"] = "Search file: " + path + ": " + out["error"];
    return out;
  }

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
    out["error"] = "Failed to open file for reading: " + fullpath;
    log_tool_diagnostic(search_label);
    out["display"] = "Search file: " + path + ": " + out["error"];
    return out;
  }

  stringstream buffer;
  buffer << in_file.rdbuf();
  string content = buffer.str();
  in_file.close();

  // (Range validation moved earlier, after begin/end parsing.)

  // Line range reading mode: if text is empty and both begin and end are provided, return those lines directly
  if (text.empty() && begin_line >= 1 && end_line >= begin_line) {
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
      out["error"] = "Line " + to_string(begin_line) + " is beyond the end of file (" + to_string(lines.size()) + " lines).";
      log_tool_diagnostic(search_label);
      out["display"] = "Search file: " + path + ": " + out["error"];
      return out;
    }

    // Store actual range for display
    out["actual_start"] = to_string(start + 1);
    out["actual_end"] = to_string(end + 1);
    out["match_count"] = "1";

    string result = "Lines " + to_string(start + 1) + "-" + to_string(end + 1) + " of " + path + ":\n";
    for (int i = start; i <= end; i++) {
      result += lines[i] + "\n";
    }

    out["content"] = result;
  } else if (!text.empty()) {
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
        int start_line = 1, end_line_local = 1;
        for (size_t i = 0; i < pos; i++) {
          if (content[i] == '\n') start_line++;
        }

        size_t end_pos = pos + unescaped_text.length();
        for (size_t i = 0; i < end_pos && i < content.length(); i++) {
          if (content[i] == '\n') end_line_local++;
        }

        result += "--- Match " + to_string(match_count) + " (Lines " + to_string(start_line) + "-" + to_string(end_line_local) + ") ---\n";

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
        out["error"] = "Failed to reopen file for line-by-line reading: " + fullpath;
        log_tool_diagnostic(search_label);
        out["display"] = "Search file: " + path + ": " + out["error"];
        return out;
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

      out["match_count"] = to_string(match_count);

      if (match_count > 0) {
        out["content"] = result;
      }
    }
  }

  // Single exit point: log and build display string from the final state of out.
  log_tool_diagnostic(search_label);
  if (out["error"].empty()) {
      int n = atoi(out["match_count"].c_str());
      out["display"] = "Search file: " + path + ": " + to_string(n) + (n == 1 ? " match" : " matches");
  } else {
      out["display"] = "Search file: " + path + ": " + out["error"];
  }
  return out;
}

vector<map<string, string>> FileSystemTools::read_files(const vector<string>& paths) {
  // Output function call to both stdout and logfile for each file
  for (const auto& path : paths) {
    string path_str = "\"" + path + "\"";
    log_tool_diagnostic("read_file(" + path_str + ")");
  }

  vector<map<string, string>> results;

  // Check if any file is a PDF to determine if we need Docling
  bool needs_pdf_processing = false;
  for (const auto& path : paths) {
    if (_file_ext(path) == ".pdf") {
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
    bool is_pdf = (_file_ext(path) == ".pdf");

    if (is_pdf) {
      // Unified PDF handling - local files read directly, URLs use NetworkTools
      cerr << "Processing PDF: " + path << endl;

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
        results.push_back(result);
        continue;
      }

      stringstream buffer;
      buffer << in_file.rdbuf();
      string content = buffer.str();

      // Log success without content - only file size, never the actual content
      // Use logOnly=true to ensure content is never shown on stdout or in logfile
      log_diagnostic("Successfully read: " + path + " (size=" + to_string(content.length()) + " bytes)", true /* logOnly */);

      result["status"] = "success";
      result["content"] = content;
    }

    results.push_back(result);
  }

  return results;
}

map<string, string> FileSystemTools::write_file(const string& path, const string& content) {
  // Build human-readable function call syntax
  string path_str = "\"" + path + "\"";

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
    log_diagnostic(result["error"], true /* logOnly */);
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
    log_diagnostic(result["error"], true /* logOnly */);
    return result;
  }
  result["status"] = "success";
  // Log success without content - use logOnly=true to ensure content is never shown
  log_diagnostic("Successfully written: " + path + " (size=" + to_string(writable_content.length()) + " bytes)", true /* logOnly */);
  return result;
}

map<string, string> FileSystemTools::edit_file(const string& path, const string& old_str, const string& new_str) {
  // Build human-readable function call syntax
  string path_str = "\"" + path + "\"";

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
    log_diagnostic(result["error"], true /* logOnly */);
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
      log_diagnostic(result["error"], true /* logOnly */);
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
    log_diagnostic(result["error"], true /* logOnly */);
    return result;
  }

  if (content.empty()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "CRITICAL ERROR: Content is empty - refusing to write empty file";
      log_diagnostic(result["error"], true /* logOnly */);
      return result;
  }

  ofstream out_file(fullpath);
  if (!out_file.is_open()) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to open file for writing after edit";
    log_diagnostic(result["error"], true /* logOnly */);
    return result;
  }

  out_file << content;
  out_file.flush(); // Ensure data is written to disk immediately
  out_file.close();

  map<string, string> result;
  if (!out_file) {
    result["status"] = "error";
    result["error"] = "Write failed after edit for file: " + path;
    log_diagnostic(result["error"], true /* logOnly */);
    return result;
  }
  result["status"] = "updated";
  result["changes"] = to_string(changes_count);
  log_tool_diagnostic(edit_label);
  return result;
}
