#include "filesystem.h"
#include "parsers.h"
#include "tokens.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pwd.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits.h>

using namespace std;

// --- Helper to escape tags from disk so they don't break the LLM's XML parser ---
static void escape_parameter_tags(std::string& str) {
    std::string from = Tokens::PARAM_END;
    std::string to = Tokens::PARAM_END_ESC;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

// --- Consolidated Diagnostic Logging Function ---
void log_diagnostic(const string& message, bool logOnly /* = false */, bool debugOnly /* = false */,
                    const string& tag /* = "" */) {
    if (!chat_log.is_open()) {
        cerr << "[ERROR] chat_log is not open - cannot write diagnostic message\n";
        return;
    }

    string full_message = message;
    if (!tag.empty()) {
        full_message = tag + " " + message;
    }

    if (!logOnly) {
        if (!debugOnly || is_debug) {
            cout << full_message << endl;
            fflush(stdout);
        }
    }

    chat_log << full_message << "\n";
    chat_log.flush();
}

const string FileSystemTools::HOME = "/home/ai";

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
  if (path[0] != '/') {
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
  string safe_cmd = command;
  size_t pos = 0;
  while ((pos = safe_cmd.find("'", pos)) != string::npos) {
    safe_cmd.replace(pos, 1, "'\\''");
    pos += 4;
  }

  string full_command = "bash -c 'cd \"$(cat ~/.cwd 2>/dev/null || echo " + HOME + ")\" && " + safe_cmd + " 2>&1'";

  string result;
  FILE* pipe = popen(full_command.c_str(), "r");
  if (!pipe) {
    return "Error: Failed to execute command";
  }

  char buffer[1024];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }

  pclose(pipe);
  if (result.empty()) {
    result = "[Command executed with no output]\n";
  }
  return result;
}

string FileSystemTools::search_file(const string& path, const string& text) {
  string fullpath = _get_fullpath(path);
  ifstream in_file(fullpath);
  if (!in_file.is_open()) {
    return "Error: Failed to open file for reading: " + fullpath;
  }

  stringstream buffer;
  buffer << in_file.rdbuf();
  string content = buffer.str();
  in_file.close();

  bool search_with_newlines = (text.find('\n') != string::npos);
  string result = "";
  int match_count = 0;
  int context = 5;

  if (search_with_newlines) {
    size_t pos = 0;
    while ((pos = content.find(text, pos)) != string::npos) {
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

      size_t end_pos = pos + text.length();
      for (size_t i = 0; i < end_pos && i < content.length(); i++) {
        if (content[i] == '\n') end_line++;
      }

      result += "--- Match " + to_string(match_count) + " (Lines " + to_string(start_line) + "-" + to_string(end_line) + ") ---\n```\n";

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
      result += "```\n\n";
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
      if (lines[i].find(text) != string::npos) {
        match_count++;
        if (match_count > 10) {
          result += "... (Truncated after 10 matches)\n";
          break;
        }
        int start = max(0, i - context);
        int end = min((int)lines.size() - 1, i + context);
        result += "--- Match " + to_string(match_count) + " (Lines " + to_string(start + 1) + "-" + to_string(end + 1) + ") ---\n```\n";
        for (int j = start; j <= end; j++) {
          result += lines[j] + "\n";
        }
        result += "```\n\n";
        i = end;
      }
      i++;
    }
  }

  if (match_count == 0) {
    log_diagnostic("No occurrences found for text: " + text, true, false, "[Search]");
    return "No occurrences found for text.\n";
  }

  escape_parameter_tags(result); // Escape any literal XML tags before sending to LLM
  return result;
}

vector<map<string, string>> FileSystemTools::read_files(const vector<string>& paths) {
  vector<map<string, string>> results;

  for (const auto& path : paths) {
    map<string, string> result;
    result["path"] = path;
    result["content"] = "";
    result["error"] = "";

    string fullpath = _get_fullpath(path);

    ifstream in_file(fullpath);
    if (!in_file.is_open()) {
      result["error"] = "Failed to open file for reading: " + fullpath;
      results.push_back(result);
      continue;
    }

    stringstream buffer;
    buffer << in_file.rdbuf();
    string content = buffer.str();

    escape_parameter_tags(content); // Escape any literal XML tags before sending to LLM

    result["status"] = "success";
    result["content"] = content;
    results.push_back(result);
  }

  return results;
}

map<string, string> FileSystemTools::write_file(const string& path, const string& content) {
  string fullpath = _get_fullpath(path);

  ofstream out_file(fullpath);
  if (!out_file.is_open()) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to open file for writing: " + fullpath;
    return result;
  }

  out_file << content;
  out_file.close();

  map<string, string> result;
  result["status"] = "success";
  return result;
}

map<string, string> FileSystemTools::edit_file(const string& path, const string& old_str, const string& new_str) {
  string fullpath = _get_fullpath(path);

  ifstream in_file(fullpath);
  if (!in_file.is_open()) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to open file for reading: " + fullpath;
    return result;
  }

  stringstream buffer;
  buffer << in_file.rdbuf();
  string content = buffer.str();
  in_file.close();

  size_t pos = 0;
  int changes_count = 0;

  if (old_str.empty()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "OLD_TEXT cannot be empty";
      return result;
  }

  while ((pos = content.find(old_str, pos)) != string::npos) {
    content.replace(pos, old_str.length(), new_str);
    pos += new_str.length();
    changes_count++;
  }

  if (changes_count == 0) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "String not found in file. Ensure the OLD_TEXT exactly matches the file contents, including all whitespace and newlines.";
    return result;
  }

  if (content.empty()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "CRITICAL ERROR: Content is empty - refusing to write empty file";
      return result;
  }

  ofstream out_file(fullpath);
  if (!out_file.is_open()) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to open file for writing after edit";
    return result;
  }

  out_file << content;
  out_file.close();

  map<string, string> result;
  result["status"] = "updated";
  result["changes"] = "Successfully applied replacement (" + to_string(changes_count) + " total occurrences modified).";
  return result;
}

map<string, string> FileSystemTools::chmod_file(const string& path, int mode) {
  string fullpath = _get_fullpath(path);
  int octal_mode = 0;
  int temp_mode = mode;
  int multiplier = 1;
  while (temp_mode > 0) {
    octal_mode += (temp_mode % 10) * multiplier;
    temp_mode /= 10;
    multiplier *= 8;
  }

  if (::chmod(fullpath.c_str(), octal_mode) != 0) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to change permissions";
    return result;
  }

  map<string, string> result;
  result["status"] = "success";
  return result;
}
