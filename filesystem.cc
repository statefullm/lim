#include "filesystem.h"
#include "parsers.h"
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

// Static constant definition
const string FileSystemTools::HOME = "/home/ai";

// Remove trailing spaces from a string (used for cleaning tool arguments)
string remove_trailing_spaces(const string& str) {
    string result;
    result.reserve(str.length());

    string current_line;
    for (char c : str) {
        if (c == '\n') {
            // Remove trailing spaces from the current line before adding newline
            while (!current_line.empty() && isspace(current_line.back())) {
                current_line.pop_back();
            }
            result += current_line + '\n';
            current_line.clear();
        } else {
            current_line += c;
        }
    }

    // Process the last line (if any)
    while (!current_line.empty() && isspace(current_line.back())) {
        current_line.pop_back();
    }
    result += current_line;

    return result;
}

FileSystemTools::FileSystemTools() {
  // Constructor
}

string FileSystemTools::_get_fullpath(const string& path) {
  // Get current working directory by reading ~/.cwd directly
  // This uses ~/.cwd which is synchronized via cd() function in ~/.bashrc
  string home_cwd = HOME + "/.cwd";
  ifstream cwd_file(home_cwd);
  string cwd_str;

  if (cwd_file.is_open()) {
    getline(cwd_file, cwd_str);
    cwd_file.close();
  } else {
    // Fallback to HOME if ~/.cwd doesn't exist
    cwd_str = HOME;
  }

  string fullpath;

  // If path is relative, join with cwd
  if (path[0] != '/') {
    fullpath = cwd_str + "/" + path;
  } else {
    fullpath = path;
  }

  // Resolve to absolute path
  char result[PATH_MAX];
  if (realpath(fullpath.c_str(), result) != nullptr) {
    return string(result);
  }
  return fullpath;
}

string FileSystemTools::exec_shell(const string& command) {
  // Add robust single-quote escaping so the LLM's command doesn't break the bash -c '' wrapper
  string safe_cmd = command;
  size_t pos = 0;
  while ((pos = safe_cmd.find("'", pos)) != string::npos) {
    safe_cmd.replace(pos, 1, "'\\''");
    pos += 4;
  }

/*
   Run the command in the cwd
   This requires ~/.bashrc to define
   cd() {
     builtin cd "$@" && pwd > ~/.cwd
   }
*/
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

  vector<string> lines;
  string line;
  while (getline(in_file, line)) {
    lines.push_back(line);
  }
  in_file.close();

  string result = "";
  int match_count = 0;
  int context = 5; // Configurable number of lines before and after

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

      i = end; // fast-forward to avoid overlapping context windows
    }
    i++;
  }

  if (match_count == 0) {
    return "No occurrences found for text: " + text;
  }
  return result;
}

vector<map<string, string>> FileSystemTools::read_files(const vector<string>& paths) {
  vector<map<string, string>> results;

  for (const auto& path : paths) {
    map<string, string> file_data;
    file_data["path"] = path;

    string fullpath = _get_fullpath(path);
    ifstream file(fullpath);

    if (file.is_open()) {
      stringstream buffer;
      buffer << file.rdbuf();
      file_data["content"] = buffer.str();
      file_data["error"] = "";
      file.close();
    } else {
      file_data["content"] = "";
      file_data["error"] = "Failed to open file: " + fullpath;
    }
    results.push_back(file_data);
  }

  return results;
}

map<string, string> FileSystemTools::write_file(const string& path, const string& content) {
  string fullpath = _get_fullpath(path);

  // Check if path is within HOME
  string normalized_fullpath = fullpath;
  if (normalized_fullpath.substr(0, HOME.size()) != HOME) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Write access denied outside " + HOME;
    return result;
  }

  // CRITICAL: Verify content is not empty before writing (prevents accidental data loss)
  if (content.empty()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "CRITICAL ERROR: Cannot write empty content";
      return result;
  }

  ofstream file(fullpath);
  if (file.is_open()) {
    file << content;
    file.close();

    map<string, string> result;
    result["status"] = "success";
    return result;
  } else {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to open file for writing: " + fullpath;
    return result;
  }
}

map<string, string> FileSystemTools::edit_file(const string& path, const string& old_str, const string& new_str) {
  string fullpath = _get_fullpath(path);

  // Check if path is within HOME
  string normalized_fullpath = fullpath;
  if (normalized_fullpath.substr(0, HOME.size()) != HOME) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Edit access denied outside " + HOME;
    return result;
  }

  // Read current content
  ifstream in_file(fullpath);
  if (!in_file.is_open()) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "Failed to open file for reading: " + fullpath;
    return result;
  }

  stringstream buffer;
  buffer << in_file.rdbuf();

  // CRITICAL: Verify that content was actually read
  if (buffer.fail() || buffer.bad()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "Failed to read file content";
      return result;
  }

  string content = buffer.str();
  in_file.close();

  // CRITICAL: Verify that content is not empty before proceeding
  if (content.empty()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "File is empty - cannot perform edit operation";
      return result;
  }

  // --- ATOMIC PRE-FLIGHT CHECK ---
  if (old_str.empty()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "The 'old' search string is completely empty. You cannot replace an empty string.";
      return result;
  }

  if (content.find(old_str) == string::npos) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "Exact match not found for the following string:\n```\n" + old_str + "\n```\nEnsure your spacing, indentation, and newlines match the file exactly.";
      return result;
  }

  int changes_count = 0;
  string snippets = "";

  // Find all occurrences first to avoid infinite loops when new_str contains old_str
  vector<size_t> match_positions;
  size_t search_pos = 0;
  while ((search_pos = content.find(old_str, search_pos)) != string::npos) {
    match_positions.push_back(search_pos);
    // Advance by old_str.length() to find the next non-overlapping occurrence
    search_pos += old_str.length();
  }

  // Replace in reverse order to avoid position shifting issues
  for (int i = match_positions.size() - 1; i >= 0; --i) {
    size_t pos = match_positions[i];

    // --- GENERATE CONTEXT SNIPPET ---
    size_t window = 150; // Grab ~150 chars before and after
    size_t context_start = (pos > window) ? pos - window : 0;
    size_t nl_start = content.rfind('\n', context_start);
    context_start = (nl_start == string::npos) ? 0 : nl_start + 1; // Snap to newline

    size_t context_end = pos + new_str.length() + window;
    if (context_end > content.length()) context_end = content.length();
    size_t nl_end = content.find('\n', context_end);
    context_end = (nl_end == string::npos) ? content.length() : nl_end; // Snap to newline

    snippets += "\n\n--- Edit Occurrence " + to_string(i + 1) + " Context ---\n";
    snippets += "...\n" + content.substr(context_start, context_end - context_start) + "\n...";

    content.replace(pos, old_str.length(), new_str);
    changes_count++;
  }

  // CRITICAL: Verify content is not empty before writing (prevents data loss)
  if (content.empty()) {
      map<string, string> result;
      result["status"] = "error";
      result["error"] = "CRITICAL ERROR: Content is empty - refusing to write empty file";
      return result;
  }

  // Write updated content
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
  result["changes"] = "Successfully applied replacement (" + to_string(changes_count) + " total occurrences modified)." + snippets;
  return result;
}

map<string, string> FileSystemTools::chmod_file(const string& path, int mode) {
  string fullpath = _get_fullpath(path);

  // Check if path is within HOME
  string normalized_fullpath = fullpath;
  if (normalized_fullpath.substr(0, HOME.size()) != HOME) {
    map<string, string> result;
    result["status"] = "error";
    result["error"] = "chmod only allowed inside " + HOME;
    return result;
  }

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
    result["error"] = "Failed to change file permissions: " + string(strerror(errno));
    return result;
  }

  map<string, string> result;
  result["status"] = "success";
  return result;
}
