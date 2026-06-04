#include "tools.h"
#include "filesystem.h"
#include "network.h"
#include "parsers.h"
#include "tokens.h"
#include <set>
#include <cstring>
#include <signal.h>

using namespace std;
using namespace Tokens;

extern volatile sig_atomic_t stop_generation;

static bool param_has_newline_impl(const string& s) {
    return s.find('\n') != string::npos || s.find('\r') != string::npos;
}

const string PATH_NEWLINE_ERROR = "System Error: Invalid tool format. The path parameter must not contain newlines.";

bool param_has_newline(const string& s) {
    return param_has_newline_impl(s);
}

string execute_tool_call(const string& tool_call, set<string>& clean_files) {
  string result = "";
  string tool_name = "";
  size_t ns = tool_call.find(FUNC_START);
  if (ns != string::npos) {
      ns += string(FUNC_START).length();
      size_t ne = tool_call.find('>', ns);
      if (ne != string::npos) {
          tool_name = tool_call.substr(ns, ne - ns);
      }
  }

  // Check for interrupt before starting tool execution
  if (stop_generation) return "[Tool interrupted by user]";

  if (tool_name == "read_files") {
    vector<string> paths = extract_array_arg_bounded(tool_call, "paths");
    if (!paths.empty()) {
      FileSystemTools fs;
      NetworkTools net;
      result = "Files content:\n";

      vector<string> local_paths_to_read;

      for (const auto& path : paths) {
        bool is_url = (path.find("http://") == 0 || path.find("https://") == 0);

        if (!is_url && clean_files.count(path)) {
          result += "Path: " + path + "\n";
          result += "Content:\n[Cache hit - unchanged since last read]\n";
          result += "---\n";
        } else if (is_url) {
          auto url_results = net.fetch_urls({path});
          for (const auto& file : url_results) {
            string f_path, f_content, f_error;
            if (file.count("path")) f_path = file.at("path");
            if (file.count("content")) f_content = file.at("content");
            if (file.count("error")) f_error = file.at("error");
            result += "Path: " + f_path + "\n";
            result += "Content:\n" + f_content + "\n";
            if (!f_error.empty()) result += "Error: " + f_error + "\n";
            result += "---\n";

            if (f_error.empty() && !f_path.empty()) clean_files.insert(f_path);
          }
        } else {
          local_paths_to_read.push_back(path);
        }
      }

      if (!local_paths_to_read.empty()) {
        auto results = fs.read_files(local_paths_to_read);
        for (const auto& file : results) {
          string f_path, f_content, f_error;
          if (file.count("path")) f_path = file.at("path");
          if (file.count("content")) f_content = file.at("content");
          if (file.count("error")) f_error = file.at("error");
          result += "Path: " + f_path + "\n";
          result += "Content:\n" + f_content + "\n";
          if (!f_error.empty()) result += "Error: " + f_error + "\n";
          result += "---\n";

          if (f_error.empty() && !f_path.empty()) clean_files.insert(f_path);
        }
      }
    } else {
      result = "Error: No paths provided to read_files";
    }
  } else if (tool_name == "search_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) return PATH_NEWLINE_ERROR;
    string text = extract_string_arg_bounded(tool_call, "text");
    string begin_str = extract_string_arg_bounded(tool_call, "begin");
    string end_str = extract_string_arg_bounded(tool_call, "end");
    int begin_line = 0;
    int end_line = 0;
    if (!begin_str.empty() && begin_str.find_first_not_of("0123456789") == string::npos) {
      begin_line = atoi(begin_str.c_str());
    }
    if (!end_str.empty() && end_str.find_first_not_of("0123456789") == string::npos) {
      end_line = atoi(end_str.c_str());
    }
    if (!path.empty()) {
      FileSystemTools fs;
      result = fs.search_file(path, text, begin_line, end_line);
    } else {
      result = "Error: path is required for search_file";
    }
  } else if (tool_name == "write_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) return PATH_NEWLINE_ERROR;
    string content = extract_string_arg_bounded(tool_call, "content");
    content = remove_trailing_spaces(content);
    clean_files.erase(path);
    if (!path.empty()) {
      FileSystemTools fs;
      auto result_map = fs.write_file(path, content);
      string r_status, r_error;
      if (result_map.count("status")) r_status = result_map.at("status");
      if (result_map.count("error")) r_error = result_map.at("error");
      result = "Status: " + r_status;
      if (!r_error.empty()) result += ", Error: " + r_error;
    } else {
      result = "Error: No path provided to write_file";
    }
  } else if (tool_name == "edit_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) return PATH_NEWLINE_ERROR;
    string old_str = extract_string_arg_bounded(tool_call, "old");
    string new_str = extract_string_arg_bounded(tool_call, "new");
    new_str = remove_trailing_spaces(new_str);
    clean_files.erase(path);
    if (!path.empty()) {
      FileSystemTools fs;
      auto result_map = fs.edit_file(path, old_str, new_str);
      string r_status, r_changes, r_error;
      if (result_map.count("status")) r_status = result_map.at("status");
      if (result_map.count("changes")) r_changes = result_map.at("changes");
      if (result_map.count("error")) r_error = result_map.at("error");
      result = "Status: " + r_status;
      if (!r_changes.empty()) {
          int n = atoi(r_changes.c_str());
          result += ", " + to_string(n) + (n == 1 ? " change" : " changes");
      }
      if (!r_error.empty()) result += ", Error: " + r_error;
    } else {
      result = "Error: No path provided to edit_file";
    }
  } else if (tool_name == "exec_shell") {
    string command = extract_string_arg_bounded(tool_call, "command");
    clean_files.clear();
    if (!command.empty()) {
      FileSystemTools fs;
      result = fs.exec_shell(command);
    } else {
      result = "Error: No command provided to exec_shell";
    }
  } else if (tool_name == "web_search") {
    string query = extract_string_arg_bounded(tool_call, "query");
    clean_files.clear();
    if (!query.empty()) {
      NetworkTools net;
      result = net.web_search(query);
    } else {
      result = "Error: No query provided to web_search";
    }
  } else {
    result = "Error: Unknown tool call";
  }

  return result;
}

string sanitize(string text) {
    size_t pos = 0;
    while ((pos = text.find(PARAM_END, pos)) != string::npos) {
        text.replace(pos, strlen(PARAM_END), PARAM_END_ESC);
        pos += strlen(PARAM_END_ESC);
    }
    return text;
}
