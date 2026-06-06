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

string execute_tool_call(const string& tool_call, set<string>& clean_files, string& display_result) {
  display_result = "";
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
          display_result += (display_result.empty() ? "Read files: " : "\n            ") + path + " (cached)";
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

            if (!f_error.empty()) {
              display_result += (display_result.empty() ? "Read files: " : "\n            ") + f_path + ": " + f_error;
            } else {
              display_result += (display_result.empty() ? "Read files: " : "\n            ") + f_path + ": " + to_string(f_content.length()) + " bytes";
            }
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

          if (!f_error.empty()) {
            display_result += (display_result.empty() ? "Read files: " : "\n            ") + f_path + ": " + f_error;
          } else {
            display_result += (display_result.empty() ? "Read files: " : "\n            ") + f_path + ": " + to_string(f_content.length()) + " bytes";
          }
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
    if (!begin_str.empty()) {
      if (begin_str.find_first_not_of("0123456789") == string::npos) {
        begin_line = atoi(begin_str.c_str());
        if (begin_line < 1) return "Error: 'begin' must be a positive integer (>= 1).";
      } else {
        return "Error: 'begin' must be a positive integer.";
      }
    }
    if (!end_str.empty()) {
      if (end_str.find_first_not_of("0123456789") == string::npos) {
        end_line = atoi(end_str.c_str());
        if (end_line < 1) return "Error: 'end' must be a positive integer (>= 1).";
      } else {
        return "Error: 'end' must be a positive integer.";
      }
    }
    if (!path.empty()) {
      FileSystemTools fs;
      auto result_map = fs.search_file(path, text, begin_line, end_line);

      string r_content, r_error, r_match_count, r_actual_start, r_actual_end;
      if (result_map.count("content")) r_content = result_map.at("content");
      if (result_map.count("error")) r_error = result_map.at("error");
      if (result_map.count("match_count")) r_match_count = result_map.at("match_count");
      if (result_map.count("actual_start")) r_actual_start = result_map.at("actual_start");
      if (result_map.count("actual_end")) r_actual_end = result_map.at("actual_end");

      // Build the LLM-facing result string
      if (!r_error.empty()) {
          result = "Error: " + r_error;
      } else if (r_content.empty()) {
          result = "No occurrences found for text.";
      } else {
          result = r_content;
      }

      // Build display_result from structured data -- no string parsing needed
      int n_matches = atoi(r_match_count.c_str());
      display_result = "Search file: " + path;
      if (text.empty() && begin_line > 0 && end_line >= begin_line) {
          // Use the actual clamped range returned by search_file
          int a_start = atoi(r_actual_start.c_str());
          int a_end   = atoi(r_actual_end.c_str());
          if (a_start > 0 && a_end > 0) {
              display_result += ", lines " + to_string(a_start) + "-" + to_string(a_end);
          } else {
              display_result += ", lines " + to_string(begin_line) + "-" + to_string(end_line);
          }
          int n_found = r_error.empty() ? 1 : 0;
          display_result += ": " + to_string(n_found) + " match" + (n_found != 1 ? "es" : "");
      } else if (begin_line > 0 && end_line > 0) {
          display_result += ", lines " + to_string(begin_line) + "-" + to_string(end_line);
          display_result += ": 0 matches";
      } else {
          display_result += ": " + to_string(n_matches) + (n_matches == 1 ? " match" : " matches");
      }
    } else {
      result = "Error: path is required for search_file";
    }
  } else if (tool_name == "write_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) return PATH_NEWLINE_ERROR;
    string content = extract_string_arg_bounded(tool_call, "content");
    clean_files.erase(path);
    if (!path.empty()) {
      FileSystemTools fs;
      auto result_map = fs.write_file(path, content);
      string r_status, r_error;
      if (result_map.count("status")) r_status = result_map.at("status");
      if (result_map.count("error")) r_error = result_map.at("error");
      result = "Status: " + r_status;
      if (!r_error.empty()) result += ", Error: " + r_error;

      if (!r_error.empty()) {
          display_result = "Write file: " + path + ": " + r_error;
      } else {
          display_result = "Write file: " + path + ": " + to_string(content.length()) + " bytes";
      }
    } else {
      result = "Error: No path provided to write_file";
    }
  } else if (tool_name == "edit_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) return PATH_NEWLINE_ERROR;
    string old_str = extract_string_arg_bounded(tool_call, "old");
    string new_str = extract_string_arg_bounded(tool_call, "new");
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
          display_result = "Edit file: " + path + ": " + to_string(n) + (n == 1 ? " change" : " changes");
      } else {
          display_result = "Edit file: " + path;
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
      display_result = result;
    } else {
      result = "Error: No command provided to exec_shell";
    }
  } else if (tool_name == "web_search") {
    string query = extract_string_arg_bounded(tool_call, "query");
    clean_files.clear();
    if (!query.empty()) {
      NetworkTools net;
      result = net.web_search(query);
      display_result = "Web search: \"" + query + "\"";
      // Count results
      int n_results = 0;
      size_t wp = 0;
      while ((wp = result.find("Title:", wp)) != string::npos) {
          n_results++;
          wp += 6;
      }
      display_result += ": " + to_string(n_results) + (n_results == 1 ? " result" : " results");
    } else {
      result = "Error: No query provided to web_search";
    }
  } else {
    result = "Error: Unknown tool call";
  }

  return result;
}

string sanitize(string text) {
    escape_parameter_tags(text);
    return text;
}
