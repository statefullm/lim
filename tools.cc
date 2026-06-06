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

// Tool metadata: required parameters per tool.
struct ToolSpec { string name; vector<string> params; };
static const vector<ToolSpec> tool_specs = {
    {"read_files",  {"paths"}},
    {"search_file", {"path"}},
    {"write_file",  {"path", "content"}},
    {"edit_file",   {"path", "old", "new"}},
    {"exec_shell",  {"command"}},
    {"web_search",  {"query"}}
};

static bool check_params(const string& tool_name, const string& tool_call) {
    for (const auto& spec : tool_specs) {
        if (spec.name == tool_name) {
            for (const auto& param : spec.params) {
                string key = string(PARAM_START) + param + ">";
                if (tool_call.find(key) == string::npos) return false;
            }
            return true;
        }
    }
    return false;  // unknown tool
}

static bool is_known_tool(const string& name) {
    for (const auto& spec : tool_specs) {
        if (spec.name == name) return true;
    }
    return false;
}

ToolResult execute_tool_call(const string& tool_call, set<string>& clean_files) {
  ToolResult out;
  string display_result = "";
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
  if (stop_generation) { out.content = "[Tool interrupted by user]"; return out; }

  // Validate: is this a recognized tool with required parameters?
  out.recognized = is_known_tool(tool_name);
  out.params_valid = check_params(tool_name, tool_call);

  if (!out.recognized) {
      out.content = "Error: Unknown tool call";
      out.is_error = true;
      return out;
  }
  if (!out.params_valid) {
      out.content = "System Error: Malformed tool call. Required parameter tags are missing.";
      out.is_error = true;
      return out;
  }

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
    if (param_has_newline(path)) { out.content = PATH_NEWLINE_ERROR; out.is_error = true; return out; }
    string text = extract_string_arg_bounded(tool_call, "text");
    string begin_str = extract_string_arg_bounded(tool_call, "begin");
    string end_str = extract_string_arg_bounded(tool_call, "end");
    if (!path.empty()) {
      FileSystemTools fs;
      auto result_map = fs.search_file(path, text, begin_str, end_str);

      string r_content, r_error, r_display;
      if (result_map.count("content")) r_content = result_map.at("content");
      if (result_map.count("error")) r_error = result_map.at("error");
      if (result_map.count("display")) r_display = result_map.at("display");

      // Build the LLM-facing result string
      if (!r_error.empty()) {
          result = "Error: " + r_error;
          out.is_error = true;
      } else if (r_content.empty()) {
          result = "No occurrences found for text.";
      } else {
          result = r_content;
      }

      display_result = r_display;
    } else {
      result = "Error: path is required for search_file";
    }
  } else if (tool_name == "write_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) { out.content = PATH_NEWLINE_ERROR; out.is_error = true; return out; }
    string content = extract_string_arg_bounded(tool_call, "content");
    clean_files.erase(path);
    out.is_mutating = true;
    if (!path.empty()) {
      FileSystemTools fs;
      auto result_map = fs.write_file(path, content);
      string r_status, r_error;
      if (result_map.count("status")) r_status = result_map.at("status");
      if (result_map.count("error")) r_error = result_map.at("error");
      result = "Status: " + r_status;
      if (!r_error.empty()) {
          result += ", Error: " + r_error;
          out.is_error = true;
          display_result = "Write file: " + path + ": " + r_error;
      } else {
          display_result = "Write file: " + path + ": " + to_string(content.length()) + " bytes";
      }
    } else {
      result = "Error: No path provided to write_file";
    }
  } else if (tool_name == "edit_file") {
    string path = extract_string_arg_bounded(tool_call, "path");
    if (param_has_newline(path)) { out.content = PATH_NEWLINE_ERROR; out.is_error = true; return out; }
    string old_str = extract_string_arg_bounded(tool_call, "old");
    string new_str = extract_string_arg_bounded(tool_call, "new");
    clean_files.erase(path);
    out.is_mutating = true;
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
      if (!r_error.empty()) {
          result += ", Error: " + r_error;
          out.is_error = true;
          // Check for expected edit errors (mismatch)
          if (r_error.find("not found") != string::npos ||
              r_error.find("exact match") != string::npos) {
              out.is_expected_error = true;
          }
      }
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

  out.content = result;
  out.display = display_result;
  if (out.is_error) return out;
  // Detect error from content string as fallback for tools that set result but not is_error
  if (result.find("Error:") != string::npos) {
      out.is_error = true;
  }
  return out;
}



