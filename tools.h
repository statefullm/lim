#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#include <vector>
#include <set>
#include <map>

using namespace std;

extern const string PATH_NEWLINE_ERROR;

struct ToolResult {
    string content;          // Text sent to the LLM as feedback
    string display;          // Short summary shown in console/browser
    bool is_error = false;   // Whether the tool call failed
    bool is_expected_error = false;  // Whether failure is expected (e.g., edit mismatch)
    bool is_mutating = false;  // Whether the tool modifies files
    bool recognized = true;  // Whether the tool name was recognized
    bool params_valid = true;  // Whether required parameters are present
    string parsed_tool_name; // The tool name as extracted from the XML tag (for diagnostics)
    vector<string> missing_params; // List of missing required param names (for diagnostics)
};

bool param_has_newline(const string& s);
ToolResult execute_tool_call(const string& tool_call, map<string, string>& file_cache);


#endif // TOOLS_H
