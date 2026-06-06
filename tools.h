#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#include <set>

using namespace std;

extern const string PATH_NEWLINE_ERROR;

struct ToolResult {
    string content;          // Text sent to the LLM as feedback
    string display;          // Short summary shown in console/browser
    bool is_error = false;   // Whether the tool call failed
    bool is_expected_error = false;  // Whether failure is expected (e.g., edit mismatch)
};

bool param_has_newline(const string& s);
ToolResult execute_tool_call(const string& tool_call, set<string>& clean_files);
string sanitize(string text);

#endif // TOOLS_H
