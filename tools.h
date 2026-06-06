#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#include <set>

using namespace std;

extern const string PATH_NEWLINE_ERROR;

bool param_has_newline(const string& s);
string execute_tool_call(const string& tool_call, set<string>& clean_files, string& display_result);
string sanitize(string text);

#endif // TOOLS_H
