#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#include <vector>
#include <set>
#include <map>

extern const std::string PATH_NEWLINE_ERROR;

struct ToolResult {
  std::string content;          // Text sent to the LLM as feedback
  std::string display;          // Short summary shown in console/browser
  bool is_error = false;   // Whether the tool call failed
  bool is_expected_error = false;  // Whether failure is expected (e.g., edit mismatch)
  bool is_mutating = false;  // Whether the tool modifies files
  bool recognized = true;  // Whether the tool name was recognized
  bool params_valid = true;  // Whether required parameters are present
  bool malformed_xml = false;  // Structural XML issue (e.g., missing PARAM_END causing param bleed)
  bool path_inferred = false;  // Path was auto-inferred from preceding search_file (suppress strike)
  std::string parsed_tool_name; // The tool name as extracted from the XML tag (for diagnostics)
  std::vector<std::string> missing_params; // List of missing required param names (for diagnostics)
};

#include "session.h"

bool param_has_newline(const std::string& s);
ToolResult execute_tool_call(const std::string& tool_call, SessionState& state);
// Lightweight validation: check if a raw tool call XML has a recognized name and all required params.
// Does NOT execute the tool; returns true only if the call is structurally valid.
bool validate_tool_call(const std::string& tool_call);


#endif // TOOLS_H
