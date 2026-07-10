#ifndef OUTPUT_H
#define OUTPUT_H

#include <string>
#include <cstdarg>
#include <sstream>
#include <iostream>

// --- Segment Constants ---
// Segment prefix characters for viewer.html segment-based rendering.
extern const char SEG_LLM_TEXT;  // LLM-generated text (rendered through marked)
extern const char SEG_HTML;      // Any other raw HTML (tool results, user input, dividers)
extern const char SEG_SPEED;     // Speed/context diagnostic for status bar
extern const char SEG_THINK;     // Think/reasoning block content (scrollable box with KaTeX)
extern const char SEG_TURN_END;  // Generation complete signal for viewer

// --- FIFO / Pipe Management ---
extern int pipe_fd;
extern const char* FIFO_PATH;

// --- Output Mode Control ---
extern bool g_browser_warning_suppressed;

int get_output_mode();
bool should_output_to_stdout();
bool should_output_to_browser();
bool should_output_think_blocks();

// --- Initialization ---
void init_output_stream();

// --- Pipe Writing ---
void pipe_write(const char* data, size_t len);

// --- Streaming Functions ---
void stream(const std::string& raw_token);
void stream_tool_result(const std::string& html);
void stream_html(const std::string& html);
void stream_speed(const std::string& speed_text);
void stream_think(const std::string& text);

// --- Console Output Helpers ---
// Track whether stdout ended with a newline, so callers can ensure clean line breaks.
extern bool g_stdout_ended_with_newline;

template<typename... Args>
void format_and_print(Args&&... args) {
  std::ostringstream oss;
  ((oss << std::forward<Args>(args)), ...);
  std::cout << oss.str();
}

#define message(...) format_and_print(__VA_ARGS__)

template<typename... Args>
void console(Args&&... args) {
  if (should_output_to_stdout())
    ((std::cout << std::forward<Args>(args)), ...);
}

// Special function for think block output - outputs to stdout when enabled (modes 1 and 3).
template<typename... Args>
void console_think(Args&&... args) {
  if (should_output_to_stdout()) {
    std::cout << "\033[0m";  // Reset colors - thinking should NOT be blue
    ((std::cout << std::forward<Args>(args)), ...);
  }
}

static inline void consoleFlush() {
  if (should_output_to_stdout()) std::cout.flush();
}

static inline void consoleThinkFlush() {
  if (should_output_think_blocks()) std::cout.flush();
}

// Ensure stdout cursor is at the start of a new line.
// No-op if stdout already ended with '\n'; prints exactly one '\n' otherwise.
static inline void consoleEnsureNewline() {
  if (should_output_to_stdout() && !g_stdout_ended_with_newline) {
    std::cout << "\n";
    g_stdout_ended_with_newline = true;
    std::cout.flush();
  }
}

// Update the newline-tracking flag after a direct cout write.
static inline void consoleMarkNewline(bool ended_with_nl) {
  g_stdout_ended_with_newline = ended_with_nl;
}

// --- Diagnostic Output ---
void diag(const std::string& msg, const char* color);

#endif // OUTPUT_H
