#ifndef OUTPUT_H
#define OUTPUT_H

#include <string>
#include <cstdarg>
#include <sstream>
#include <iostream>

using namespace std;

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
void stream(const string& raw_token);
void stream_tool_result(const string& html);
void stream_html(const string& html);
void stream_speed(const string& speed_text);
void stream_think(const string& text);
void clear_viewer();

// --- Console Output Helpers ---
template<typename... Args>
void format_and_print(Args&&... args) {
    ostringstream oss;
    ((oss << forward<Args>(args)), ...);
    cout << oss.str();
}

#define message(...) format_and_print(__VA_ARGS__)

template<typename... Args>
void console(Args&&... args) {
  if (should_output_to_stdout())
    ((cout << forward<Args>(args)), ...);
}

// Special function for think block output - outputs to stdout when enabled (modes 1 and 3).
template<typename... Args>
void console_think(Args&&... args) {
  if (should_output_to_stdout()) {
    cout << "\033[0m";  // Reset colors - thinking should NOT be blue
    ((cout << forward<Args>(args)), ...);
  }
}

static inline void consoleFlush() {
    if (should_output_to_stdout()) cout.flush();
}

static inline void consoleThinkFlush() {
    if (should_output_think_blocks()) cout.flush();
}

#endif // OUTPUT_H
