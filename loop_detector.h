#ifndef LOOP_DETECTOR_H
#define LOOP_DETECTOR_H

#include <string>
#include <vector>
#include <deque>
#include <map>
#include <functional>
#include <cctype>
#include "tokens.h"

using namespace std;
using namespace Tokens;

// --- SEQUENTIAL INTERVENTION MESSAGES ---
static const string SYSTEM_PROMPT_REMINDER = "Follow the system prompt strictly.";

inline vector<string> loopMessages = {
    "You are in a loop.",
    "You already have this information.",
    "Please proceed.",
    "Continue.",
    "You did this already.",
    "Can we finish?",
    "Let's break out of this loop!"
};
inline int loopMessageIndex = 0;

inline string get_next_loop_message() {
    string msg = loopMessages[loopMessageIndex];
    loopMessageIndex = (loopMessageIndex + 1) % loopMessages.size();
    return SYSTEM_PROMPT_REMINDER + " " + msg;
}

// --- Macro-Loop Detection ---
class LoopDetector {
private:
    deque<size_t> tool_history;
    map<size_t, int> freq_map;  // O(1) occurrence counts, kept in sync with tool_history
    size_t max_window_size;

    string normalize_str(const string& s) const {
        string tool_name;
        size_t fs = s.find(FUNC_START);
        if (fs != string::npos) {
            size_t gt = s.find('>', fs);
            if (gt != string::npos) {
                tool_name = s.substr(fs, gt - fs);
            }
        }

        // Extract parameter values using PARAM_START / PARAM_END constants.
        // Captures semantic intent regardless of whitespace/formatting differences.
        string param_values;
        string pstart(PARAM_START);
        string pend(PARAM_END);
        size_t ps = 0;
        while ((ps = s.find(pstart, ps)) != string::npos) {
            size_t pe = s.find('>', ps);
            if (pe == string::npos) break;
            size_t pc = s.find(pend, pe);
            if (pc == string::npos) break;
            string value = s.substr(pe + 1, pc - pe - 1);
            // Collapse whitespace within each value
            string collapsed;
            bool last_space = true;
            for (char c : value) {
                if (isspace(c)) {
                    if (!last_space) { collapsed += ' '; last_space = true; }
                } else { collapsed += c; last_space = false; }
            }
            param_values += collapsed + "|";
            ps = pc + pend.length();
        }

        // If no parameters were extracted (malformed tags), fall back to
        // stripping whitespace from the entire tool call body to avoid
        // hash collisions between different calls to the same tool.
        if (param_values.empty()) {
            string fallback;
            for (char c : s) { if (!isspace(c)) fallback += c; }
            return fallback;
        }

        return tool_name + ":" + param_values;
    }

    void add_to_map(size_t h) { freq_map[h]++; }
    void remove_from_map(size_t h) { if (--freq_map[h] == 0) freq_map.erase(h); }

public:
    LoopDetector(size_t window_size = 15) : max_window_size(window_size) {}

    // O(1): Block if this command has appeared >= 2 times in the window.
    // Catches direct repeats (A->A) and cycles (A->B->C->D->E->A).
    bool would_repeat(const string& tool_call) const {
        string norm_tool = normalize_str(tool_call);
        size_t tool_hash = hash<string>{}(norm_tool);
        auto it = freq_map.find(tool_hash);
        return (it != freq_map.end() && it->second >= 2);
    }

    // O(1): Record a tool call; return true if it now appears >= 3 times.
    bool record_and_check(const string& tool_call) {
        string norm_tool = normalize_str(tool_call);
        size_t tool_hash = hash<string>{}(norm_tool);

        tool_history.push_back(tool_hash);
        add_to_map(tool_hash);

        if (tool_history.size() > max_window_size) {
            remove_from_map(tool_history.front());
            tool_history.pop_front();
        }

        return (freq_map[tool_hash] >= 3);
    }

    // O(1): Count only CONSECUTIVE occurrences of the most recent hash.
    int get_loop_strikes() const {
        if (tool_history.empty()) return 0;
        size_t last = tool_history.back();
        int count = 0;
        for (auto it = tool_history.rbegin(); it != tool_history.rend(); ++it) {
            if (*it == last) count++;
            else break;
        }
        return count;
    }

    void clear_history() {
        tool_history.clear();
        freq_map.clear();
        loopMessageIndex = 0;
    }
};

#endif // LOOP_DETECTOR_H
