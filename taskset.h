#ifndef TASKSET_H
#define TASKSET_H

#include <string>
#include <vector>
#include <set>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <unistd.h>

// ============================================================================
// Core detection and taskset helpers
//
// Detects P/E cores on Intel hybrid CPUs with a single shell command that
// reads /sys/.../topology/thread_siblings_list per-CPU.  P-cores have
// hyperthreading (siblings like "0-1"), E-cores are single-threaded ("16").
//
// Environment variables:
//   LIM_TASKSET="P_CORES:E_CORES"
//     Override auto-detection with explicit core masks.
//     Examples:
//       export LIM_TASKSET="0-15:16-23"   // classic i9-12900K layout
//       export LIM_TASKSET="0-7:"         // P-cores 0-7, no E-core pinning
//       export LIM_TASKSET=":8-15"        // no P-core pinning, E-cores 8-15
//       export LIM_TASKSET="::"           // disable all taskset pinning
//
//   LIM_TASKSET_CMD="<command> <args>"
//     Override the pinning command. Default is "taskset -c".
//     On macOS, you can install numactl via Homebrew and use:
//       export LIM_TASKSET="0-3:4-7"
//       export LIM_TASKSET_CMD="numactl --cpunodebind"
//     Or write a wrapper script and point to it:
//       export LIM_TASKSET_CMD="/path/to/pin_cores.sh"
//
//     If the command doesn't exist on $PATH, pinning is silently skipped.
// ============================================================================

namespace Taskset {

// ---------------------------------------------------------------------------
// Check whether the pinning command exists on $PATH.  Cached after first call.
// ---------------------------------------------------------------------------
static bool has_pinning_cmd() {
    static int cached = -1; // -1=unknown, 0=no, 1=yes
    if (cached != -1) return cached != 0;

    const char* env = std::getenv("LIM_TASKSET_CMD");
    std::string cmd = env ? env : "taskset";

    // Extract just the command name (first word) for which() check
    size_t space = cmd.find(' ');
    std::string prog = space != std::string::npos ? cmd.substr(0, space) : cmd;

    FILE* fp = popen(("which " + prog).c_str(), "r");
    if (fp) {
        char buf[256];
        cached = fgets(buf, sizeof(buf), fp) != nullptr ? 1 : 0;
        pclose(fp);
    } else {
        cached = 0;
    }
    return cached != 0;
}

// ---------------------------------------------------------------------------
// Get the pinning command string (default "taskset -c", overridable).
// Returns empty string if no command is available.
// ---------------------------------------------------------------------------
static std::string pinning_cmd() {
    if (!has_pinning_cmd()) return "";

    const char* env = std::getenv("LIM_TASKSET_CMD");
    return env ? std::string(env) : "taskset -c";
}

// ---------------------------------------------------------------------------
// Parse "P:E" env var or detect from topology.  Returns true if we have
// a usable core split (either user-specified or auto-detected hybrid).
// ---------------------------------------------------------------------------
static bool get_core_split(std::string& p_mask, std::string& e_mask) {
    const char* env = std::getenv("LIM_TASKSET");

    // --- User override via LIM_TASKSET="P:E" ---
    if (env && std::strlen(env) > 0) {
        std::string spec(env);
        size_t first_colon = spec.find(':');
        if (first_colon == std::string::npos) return false;

        p_mask = spec.substr(0, first_colon);
        e_mask = "";
        size_t second_colon = spec.find(':', first_colon + 1);
        if (second_colon != std::string::npos && second_colon + 1 < spec.size()) {
            e_mask = spec.substr(second_colon + 1);
        } else if (first_colon + 1 < spec.size() && second_colon == std::string::npos) {
            e_mask = spec.substr(first_colon + 1);
        }

        // Trim whitespace
        auto trim = [](std::string& s) {
            size_t a = s.find_first_not_of(" \t");
            size_t b = s.find_last_not_of(" \t");
            if (a == std::string::npos) { s.clear(); }
            else s = s.substr(a, b - a + 1);
        };
        trim(p_mask);
        trim(e_mask);

        return true;
    }

    // --- Auto-detect: read thread_siblings_list from sysfs ---
    // P-cores have hyperthreading (siblings contain "-"), E-cores don't.
    // "shopt -s nullglob" ensures the loop body never runs if the glob doesn't match
    // (e.g., on macOS where /sys/devices/system/cpu/ doesn't exist).
    FILE* fp = popen(
        "bash -c '"
        "shopt -s nullglob; "
        "for d in /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list; do "
        "  n=$(basename $(dirname $(dirname $d)) | sed \"s/cpu//\"); "
        "  s=$(cat $d 2>/dev/null); "
        "  echo \"$n:$s\"; "
        "done'",
        "r"
    );
    if (!fp) return false;

    std::vector<int> p_cores, e_cores;
    char line[128];
    while (fgets(line, sizeof(line), fp)) {
        // Format: "cpu_num:sibling_list"  e.g. "0:0-1" or "16:16"
        int id = -1;
        std::string siblings(line);
        size_t cp = siblings.find(':');
        if (cp == std::string::npos) continue;

        id = std::atoi(siblings.substr(0, cp).c_str());
        std::string sib = siblings.substr(cp + 1);
        // Trim trailing newline
        while (!sib.empty() && (sib.back() == '\n' || sib.back() == '\r')) sib.pop_back();

        // Reject malformed lines (e.g., from a glob that didn't match: "[0-9]*:")
        std::string cpu_str = siblings.substr(0, cp);
        if (cpu_str.find_first_not_of("0123456789") != std::string::npos) continue;
        if (id < 0) continue;

        // If siblings contain a dash, this CPU has hyperthreading -> P-core
        if (sib.find('-') != std::string::npos) {
            p_cores.push_back(id);
        } else {
            e_cores.push_back(id);
        }
    }
    pclose(fp);

    // Only report hybrid if we found both types
    if (!p_cores.empty() && !e_cores.empty()) {
        std::sort(p_cores.begin(), p_cores.end());
        std::sort(e_cores.begin(), e_cores.end());

        // Build compact masks: "0-15", "16,17,18,...,23" -> "16-23"
        auto to_mask = [](const std::vector<int>& c) -> std::string {
            if (c.empty()) return "";
            std::string m;
            int run = c[0], prev = c[0];
            for (size_t i = 1; i <= c.size(); i++) {
                bool end = (i == c.size()) || (c[i] != prev + 1);
                if (end) {
                    if (!m.empty()) m += ',';
                    if (run == prev) m += std::to_string(run);
                    else if (prev == run + 1) m += std::to_string(run) + ',' + std::to_string(prev);
                    else m += std::to_string(run) + '-' + std::to_string(prev);
                    if (i < c.size()) run = c[i];
                }
                if (i < c.size()) prev = c[i];
            }
            return m;
        };

        p_mask = to_mask(p_cores);
        e_mask = to_mask(e_cores);
        return true;
    }

    // Non-hybrid: no split
    return false;
}

// ---------------------------------------------------------------------------
// Return "<cmd> <mask> " for P-core workloads (heavy compute).
// Returns "" if no pinning is available or desired.
// ---------------------------------------------------------------------------
static std::string p_core_taskset() {
    std::string cmd = pinning_cmd();
    if (cmd.empty()) return "";

    std::string p, e;
    if (get_core_split(p, e) && !p.empty())
        return cmd + " " + p + " ";
    return "";
}

// ---------------------------------------------------------------------------
// Return "<cmd> <mask> " for E-core workloads (light background).
// Returns "" if no pinning is available or desired.
// ---------------------------------------------------------------------------
static std::string e_core_taskset() {
    std::string cmd = pinning_cmd();
    if (cmd.empty()) return "";

    std::string p, e;
    if (get_core_split(p, e) && !e.empty())
        return cmd + " " + e + " ";
    return "";
}

// ---------------------------------------------------------------------------
// Return the number of physical cores (no hyperthreading duplicates).
// On hybrid CPUs: count of P-core HT-siblings + E-cores.
// On non-hybrid HT CPUs: nproc / 2.
// Falls back to sysconf(_SC_NPROCESSORS_ONLN) if detection fails.
// ---------------------------------------------------------------------------
static int physical_core_count() {
    FILE* fp = popen(
        "bash -c '"
        "shopt -s nullglob; "
        "for d in /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list; do "
        "  n=$(basename $(dirname $(dirname $d)) | sed \"s/cpu//\"); "
        "  s=$(cat $d 2>/dev/null); "
        "  echo \"$n:$s\"; "
        "done'",
        "r"
    );
    if (!fp) return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));

    std::vector<int> ht_logical, non_ht;
    char line[128];
    while (fgets(line, sizeof(line), fp)) {
        std::string siblings(line);
        size_t cp = siblings.find(':');
        if (cp == std::string::npos) continue;

        std::string cpu_str = siblings.substr(0, cp);
        if (cpu_str.find_first_not_of("0123456789") != std::string::npos) continue;

        std::string sib = siblings.substr(cp + 1);
        while (!sib.empty() && (sib.back() == '\n' || sib.back() == '\r')) sib.pop_back();

        if (sib.find('-') != std::string::npos) {
            // HT core: count the logical CPUs in the range
            size_t dash = sib.find('-');
            int lo = std::atoi(sib.substr(0, dash).c_str());
            int hi = std::atoi(sib.substr(dash + 1).c_str());
            ht_logical.push_back(hi - lo); // number of siblings per physical core
        } else {
            non_ht.push_back(std::atoi(cpu_str.c_str()));
        }
    }
    pclose(fp);

    if (ht_logical.empty() && non_ht.empty()) {
        return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    }

    // Count physical cores
    int physical = 0;
    if (!ht_logical.empty()) {
        // All HT entries share the same sibling count (e.g., 2 for hyperthreading)
        int siblings_per_core = ht_logical[0] + 1; // "0-1" means 2 logical per physical
        physical += static_cast<int>(ht_logical.size()) / siblings_per_core;
    }
    physical += static_cast<int>(non_ht.size());

    return physical > 0 ? physical : static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
}

// ---------------------------------------------------------------------------
// Return the number of physical P-cores (no E-cores, no HT siblings).
// On hybrid CPUs this excludes slow efficiency cores from inference threading.
// On non-hybrid CPUs falls back to physical_core_count().
// ---------------------------------------------------------------------------
static int p_core_thread_count() {
    std::string p, e;
    if (!get_core_split(p, e)) {
        // Non-hybrid: all cores are equal, use physical count
        return physical_core_count();
    }
    // Hybrid: count unique physical P-cores by deduplicating sibling lists.
    // Each unique thread_siblings_list value represents one physical core.
    // P-cores have HT siblings (e.g., "0-1"), E-cores are single-threaded (e.g., "16").
    FILE* fp = popen(
        "bash -c '"
        "shopt -s nullglob; "
        "for d in /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list; do "
        "  s=$(cat $d 2>/dev/null); "
        "  echo \"$s\"; "
        "done'",
        "r"
    );
    if (!fp) return physical_core_count();

    std::set<std::string> unique_siblings;
    char line[128];
    while (fgets(line, sizeof(line), fp)) {
        std::string sib(line);
        while (!sib.empty() && (sib.back() == '\n' || sib.back() == '\r')) sib.pop_back();
        // If siblings contain a dash, this CPU has hyperthreading -> P-core
        if (sib.find('-') != std::string::npos) {
            unique_siblings.insert(sib);
        }
    }
    pclose(fp);

    return !unique_siblings.empty() ? static_cast<int>(unique_siblings.size()) : physical_core_count();
}

// ---------------------------------------------------------------------------
// Log a one-line diagnostic describing the detected topology.
// ---------------------------------------------------------------------------
static void log_core_detection(std::ostream& os) {
    std::string p, e;
    bool hybrid = get_core_split(p, e);
    std::string cmd = pinning_cmd();

    const char* env = std::getenv("LIM_TASKSET");
    if (env && std::strlen(env) > 0)
        os << "LIM_TASKSET=" << env << " (user override)" << std::endl;

    const char* cmd_env = std::getenv("LIM_TASKSET_CMD");
    if (cmd_env && std::strlen(cmd_env) > 0)
        os << "LIM_TASKSET_CMD=" << cmd_env << " (user override)" << std::endl;

    if (hybrid) {
        os << "Core topology: P-cores=[" << p
           << "] E-cores=[" << e
           << "] pinning=" << (cmd.empty() ? "none (no taskset on PATH)" : cmd)
           << std::endl;
    } else {
        int n = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
        os << "Core topology: no hybrid detected, " << n << " online CPU(s)" << std::endl;
    }
}

} // namespace Taskset

#endif // TASKSET_H
