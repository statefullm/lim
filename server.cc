
#include "server.h"
#include "output.h"
#include "filesystem.h"
#include "taskset.h"
#include "network.h"
#include <sys/stat.h>
#include <sys/inotify.h>
#include <sys/wait.h>
#include <dirent.h>
#include <poll.h>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <unistd.h>

using namespace std;

// --- LIM Server Process Management ---
static const char* INOTIFY_DIR = "/tmp";
static const char* SERVER_READY_PATH = "/tmp/lim.server_ready";
static const char* BROWSER_READY_PATH = "/tmp/lim.browser_ready";
static const char* SERVER_PID_PATH = "/tmp/lim.server.pid";
static bool marker_file_valid(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) return false;
    int ch = fgetc(fp);
    fclose(fp);
    return ch != EOF;
}

static bool wait_for_file(const char* dir, const char* filename, bool check_existing, long timeout_ms = -1) {
    int fd = inotify_init();
    if (fd < 0) return false;

    int wd = inotify_add_watch(fd, dir, IN_CREATE | IN_MODIFY);
    if (wd < 0) {
        close(fd);
        return false;
    }

    string path = string(dir) + "/" + filename;

    // The existing-file check must run AFTER the watch is armed: a marker
    // created between an earlier check and the watch would fire no event and
    // be missed.  With the watch armed first, every creation is seen by one of
    // the two (this check, or the IN_CREATE/IN_MODIFY event below).
    if (check_existing && marker_file_valid(path.c_str())) {
        inotify_rm_watch(fd, wd);
        close(fd);
        return true;
    }

    // Deadline-bounded: startup gates use a finite timeout (the default caller
    // passes 15 s); the interactive browser prompt waits unbounded (-1) since
    // the user is expected to Ctrl+C out of it.  Poll in slices rather than
    // blocking in read() so the deadline can be honored.
    long slice = (timeout_ms >= 0) ? 250 : -1;
    auto deadline = chrono::steady_clock::now() + chrono::milliseconds(timeout_ms);

    struct pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;

    while (true) {
        if (timeout_ms >= 0) {
            long remaining = chrono::duration_cast<chrono::milliseconds>(
                deadline - chrono::steady_clock::now()).count();
            if (remaining < slice) slice = remaining;
        }
        if (timeout_ms >= 0 && slice <= 0) {
            // Deadline passed.  (timeout_ms == -1 waits unbounded: slice stays
            // -1 and poll() below blocks indefinitely.)
            inotify_rm_watch(fd, wd);
            close(fd);
            return false;
        }

        int pr = poll(&pfd, 1, (int)slice);
        if (pr < 0) {
            if (errno == EINTR) {
                inotify_rm_watch(fd, wd);
                close(fd);
                return false;
            }
            inotify_rm_watch(fd, wd);
            close(fd);
            return false;
        }
        if (pr == 0) continue;  // Slice expired with no events; recheck the deadline.

        char buf[4096];
        ssize_t len = read(fd, buf, sizeof(buf));
        if (len < 0) {
            if (errno == EINTR) {
                inotify_rm_watch(fd, wd);
                close(fd);
                return false;
            }
            inotify_rm_watch(fd, wd);
            close(fd);
            return false;
        }

        for (char* p = buf; p < buf + len; ) {
            struct inotify_event* event = (struct inotify_event*)p;
            if (event->len) {
                string name(event->name);
                if (name == filename && marker_file_valid(path.c_str())) {
                    inotify_rm_watch(fd, wd);
                    close(fd);
                    return true;
                }
            }
            p += sizeof(struct inotify_event) + event->len;
        }
    }
}

// Read the server PID from the PID file. Returns 0 if unreadable or invalid.
static pid_t read_server_pid() {
    FILE* fp = fopen(SERVER_PID_PATH, "r");
    if (!fp) return 0;
    char buf[32];
    pid_t pid = 0;
    if (fgets(buf, sizeof(buf), fp)) {
        pid = strtol(buf, nullptr, 10);
        if (pid <= 1) pid = 0;
    }
    fclose(fp);
    return pid;
}



// True if pid is alive and is our limServer.py.  Two criteria, both required:
// argv[0] is a python interpreter, AND one NUL-separated cmdline ARG is of the
// form <path>/limServer.py.  The server runs as "python3 <LIM_CONFIG_DIR>/
// limServer.py" (a shebang launch puts the interpreter in argv[0] as well),
// and LIM_CONFIG_DIR is never empty for a running server (startup refuses it),
// so the slash is always present.  The arg match is exact (an arg ENDS with
// /limServer.py), not an arbitrary substring, so lim's own restore args or a
// tool's path that merely mentions the script never match.  The argv[0]
// requirement excludes editors and other tools that have the script OPEN
// (vim/less <path>/limServer.py): a cmdline match alone must never make a
// process a kill candidate.  Self is excluded outright.  Single-user system:
// any live process on our port that fails this check is a misconfiguration,
// not a competing session -- we report it, we never kill it.
static bool is_our_server_pid(pid_t pid) {
    if (pid <= 1 || pid == getpid() || kill(pid, 0) != 0) return false;
    FILE* fp = fopen(("/proc/" + to_string(pid) + "/cmdline").c_str(), "r");
    if (!fp) return false;
    char buf[1024] = {};
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    // argv[0] (the first NUL-terminated arg) must be a python interpreter:
    // its basename starts with "python" (python, python3, python3.11, ...).
    // An empty cmdline (e.g. a zombie) has no argv[0] and fails here.
    {
        size_t j = 0;
        while (j < n && buf[j] != '\0') j;
        const char* base = buf;
        for (size_t k = 0; k < j; k++)
            if (buf[k] == '/') base = buf + k + 1;
        if (strncmp(base, "python", 6) != 0) return false;
    }
    static const char needle[] = "limServer.py";
    const size_t nlen = sizeof(needle) - 1;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j < n && buf[j] != '\0') j++;  // arg end (exclusive)
        size_t arglen = j - i;
        if (arglen > nlen) {
            const char* start = buf + i + arglen - nlen;
            if (start[-1] == '/' && memcmp(start, needle, nlen) == 0) {
                return true;
            }
        }
        i = j + 1;  // step past the NUL separator
    }
    return false;
}

// Scan /proc for a live process whose cmdline mentions limServer.py.  Fallback
// for when the PID file is gone (e.g. /tmp cleaned) while the server itself
// still runs.  Returns 0 when none is found; any result passes
// is_our_server_pid (a zombie's empty cmdline can never match).
static pid_t find_our_server_pid() {
    DIR* dir = opendir("/proc");
    if (!dir) return 0;
    pid_t found = 0;
    struct dirent* ent;
    while (found == 0 && (ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] < '0' || ent->d_name[0] > '9') continue;
        pid_t candidate = (pid_t)strtol(ent->d_name, nullptr, 10);
        if (candidate == getpid()) continue;  // our own argv may name the script
        if (is_our_server_pid(candidate)) found = candidate;
    }
    closedir(dir);
    return found;
}

// The pid of a verifiably-live limServer: PID file first, /proc cmdline scan
// as fallback.  Returns 0 when none can be identified.
static pid_t resolve_our_server_pid() {
    pid_t pid = read_server_pid();
    if (is_our_server_pid(pid)) return pid;
    return find_our_server_pid();
}

// True if pid has an open fd on the same FIFO inode lim writes to (pipe_fd),
// i.e. lim's writes actually reach this process.  The comparison is against
// pipe_fd rather than the path on purpose: the node can be deleted while lim
// and the server both still hold the old inode, and output still flows then.
// Only a node REPLACED under a running server (lim re-mkfifo'd a fresh node)
// breaks the link.  Falls back to the path when pipe_fd is unavailable
// (closed after a failed write, pre-reopen).  If the comparison can't be made
// (no /proc access), assume yes -- never kill on a guess.
static bool pid_serves_our_fifo(pid_t pid) {
    struct stat want{};
    bool have = false;
    if (pipe_fd >= 0 && fstat(pipe_fd, &want) == 0 && S_ISFIFO(want.st_mode)) {
        have = true;
    } else if (stat(FIFO_PATH, &want) == 0 && S_ISFIFO(want.st_mode)) {
        have = true;
    }
    if (!have) return true;

    char dirpath[64];
    snprintf(dirpath, sizeof(dirpath), "/proc/%d/fd", (int)pid);
    DIR* dir = opendir(dirpath);
    if (!dir) return true;
    bool ok = false;
    struct dirent* ent;
    while (!ok && (ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        char link[512];
        snprintf(link, sizeof(link), "%s/%s", dirpath, ent->d_name);
        // stat(), NOT lstat(): /proc/<pid>/fd/N entries are magic links, and
        // lstat reports the link itself (always S_IFLNK, never S_IFIFO) --
        // lstat here matched no fd of any process, so every healthy server
        // looked stale and was replaced at every restart (the browser-reload
        // regression).  stat() follows the link to the target's st_dev/
        // st_ino; a fd whose target has been deleted fails (ENOENT) and
        // simply does not match.
        struct stat lst{};
        if (stat(link, &lst) == 0 && S_ISFIFO(lst.st_mode) &&
            lst.st_dev == want.st_dev && lst.st_ino == want.st_ino) {
            ok = true;
        }
    }
    closedir(dir);
    return ok;
}

// True if pid is alive: kill(pid, 0) succeeds and the /proc state is not a
// zombie.  kill() alone is not enough -- it also succeeds on unreaped zombies
// that hold nothing.  The state char is the field after the space following the
// LAST ')' of the (comm) field, since comm may itself contain spaces and
// parentheses.  If the state can't be read, assume alive (the kill() result
// stands).  A zombie counts as gone.
static bool pid_alive(pid_t pid) {
    if (pid <= 1 || kill(pid, 0) != 0) return false;
    FILE* fp = fopen(("/proc/" + to_string(pid) + "/stat").c_str(), "r");
    if (!fp) return true;
    char buf[1024] = {};
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    (void)n;
    fclose(fp);
    const char* close = strrchr(buf, ')');
    if (close) {
        size_t i = (size_t)(close - buf) + 2;  // past ')' and the following space
        if (i < sizeof(buf) && close[1] == ' ' && buf[i]) return buf[i] != 'Z';
    }
    return true;
}

// SIGKILL a verified-ours server and wait (bounded: 100 ms sleep slices, up to
// 3 s) until the pid is gone.  A zombie counts as gone (pid_alive), so the
// wait cannot spin on an unreaped zombie.  Used ONLY for stale servers: one
// left bound to a previous LIM_PORT, or one left reading a deleted/replaced
// FIFO node (it holds an inode lim no longer writes to).  Returns true when
// gone.
static bool kill_our_server(pid_t pid) {
    if (pid <= 1) return false;
    kill(pid, SIGKILL);
    auto deadline = chrono::steady_clock::now() + chrono::seconds(3);
    while (chrono::steady_clock::now() < deadline) {
        if (!pid_alive(pid)) return true;
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    return !pid_alive(pid);
}

bool is_lim_server_running() {
    int port = get_server_port();

    // Robust approach: try to bind the port ourselves. If it succeeds, nobody
    // is listening; if EADDRINUSE, someone is. This avoids spawning subprocesses
    // (ss/netstat/lsof) and works consistently across all Linux distributions.
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;  // Can't create socket -- assume not running

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    bool in_use = (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0 && errno == EADDRINUSE);
    close(sock);
    return in_use;
}

// True when a server is listening AND verifiably reading the same FIFO inode
// lim writes to.  Stronger than is_lim_server_running(): a server left on a
// deleted or replaced FIFO node (manual unlink, /tmp cleanup) still binds the
// port but receives nothing.  /reset uses this to decide whether to (re)start.
bool is_lim_server_healthy() {
    if (!is_lim_server_running()) return false;
    pid_t pid = resolve_our_server_pid();
    if (pid <= 1) return false;
    return pid_serves_our_fifo(pid);
}

bool g_lim_server_reused = false;

ServerStart start_lim_server_if_needed() {
    g_lim_server_reused = false;

    if (is_lim_server_running()) {
        pid_t pid = resolve_our_server_pid();
        if (pid > 1) {
            if (pid_serves_our_fifo(pid)) {
                // The normal path: the persistent server survived the previous
                // session.  Quiet on purpose -- the exceptional branches below
                // are the ones that log.
                g_lim_server_reused = true;
                return ServerStart::Reused;
            }
            // Alive and ours, but reading a different FIFO inode (the node was
            // deleted or replaced while it ran).  lim's writes would go nowhere,
            // and the port is still bound, so neither a relaunch nor /reset
            // could ever recover it on its own: replace it so the new server
            // attaches to the node lim actually writes to.
            log_diagnostic("Persistent limServer (pid " + to_string(pid) +
                ") is alive but not reading the FIFO node lim writes to; replacing it...");
            if (!kill_our_server(pid)) {
                log_diagnostic("ERROR: limServer (pid " + to_string(pid) +
                    ") did not exit within 3 s. Refusing to start a second server on the shared FIFO. "
                    "Kill the process manually; browser output is enabled, but inoperative while the old server is alive.", true);
                return ServerStart::Error;
            }
            // Fall through: the port is now free; start a fresh server below.
        } else {
            log_diagnostic("Port " + to_string(get_server_port()) +
                " is in use by a process that is not our limServer. Kill that process or set LIM_PORT; "
                "browser output is enabled, but with no server listening it is inoperative until the port is free.", true);
            return ServerStart::Error;
        }
    }

    // Port is free.  A verified-ours server may still be bound to a previous
    // LIM_PORT and holding the shared FIFO inode: two servers on one FIFO would
    // split the stream, so tear the stale one down before starting.  The
    // FIFO-holds check is this branch's own criterion: a candidate that does
    // not hold the node lim writes to cannot be the stale server described
    // here (there is no shared stream to split), and is never killed on a
    // cmdline match alone.
    pid_t stale = resolve_our_server_pid();
    if (stale > 1) {
        if (pid_serves_our_fifo(stale)) {
            log_diagnostic("Stopping stale limServer from a previous LIM_PORT (pid " + to_string(stale) + ")");
            if (!kill_our_server(stale)) {
                log_diagnostic("ERROR: stale limServer (pid " + to_string(stale) +
                    ") did not exit within 3 s. Refusing to start a second server on the shared FIFO. "
                    "Kill the process manually or reboot; browser output is enabled, but inoperative while the old server is alive.", true);
                return ServerStart::Error;
            }
        } else {
            log_diagnostic("limServer-like process (pid " + to_string(stale) +
                ") does not hold the FIFO node lim writes to; leaving it alone and starting fresh.");
        }
    }

    if (LIM_CONFIG_DIR.empty()) {
        log_diagnostic("ERROR: LIM_CONFIG_DIR is not set. Cannot start limServer.", true);
        return ServerStart::Error;
    }

    unlink(SERVER_READY_PATH);
    unlink(BROWSER_READY_PATH);
    unlink(SERVER_PID_PATH);
    // The FIFO is deliberately NOT unlinked: it persists across sessions and the
    // persistent server keeps its fd open on this inode.

    pid_t pid = fork();
    if (pid == 0) {
        // Middle: fork once more so the server is reparented to init (PID 1)
        // the moment we _exit.  If the server later dies while lim is running
        // it is reaped by init instead of lingering as a zombie of lim.
        if (fork() > 0) _exit(0);
        setsid();  // Own session: survives lim's exit and logout (no controlling
                   // terminal, no SIGHUP); not in lim's process group, so lim
                   // must never kill(-pid) or waitpid() it.
        // Quote the script path: LIM_CONFIG_DIR may contain spaces.  The
        // taskset fragment is user-supplied shell (LIM_TASKSET_CMD) and stays
        // unquoted by design.
        string cmd = "exec " + Taskset::e_core_taskset() + "python3 \"" + LIM_CONFIG_DIR + "/limServer.py\"";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        _exit(1);  // NOT exit(): would run lim's atexit handlers in the child.
    }
    if (pid < 0) {
        log_diagnostic("ERROR: fork failed while starting limServer.", true);
        return ServerStart::Error;
    }
    // Reap the middle, which _exit()s immediately.  Bounded: if its own fork()
    // failed it execs the server instead and outlives the loop -- then it
    // stays a child of lim (the pre-double-fork behavior), on a pid-exhaustion
    // failure where everything else is broken anyway.
    for (int i = 0; i < 100; i++) {
        if (waitpid(pid, nullptr, WNOHANG) != 0) break;
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    // The server itself is deliberately not tracked: it outlives lim (no kill
    // at exit) and is addressable only by pid (PID file / /proc scan).
    return ServerStart::Started;
}

bool wait_for_server_ready(long timeout_ms) {
    return wait_for_file(INOTIFY_DIR, "lim.server_ready", true, timeout_ms);
}

// --- Browser Connection Status Checking ---

bool check_browser_connected() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(get_server_port());
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return false;
    }

    const char* request = "GET /status HTTP/1.0\r\nHost: localhost\r\n\r\n";
    write(sock, request, strlen(request));

    char buffer[512];
    bool connected = false;
    ssize_t bytes_read = read(sock, buffer, sizeof(buffer) - 1);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        if (strstr(buffer, "\"connected\": true") != nullptr ||
            strstr(buffer, "\"connected\":true") != nullptr) {
            connected = true;
        }
    } else {
        close(sock);
        return false;
    }

    close(sock);
    return connected;
}

int get_server_port() {
    const char* env = getenv("LIM_PORT");
    if (env != nullptr && strlen(env) > 0) {
        char* endp = nullptr;
        long val = strtol(env, &endp, 10);
        if (*endp == '\0' && val > 0 && val < 65536) return static_cast<int>(val);
    }
    return 8765;
}

string get_hostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        string h(hostname);
        size_t dot_pos = h.find('.');
        if (dot_pos != string::npos) {
            h = h.substr(0, dot_pos);
        }
        return h;
    }
    return "localhost";
}

string get_viewer_url() {
    const char* env = getenv("LIM_VIEWER_URL");
    if (env != nullptr && strlen(env) > 0) {
        return string(env);
    }
    return "http://" + get_hostname() + ":" + std::to_string(get_server_port()) + "/viewer.html";
}

void disable_browser_output() {
    if (!g_browser_warning_suppressed) {
        message("\n\033[1;35m[Browser output disabled]\033[0m\n");
    }
    g_browser_warning_suppressed = true;
}

bool prompt_for_browser_connection() {
    message("Load this URL in your browser or reload");
    message(":\n");
    message("  \033[1;35m" + get_viewer_url() + "\033[0m\n");

    // check_existing=true: the marker only exists while a client is connected
    // (the server unlinks it on last disconnect), so an existing marker means
    // the browser connected in the window after the HTTP check above -- return
    // immediately instead of waiting for an event that will never come.
    if (wait_for_file(INOTIFY_DIR, "lim.browser_ready", true)) {
        return true;
    }

    if (get_output_mode() == 3) {
        message("\n\033[1;35m[Interrupted, but stdout output is enabled. Proceeding.]\033[0m\n");
        return true;
    }

    disable_browser_output();
    return false;
}
