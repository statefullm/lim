# LLLM — Stateful, O(1) Local LLM Controller

**LLLM** is a C++ terminal-based LLM controller built on [llama.cpp](https://github.com/ggerganov/llama.cpp). It provides a persistent, stateful session with **true O(1) history injection** via a continuously appended KV-cache — no context transmission or re-tokenization on every turn.

## Why O(1)? The Fundamental Difference

Every mainstream chatbot (and most local LLM frontends) follows the *reprocess-every-turn* model: each time you send a message, the **entire conversation history** is re-transmitted in full and re-tokenized from scratch. Cost grows **quadratically with conversation length**.

LLLM takes full advantage of what a single-user local setup affords:

- **The KV-cache is never discarded.** Each turn's tokens are simply appended to `n_past`. The model continues generating from exactly where it left off.
- **New user input costs O(input tokens)** — not O(total history). Even after hundreds of turns, each new message processes only its own tokens plus the delta since last turn.
- **No re-tokenization overhead.** The system prompt is tokenized once at startup and lives in the cache forever.

This means long-running coding sessions stay fast regardless of how many tool calls, file edits, or searches accumulate.

---

## Architecture at a Glance

```
+-------------+     +---------------+     +---------------+
|  Your TTY   |<--->|   lllm (C++)  |<--->|  llama.cpp    |
| (readline)  |     |  controller   |     |  inference    |
+-------------+     +-------+-------+     +---------------+
                            |
              +-------------+-------------+
              |             |             |
        +-----v-----+ +----v-----+ +-----v------+
        | Filesystem| | Network  | |  Web       |
        |  Tools    | | Search   | | Browser    |
        |           | | & PDFs   | | (FIFO)     |
        +-----------+ +----------+ +------------+
```

- **`lllm`** — The C++ binary. Handles the REPL, KV-cache management, tool dispatch, and signal handling.
- **`lllmServer.py`** — An optional Python WebSocket server that streams output to a browser via `/tmp/lllm.fifo`.

---

## Prerequisites

1. A GPU with CUDA support (NVIDIA recommended) and the CUDA toolkit installed.
2. [llama.cpp](https://github.com/ggerganov/llama.cpp) compiled from source in a sibling directory (`../llama.cpp`).
3. Python 3 with `aiohttp` for the browser server: `pip3 install aiohttp`.
4. A GGUF model file (e.g., Qwen, Llama, Mistral).

---

## Building

```bash
make
```

---

## User Setup

### 1. The `ai` System User

LLLM **must** run as the system user `ai`. The binary enforces this at startup — it checks `getuid()` and refuses to run otherwise. This is a security measure: the LLM has filesystem write access and shell execution capabilities, so isolating it behind a dedicated user limits blast radius.

Create the user and group if they don't exist:

```bash
sudo groupadd -f ai
sudo useradd -m -g ai -s /bin/bash ai
```

### 2. Git Safe Directories

By default, Git refuses to operate in repositories owned by a different user. Since LLLM runs as `ai` but your project directories are owned by you, add the `[safe]` section to **your personal** `~/.gitconfig`:

```bash
[safe]
    directory = /home/ai
```

This prevents "fatal: detected dubious ownership in repository" errors when the `ai` user runs git commands.

### 3. Directory Permissions

Your project directories should be group-writable by `ai` so the LLM can read and write files. The recommended layout for `/home/ai`:

```bash
$ ls -ld /home/ai
drwxrwsr-x+ 32 $USER ai 4096 Jun  6 11:53 /home/ai/
```

The setgid bit (`s`) ensures new files inherit the `ai` group.

A convenient alias to fix permissions on any project directory — add this to **your personal** `~/.bashrc`:

```bash
alias fixai 'sudo chown -R $USER:ai .; chmod -R g+w .'
```

Run `fixai` in any working directory before starting a session so the `ai` user has full read/write access.

### 4. Running as User `ai`

You launch LLLM by switching to user `ai` while preserving your environment:

```bash
alias coder='sudo -u ai -E taskset -c 0-15 ~/bin/lllm ~/models/Qwen3.6-27B-UD-Q5_K_XL.gguf'
```

This alias does three things:

- **`sudo -u ai -E`** — Runs as user `ai`, preserving your environment variables (`-E`).
- **`taskset -c 0-15`** — Pins LLM inference to performance cores (P-cores), leaving efficiency cores (E-cores) free for the Python browser server and background services.
- **Model path** — Points to your preferred GGUF model.

Add this alias to your personal `~/.bashrc`.

### 5. Setting Up `/home/ai/.bashrc`

Add these lines to the `ai` user's `/home/ai/.bashrc`:

```bash
# --- /home/ai/.bashrc ---

umask 0002                    # Group-writable files by default

export MAKEFLAGS=-j8          # Parallel builds for exec_shell tool

# Track current working directory so the LLM knows where you are.
cd() {
    builtin cd "$@" && pwd > /home/ai/.cwd
}

# Optional: add custom paths
export PATH="$HOME/bin:$PATH"
```

The `cd` override writes the current directory to `/home/ai/.cwd`, which LLLM reads at startup so it knows your working directory. The `umask 0002` ensures files created by the `ai` user are group-readable/writable.

### 6. The System Prompt

Place your system prompt in `/home/ai/prompt`. This file is read once at startup and baked into the KV-cache. It defines the LLM's behavior — available tools, editing workflow, formatting rules, etc. A default `prompt` file ships with this repository. You can customize it for different use cases (coding assistant, writer, researcher, etc.).

### 7. Session Aliases

Create `/home/ai/.lllm_aliases` to define shorthand commands at the `>>>` prompt:

```
# ~/.lllm_aliases — one key=value per line; # comments are ignored

test=test the filesystem tools and clean up after
message=show me a 1-sentence concise git commit message
commit=run git commit -a
make=run make
amend=amend the previous commit
```

At the `>>>` prompt, typing `commit` expands to `run git commit -a` before being sent to the LLM. Lines starting with `#` are treated as comments. Blank lines are skipped. This is loaded fresh each session.

---

## Output Modes

Set via `LLLM_OUTPUT`:

| Value | stdout | Browser | Think | Description |
|-------|--------|---------|-------|-------------|
| 3 | ✅ | ✅ | ✅ | Both stdout and browser |
| 2 (default) | ❌ | ✅ | ✅ | Browser only, think to stdout |
| 1 | ✅ | ❌ | ✅ | Stdout only, no browser |
| 0 | ❌ | ❌ | ❌ | No output (system stderr still works) |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLLM_PORT` | `8765` | Port for the browser WebSocket server |
| `LLLM_VIEWER_URL` | *(auto)* | Override the auto-generated viewer URL |
| `LLLM_DEBUG` | unset | Set to `1` for verbose token-level logging in `log/<N>.tokens` |
| `LLLM_SHOW_TOOLS` | `1` | Set to `0` to hide tool call details from the browser |
| `LLLM_GPU_LAYERS` | `999` | Number of layers offloaded to GPU (`999` = all) |
| `LLLM_USE_MMAP` | `0` | Use memory-mapped model loading (faster startup, more RAM pressure) |
| `LLLM_USE_MLOCK` | `1` | Lock model in RAM to prevent swapping |
| `LLLM_CTX` | `262144` | Context window size (KV-cache token capacity) |
| `LLLM_BATCH` | `2048` | Batch size for token feeding |
| `LLLM_UBATCH` | `512` | Unbatched size |
| `LLLM_THREADS` | `8` | Threads for inference |
| `LLLM_THREADS_BATCH` | `8` | Threads for batch processing |
| `LLLM_TURN_TIMEOUT` | `300` | Maximum seconds per generation turn before auto-abort |
| `LLLM_MAX_AUTO_CONTINUE` | `500` | Maximum depth of automatic tool-call chaining |

---

## The AI Sandbox

The `ai` user operates in a sandboxed environment:

- **File access** is limited to directories where the `ai` group has write permission. Use `fixai` to grant access to new project directories.
- **Shell commands** executed via the `exec_shell` tool run as user `ai`. They inherit the `ai` user's PATH and environment.
- **Git integration**: The sandbox repo is a separate git repository that is writable by user `ai` using an SSH deploy key. This allows the LLM to commit, push, and manage version control autonomously without needing your personal credentials.

Set up a deploy key:

```bash
# As user ai, generate a dedicated key
sudo -u ai ssh-keygen -t ed25519 -f /home/ai/.ssh/id_ed25519_lllm -N ""

# Add the public key to your Git hosting provider as a deploy key with write access
cat /home/ai/.ssh/id_ed25519_lllm.pub
```

---

## Usage

### Starting a Session

Run your alias (e.g., `coder`). You'll see the `>>>` prompt. Type your request and press Enter.

### Available Tools

The LLM has access to six tools. You don't call them directly — just describe what you want and the LLM handles the rest:

- **`read_files`** — Read text files, PDFs, or URLs
- **`search_file`** — Search for text within a file, with optional line range
- **`edit_file`** — Replace exact text within a file (surgical editing)
- **`write_file`** — Write or overwrite a file
- **`exec_shell`** — Run a shell command as user `ai`
- **`web_search`** — Search the web via SearXNG

Tools are invoked via XML-structured tags in the LLM's output. Results are fed back as continuation tokens into the KV-cache, avoiding the overhead of re-tokenizing the full conversation history.

### Multi-line Input with Ctrl+J

The readline interface binds **Ctrl+J** to insert a literal newline into your input, while **Enter (Return)** submits the line. This lets you compose multi-line messages:

```
>>> Here's a function I need fixed:
(ctrl-j)
(ctrl-j)
def foo(x):
(ctrl-j)
    return x + 1
(ctrl-j)
(ctrl-j)
Please fix the type hints.      <-- Enter submits
```

The binding is restored after each input cycle so it doesn't leak into your shell session.

### Readline Keybindings

The prompt uses GNU readline in callback mode with `select()` polling instead of blocking I/O. This allows Ctrl+C interrupts to be handled immediately during user input, not just during generation. All standard Emacs-style bindings are available:

| Keys | Action |
|------|--------|
| Ctrl+A / Ctrl+E | Beginning / end of line |
| Ctrl+B / Ctrl+F | Backward / forward character |
| Alt+B / Alt+F | Backward / forward word |
| Ctrl+K / Ctrl+U | Kill to end / start of line |
| Ctrl+W | Kill word backward |
| Ctrl+Y | Yank (paste) |
| Ctrl+_ | Undo |
| Tab | Completion (if configured) |
| Up / Down | History navigation |
| Ctrl+R | Reverse search history |

### In-Session Commands

| Command | Effect |
|---|---|
| `quit` / `exit` | Exit the session |
| `clear` | Clear the KV-cache (resets to system prompt only). All history is lost but the process stays alive. |
| `reset` | Reset internal state (loop detector, file cache) without clearing the KV-cache |
| `reincarnate` | Ask the LLM to compose a new prompt in `~/userprompt`, then clear and restart with it |
| `continue` | Resume generation after an interruption. If interrupted mid-tool-call, resumes from the exact point of interruption |

### Interrupting Generation

Press **Ctrl+C** during generation to interrupt. The partial output is preserved in the KV-cache. Type `continue` to resume seamlessly — the LLM doesn't even know it was interrupted. This works because the KV-cache still holds all generated tokens up to the interruption point.

### Browser Output

By default (`LLLM_OUTPUT=2`), LLLM streams output to a browser via WebSocket. Load:

```
http://<hostname>:8765/viewer.html
```

The Python server (`lllmServer.py`) is auto-started by the C++ binary when browser output is enabled. It's pinned to efficiency cores (E-cores, 16–23) via `taskset`. The server reads from a named FIFO at `/tmp/lllm.fifo` and broadcasts to all connected WebSocket clients.

---

## Session History

Input history is persisted to `.lllm_history` in the current working directory and survives across sessions. Use the Up/Down arrow keys to navigate previous inputs.

---

## Logging

Each session writes a numbered log file in `log/<N>`. With `LLLM_DEBUG=1`, a companion token-level trace is written to `log/<N>.tokens` showing every token fed into and generated by the model, with labels like `FEED SYSTEM_PROMPT_INIT`, `FEED USER_INPUT`, etc.

---

## License

See [LICENSE](LICENSE) for details.