# LIM: Stateful, O(1) Local Inference Manager

**LIM** (Local Inference Manager) is a C++ terminal-based LLM controller built on [llama.cpp](https://github.com/ggml-org/llama.cpp). It provides a persistent, stateful session with **true O(1) history injection** via a continuously appended KV-cache: no context transmission or re-tokenization on every turn. The O(1) claim refers to per-turn history injection cost, not attention complexity. LIM has native filesystem tools for reading, searching, editing, and writing files, plus web searching and PDF reading.

## Why O(1)? The Fundamental Difference

Every mainstream chatbot (and most local LLM frontends) follows the *reprocess-every-turn* model: each time you send a message, the **entire conversation history** is re-transmitted in full and re-tokenized from scratch. Cost grows **quadratically with conversation length**.

LIM takes full advantage of what a single-user local setup affords:

- **The KV-cache is never discarded.** Each turn's tokens are simply appended to `n_past`. The model continues generating from exactly where it left off.
- **New user input costs O(input tokens)**, not O(total history). Even after hundreds of turns, each new message processes only its own tokens plus the delta since last turn.
- **No re-tokenization overhead.** The system prompt is tokenized once at startup and lives in the cache forever.

---

## Architecture at a Glance

```
+-------------+     +---------------+     +---------------+
|  Your TTY   |<--->|   lim (C++)   |<--->|  llama.cpp    |
| (readline)  |     |  controller   |     |  inference    |
+-------------+     +-------+-------+     +---------------+
                            |
              +-------------+-------------+
              |             |             |
        +-----v-----+ +----v-----+ +-----v------+
        | Filesystem| | Web      | |  Browser   |
        |  Tools    | | Search   | | (FIFO)     |
        |           | | & PDFs   | |            |
        +-----------+ +----------+ +------------+
```

- **`lim`**: The C++ binary. Handles the REPL, KV-cache management, tool dispatch, and signal handling.
- **`limServer.py`**: An optional Python WebSocket server that streams output to a browser via `/tmp/lim.fifo`.

---

## Prerequisites

1. A GPU with CUDA support (NVIDIA recommended) and the CUDA toolkit installed.
2. Python 3 with `aiohttp` for the browser server: `pip3 install aiohttp`.
3. A GGUF model file (e.g., Qwen, Llama, Mistral).
4. Optional: [SearXNG](https://github.com/searxng/searxng) for web search, [Docling](https://github.com/DS4SD/docling) for PDF reading.

> **Note:** llama is bundled as a git subrepo and will be built automatically by the Makefile.

---

## Building

```bash
git clone https://github.com/statellm/lim.git
cd lim
make
./lim --help
```

The build will automatically detect your GPU (if any) and compile llama with the appropriate architecture flags. No manual setup is required.

### Custom Build Options

All auto-detected values can be overridden via environment variables:

| Variable | Purpose | Example |
|---|---|---|
| `CUDA_ARCH_FLAGS` | Override GPU architecture detection | `make CUDA_ARCH_FLAGS=120a` |
| `LLAMA_BUILD_DIR` | Use pre-built llama from external directory | `export LLAMA_BUILD_DIR=/path/to/llama/build` |
| `LLAMA_CMAKE_FLAGS` | Extra cmake flags passed to the llama build | `make LLAMA_CMAKE_FLAGS="-DGGML_AVX512=off"` |
| `GGML_CUDA` | Force CUDA on/off | `make GGML_CUDA=off` for CPU-only |
| `GGML_HIPBLAS` | Enable ROCm/HIP build | `make GGML_HIPBLAS=on` |

To clean only the llama build artifacts without touching lim:

```bash
make llama-clean
```

---

## User Setup

The dedicated LLM user is controlled by the `LIM_AI_USER` environment variable (defaults to `ai`). All references below use `$LIM_AI_USER`. The group name always matches the user name.

### 1. Creating the AI User

LIM **must** run as a dedicated user. The binary enforces this at startup: it checks `getuid()` and refuses to run otherwise. This is a security measure: the LLM has filesystem write access and shell execution capabilities, so isolating it behind a dedicated user limits blast radius.

Create the user and group if they don't exist:

```bash
export LIM_AI_USER=ai
sudo groupadd -f $LIM_AI_USER
sudo useradd -m -g $LIM_AI_USER -s /bin/bash $LIM_AI_USER
```

Then add your personal user to the `$LIM_AI_USER` group so you can read and write files the LLM creates without needing `sudo chown`:

```bash
sudo usermod -aG $LIM_AI_USER $USER
```

Log out and back in (or run `newgrp $LIM_AI_USER`) for the group change to take effect.

### 2. Git Safe Directories

By default, Git refuses to operate in repositories owned by a different user. Since LIM runs as `$LIM_AI_USER` but your project directories are owned by you, add the `[safe]` section to **your personal** `~/.gitconfig`:

```bash
[safe]
    directory = /home/$LIM_AI_USER
```

This prevents "fatal: detected dubious ownership in repository" errors when the LLM runs git commands.

### 3. Directory Permissions

Your project directories should be group-writable by the `$LIM_AI_USER` group so the LLM can read and write files. Because you added your personal user to the `$LIM_AI_USER` group (step 1), access is symmetric: you can read files the LLM creates, and the LLM can read your files. The recommended layout for `/home/$LIM_AI_USER`:

```bash
$ ls -ld /home/$LIM_AI_USER
drwxrwsr-x. 42 $USER $LIM_AI_USER 4096 Jan  1 00:00 /home/$LIM_AI_USER/
```

The setgid bit (`s`) ensures new files inherit the `$LIM_AI_USER` group.

To set permissions in any project sandbox, copy `aishare` from this repository to your personal `/home/$USER/bin/`:

```bash
cp aishare /home/$USER/bin/
```

Then run it on any directory you want the AI to access:

```bash
aishare /home/$LIM_AI_USER/project
```

It sets ownership to `$USER:$LIM_AI_USER`, grants group read/write and setgid, and clears the sticky bit. When run on `/home/$LIM_AI_USER`, it fixes permissions on the home directory itself plus all top-level entries, skipping `.ssh`.

### 4. Connecting and Running LIM

LIM must be run as user `$LIM_AI_USER`. The simplest approach is to SSH into the host as that user, then launch `lim` directly. This keeps the workflow consistent whether you're using VS Code or a standalone terminal.

Add this to the `~/.bashrc` of `$LIM_AI_USER`:

```bash
alias coder='lim ~/models/Qwen3.6-27B-UD-Q5_K_XL.gguf'
```

Replace the model path with whichever GGUF you want to use. Then, before running `coder`, connect to the LIM server `$LIM_HOST`:

```bash
ssh $LIM_AI_USER@$LIM_HOST
coder
```

If running locally (no SSH needed), switch to the user first:

```bash
su - $LIM_AI_USER
coder
```

#### Core Pinning (taskset)

LIM automatically detects P-cores and E-cores on hybrid CPUs (Intel Alder Lake, Raptor Lake, etc.) by comparing `thread_siblings_list` from sysfs. It pins background services accordingly:

| Workload | Default Cores | Description |
|---|---|---|
| SearxNG search server | E-cores | Light background Python process |
| Browser WebSocket server | E-cores | Light background Python process |
| Docling PDF converter | P-cores | Heavy ML inference workload |

On non-hybrid CPUs (all cores identical), no taskset pinning is applied. If the pinning command isn't found on `$PATH` (e.g., macOS has no `taskset`), pinning is silently skipped.

Override the auto-detected layout with `LIM_TASKSET`:

```bash
# Format: LIM_TASKSET="P_CORES:E_CORES"
export LIM_TASKSET="0-15:16-23"   # Classic i9-12900K layout
export LIM_TASKSET="0-7:"         # P-cores 0-7, no E-core pinning
export LIM_TASKSET=":8-15"        # No P-core pinning, E-cores 8-15
export LIM_TASKSET="::"           # Disable all taskset pinning
```

**macOS / non-Linux systems:** macOS has no `taskset` binary. To enable manual core pinning, install an alternative and point `LIM_TASKSET_CMD` at it:

```bash
# Install numactl via Homebrew, then use it as the pinning backend
brew install numactl
export LIM_TASKSET="0-3:4-7"
export LIM_TASKSET_CMD="numactl --cpunodebind"
```

Or write a custom wrapper script and set `LIM_TASKSET_CMD` to its path. If the command isn't on `$PATH`, pinning is silently skipped -- services still start normally, just unpinned.

The detected topology and pinning status are logged to stderr at startup.

### 5. Setting Up `$HOME/.bashrc` for `$LIM_AI_USER`

Add these lines to `/home/$LIM_AI_USER/.bashrc`:

```bash
# --- /home/$LIM_AI_USER/.bashrc ---

umask 0002                    # Group-writable files by default

# Track current working directory so the LLM knows where you are.
cd() {
    builtin cd "$@" && pwd > $HOME/.cwd
}

# Block git add -A / --all to prevent the LLM from staging untracked files
git() {
    if [ "$1" = "add" ]; then
        for arg in "$@"; do
            if [ "$arg" = "-A" ] || [ "$arg" = "--all" ] || [[ "$arg" =~ ^-[a-zA-Z]*A[a-zA-Z]*$ ]]; then
                echo "ERROR: git add -A / --all is disabled: use -a" >&2
                return 1
            fi
        done
    fi
    command git "$@"
}

# Optional: add custom paths
export PATH="$HOME/bin:$PATH"
```

The `cd` override writes the current directory to `$HOME/.cwd`, which LIM reads at startup so it knows your working directory. The `umask 0002` ensures files created by `$LIM_AI_USER` are group-readable/writable.

### 6. The System Prompt

Place your system prompt in `/home/$LIM_AI_USER/prompt`. This file is read once at startup and baked into the KV-cache. It defines the LLM's behavior: available tools, editing workflow, formatting rules, etc. A default `prompt` file ships with this repository. You can customize it for different use cases (coding assistant, writer, researcher, etc.).

### 7. Message Shortcuts

Create `/home/$LIM_AI_USER/.lim_aliases` to define shorthand expansions at the `>>>` prompt:

```
# ~/.lim_aliases: one key=value per line; # comments are ignored
# Keys must start with / to distinguish them from regular messages

/test=test the filesystem tools and clean up after
/message=show me a 1-sentence concise git commit message
/commit=run git commit -a
/make=run make
/amend=amend the previous commit
```

At the `>>>` prompt, typing `/commit` expands to `run git commit -a` before being sent to the LLM. Lines starting with `#` are treated as comments. Blank lines are skipped. Built-in command names (e.g., `/quit`, `/clear`) cannot be used as alias keys. This is loaded fresh each session.

---

## Output Modes

Set via `LIM_OUTPUT`:

| Value | stdout | Browser | Description |
|-------|--------|---------|-------------|
| 3 | ✅ | ✅ | Both stdout and browser |
| 2 (default) | ❌ | ✅ | Browser only |
| 1 | ✅ | ❌ | Stdout only |
| 0 | ❌ | ❌ | No output (system stderr still works) |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LIM_AI_USER` | `ai` | Username that LIM must run as. Used by the binary for the user check, and by the VS Code extension for SSH. |
| `LIM_HOST` | unset | Hostname or IP of your LIM server. Used for SSH connection and browser viewer URL. |
| `LIM_PORT` | `8765` | Port for the browser WebSocket server |
| `LIM_VIEWER_URL` | *(auto)* | Override the auto-generated viewer URL |
| `LIM_DEBUG` | `0` | Set to `1` for verbose token-level logging in `log/<N>.tokens` |
| `LIM_EOG_RESAMPLE_MAX` | `64` | Maximum resampling attempts when a spurious EOG is detected. When the model emits an EOG token but hasn't finished its response, LIM resamples up to this many times trying to recover a non-EOG token. Increase if you see premature turn endings with Qwen3.6 or similar models. |
| `LIM_GPU_LAYERS` | `-1` | Number of layers offloaded to GPU (`-1` = auto-fit all layers). When set explicitly, bypasses auto-fitting. For MoE models that exceed VRAM, auto-fit uses partial layer offloading (dense weights on GPU, sparse expert weights on CPU) for optimal throughput. |
| `LIM_HONEST_SPEED` | `0` | Set to `1` for "honest" wall-clock speed diagnostic -- tokens / total generate() call time, including all CPU-side overhead (sampling, output rendering, tool-call detection). Default `0` reports generation time matching llama-cli's "Generation: X t/s" -- measured from first-token sample+sync to last-token sample+sync, covering N sampling operations and (N-1) decode cycles. |
| `LIM_SPEED_INTERVAL` | `100` | Number of tokens between in-loop speed diagnostic updates. |
| `LIM_USE_MLOCK` | `1` | Lock model in RAM to prevent swapping |
| `LIM_USE_MMAP` | `0` | Use memory-mapped model loading (faster startup, more RAM pressure) |
| `LIM_BATCH` | `2048` | Batch size for token feeding |
| `LIM_CTX` | `262144` | Context window size (KV-cache token capacity) |
| `LIM_THREADS` | *(auto)* | Threads for inference (physical core count) |
| `LIM_THREADS_BATCH` | *(auto)* | Threads for batch processing (physical core count) |
| `LIM_UBATCH` | `512` | Unbatched size |
| `LIM_MIN_P` | `0.0` | Minimum probability threshold: keep tokens where P >= min_p * P(top) |
| `LIM_PENALTY_FREQ` | `0.0` | Frequency penalty: discourages overused tokens proportional to frequency |
| `LIM_PENALTY_PRESENT` | `1.5` | Presence penalty: discourages repeating previously used tokens |
| `LIM_PENALTY_REPEAT` | `1.0` | Repetition penalty multiplier (1.0 = no penalty) |
| `LIM_SEED` | *(auto)* | Random seed for reproducibility (default is time-based) |
| `LIM_TEMP` | `0.7` | Sampling temperature (set to `0` for deterministic/greedy decoding) |
| `LIM_THINKING` | `1` | Set to `0` to suppress thinking blocks via a pre-filled stub for faster throughput. Not recommended for math or complex reasoning tasks, as it can cause incorrect answers by skipping intermediate steps. |
| `LIM_ESCAPE_CONTRACT` | `0` | Set to `1` to include the reserved-token escape contract in the system prompt. The escape mechanism itself is always active; this only controls whether the LLM sees the explicit rules. |
| `LIM_TOP_K` | `20` | Keep only the top_k most likely tokens before applying other samplers |
| `LIM_TOP_P` | `0.8` | Nucleus sampling: consider tokens with cumulative probability <= top_p |
| `LIM_CACHE_TYPE_K` | `Q8_0` | KV-cache key storage type (`F16`, `Q4_0`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`) |
| `LIM_CACHE_TYPE_V` | `Q8_0` | KV-cache value storage type (`F16`, `Q4_0`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`) |
| `LIM_EXEC_TRUNCATION` | `32768` | Maximum bytes of exec_shell output before truncation |
| `LIM_MAX_AUTO_CONTINUE` | `500` | Maximum depth of automatic tool-call chaining |
| `LIM_TURN_TIMEOUT` | `300` | Maximum seconds per generation turn before auto-abort |
| `LIM_TASKSET` | *(auto)* | Format: `"P_CORES:E_CORES"` (e.g., `"0-15:16-23"`). Auto-detected on hybrid CPUs. Set to `"::"` to disable all pinning. |
| `LIM_TASKSET_CMD` | `taskset -c` | Override the core-pinning command. On macOS (no `taskset`), install [numactl](https://formulae.brew.sh/formula/numactl) via Homebrew and set to `numactl --cpunodebind`. If the command isn't on `$PATH`, pinning is silently skipped. |

---

## The AI Sandbox

The `$LIM_AI_USER` operates in a sandboxed environment:

- **File access** is limited to directories where the `$LIM_AI_USER` group has write permission. Use `aishare` to grant access to new project directories. Files created by the LLM are owned by `$LIM_AI_USER:$LIM_AI_USER` with group read/write permissions (via `umask 0002`), so your personal user can access them directly since it is a member of the `$LIM_AI_USER` group.
- **Shell commands** executed via the `exec_shell` tool run as user `$LIM_AI_USER`. They inherit that user's PATH and environment.
- **Git integration**: The sandbox repo is a separate git repository that is writable by `$LIM_AI_USER` using a dedicated AI GitHub account. This allows the LLM to commit, push, and manage version control autonomously without needing your personal credentials.

Set up an SSH key:

```bash
# As `$LIM_AI_USER`, generate a dedicated key
ssh-keygen -N ""

# Add the public key to your Git hosting provider
cat ~/.ssh/id_ed25519.pub
```

---

## Usage

### Starting a Session

SSH in as `$LIM_AI_USER`, then run your `coder` alias. You'll see the `>>>` prompt. Type your request and press Enter.

### Available Tools

The LLM has access to six tools. You don't call them directly; just describe what you want and the LLM handles the rest:

- **`read_files`**: Read text files, PDFs, or URLs
- **`search_file`**: Search for text within a file, with optional line range
- **`edit_file`**: Replace exact text within a file (surgical editing)
- **`write_file`**: Write or overwrite a file
- **`exec_shell`**: Run a shell command as `$LIM_AI_USER`
- **`web_search`**: Search the web via SearXNG

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
| `/quit` or `/exit` | Auto-save the current state to `log/<N>.save`, then exit the session |
| `/clear` | Auto-save the current state to `log/<N>-clear.save`, then clear the KV-cache (reset to system prompt only). The auto-saved file lets you restore if you change your mind. Use `/save <name>` to create a permanent restore point before clearing. |
| `/undo [N]` | Undo the last N user prompts (default: 1). Auto-saves first to `log/<N>-clear.save`. Restores the session to the checkpoint just before those prompts were issued. For pure attention models (Llama, Mistral), this is instant via `seq_rm`. For hybrid models (Qwen3.5/3.6), LIM restores a saved recurrent-state snapshot and trims both caches: also instant, no re-decode needed. Falls back to clear+re-decode only if no checkpoint is available (e.g., immediately after a fast cache restore before any new turns). Only actual LLM inputs are counted; LIM commands like `/undo` itself are skipped. If N exceeds available checkpoints, falls back to `/clear`. |
| `/continue` | Resume generation after an interruption. If interrupted mid-tool-call, resumes from the exact point of interruption |
| `/reset` | Reset internal state (loop detector) without clearing the KV-cache |
| `/reincarnate` | Ask the LLM to compose a new prompt in `~/userprompt`, then clear and restart with it |
| `/save` | Save the full session state to `log/<N>.save`, overwriting any previous save for this session |
| `/save <path>` | Save the full session state to `<path>.save`. The path can be relative or absolute. If it already ends in `.save`, no extra extension is added. Use this to create named restore points at meaningful moments in your session. |
| `/restore <path>` | Restore a saved session from within the current session. Must be used immediately after `/clear`: fails if any conversation tokens have been added since the clear. Accepts the same path format as `/save`: `.save` is appended automatically if not already present. Uses the fast cache when available, falling back to full re-decode with checkpoint regeneration. |
| `/help` | Display a summary of all available commands |

### Save and Restore

You can save a running session and restore it later with zero context loss:

**Save:** Type `/save` at the `>>>` prompt to save to `log/<N>.save` (overwrites any previous save for this session). Use `/save <path>` to create named checkpoints: e.g., `/save cats` saves to `cats.save`, and `/save /tmp/checkpoint` saves to `/tmp/checkpoint.save`. If the path already ends in `.save`, no extra extension is added. The save file contains only the conversation token sequence, keeping it small and model-agnostic. Named saves also cache the full KV-cache for fast future restores.

**Restore:** Pass a save file as the last argument to `coder`. The `.save` extension is added automatically if not already present, matching `/save` behavior:

```bash
coder log/5.save    # explicit extension
coder log/5         # .save appended automatically
coder cats          # restores from cats.save
```

This restores the session exactly as it was: the full conversation, KV-cache position, and generation state. The LLM continues generating from where it left off. Typing `/clear` after a restore resets to a fresh system prompt with the current date and working directory (but first auto-saves the restored state).

**In-session restore:** You can also restore from within a running session using `/restore <path>`. This must be used immediately after `/clear`; if any conversation tokens have been added since the clear, the command will fail. It uses the same path conventions as `/save` (`.save` appended automatically) and takes advantage of the fast cache when available. Typical workflow:

```
>>> /clear
[Context Cleared Successfully]
>>> /restore cats
[Restoring session from cats.save... (75432 tokens, from cache)]
[Session restored: 75432 tokens loaded]
```

**Partial restore via checkpoints:** Save files record a checkpoint at the end of each conversation turn, storing your prompt text and the token position. On restore, if a fast-format cache is not available, LIM offers a choice of checkpoints before decoding. Use up/down arrow keys to navigate through your prompts (most recent first), with "restore all" as the final option. Press Enter to confirm. Restoring to a checkpoint replays tokens only up to the end of that turn -- as if you had just typed that prompt and received the response, and the session is ready for your next message. The available checkpoints accumulate across restore/save cycles: restoring from a save file carries over its checkpoints, and new turns add more.

**Checkpoint restore:** Add `--checkpoints` to skip the fast-format cache and trigger the checkpoint selection prompt. The flag can appear before or after the save file:

```bash
coder cats --checkpoints
coder --checkpoints cats
```

Press Ctrl+C during the restore prompt to cancel without decoding.

**Fast restore cache:** On first restore, tokens are decoded through the model to rebuild the KV-cache. During this decode, recurrent-state checkpoints are regenerated at each prompt boundary so that /undo works instantly for hybrid models (Qwen3.5/3.6) right after restore. The rebuilt cache is then automatically written to `.cache/<hash>` so all subsequent restores from the same save file are fast. Named saves (e.g., `/save cats`) also write the fast-format cache immediately for faster future restores. The name prefix in the cache filename is purely informational; LIM identifies cache files by their content hash, not their name. If you run `/save cats` followed by `/save dogs` with identical session content, only one cache file is written since both resolve to the same hash. Unnamed `/save` and auto-saves from `/quit`, `/clear`, and `/reincarnate` skip the fast cache to save disk space, relying on the automatic cache built on first restore. For fast cache restores, recurrent checkpoints build up naturally as conversation turns complete to support the instant /undo feature after the first new turn. The `.cache/` directory is safe to delete at any time to reclaim space; it will be regenerated on the next restore.

**Auto-save on clear and quit:** Before clearing the context, LIM automatically saves the current state to `log/<N>-clear.save`. Before exiting, it saves to `log/<N>.save`. These use different filenames so neither clobbers the other: if you clear and then exit, both the pre-clear and post-clear states are preserved. To keep a permanent checkpoint at any point, use `/save <name>`.

### Interrupting Generation

Press **Ctrl+C** during generation to interrupt. The partial output is preserved in the KV-cache. Type `/continue` to resume seamlessly: the LLM doesn't even know it was interrupted. This works because the KV-cache still holds all generated tokens up to the interruption point.

### Browser Output

By default (`LIM_OUTPUT=2`), LIM streams output to a browser via WebSocket. Start your LIM session first, then load:

```
http://<hostname>:8765/viewer.html
```

This ensures the server is already running when the browser connects, avoiding any need to reload the page.

The Python server (`limServer.py`) is auto-started by the C++ binary when browser output is enabled. It runs on efficiency cores (auto-detected, or all cores on non-hybrid CPUs). The server reads from a named FIFO at `/tmp/lim.fifo` and broadcasts to all connected WebSocket clients.

### VS Code Extension

The LIM Workspace extension provides a convenient way to set up your workspace: it creates an integrated terminal and opens a browser viewer once the server is ready.

The extension does two things:
- **Opens a browser** pointing to `$LIM_HOST` (or localhost) on the configured port, after waiting for the server to respond.
- **Creates a terminal** at `$HOME`. If `$LIM_HOST` is set and remote, it SSHs into that host as user `$LIM_AI_USER`; otherwise it opens a local shell.

You then run your `coder` alias in that terminal as usual.

**Install:**

```bash
make vscode          # builds and packages the extension
make install         # installs the extension into VS Code
```

Then in VS Code, click the LIM rocket icon in the status bar (or **Ctrl+Shift+P** -- `LIM: Open Workspace`). This opens a terminal panel and waits for the browser server. Run your `coder` alias in that terminal, and the viewer will connect automatically once the server starts.

To rebuild both the C++ binary and the extension together:

```bash
make all
```

---

## Session History

Input history is persisted to `.lim_history` in the current working directory and survives across sessions. Use the Up/Down arrow keys to navigate previous inputs.

---

## Logging

Each session writes a numbered log file in `log/<N>`. With `LIM_DEBUG=1`, a companion token-level trace is written to `log/<N>.tokens` showing every token fed into and generated by the model, with labels like `FEED SYSTEM_PROMPT_INIT`, `FEED USER_INPUT`, etc.

---

## License

See [LICENSE](LICENSE) for details.

---
