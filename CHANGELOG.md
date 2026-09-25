
# Changelog

## Unreleased

Release notes relative to v0.1.1.

### Highlights

- **Persistent browser server** (`limServer.py`): the server is now a long-lived service started detached (double-forked + `setsid`, reparented to init) by the first lim session that needs it, so it survives lim's clean exit and hard crashes. A relaunching lim attaches to the verified running server, skips the startup browser prompt when the browser is still connected, and streams a dashed `-- New Session --` divider -- the browser tab stays connected across lim restarts and crashes, with no manual reload on the remote laptop. `/reset` replaces the server if it has stopped or is left reading a stale FIFO node.

### Bug Fixes

- **Crash-dance fixed by construction**: after a hard crash (segfault/OOM/SIGKILL), the first lim relaunch now reuses the surviving persistent server immediately -- no degraded session, no launch/exit/relaunch dance. The broken stale-kill path that caused it, including the `TIMER_ABSTIME` bug in `kill_and_reap` that made its "3-second" sleep ~0, is deleted along with the code it patched.
- Removed the `fuser` external-tool dependency: stale-server detection and teardown now use a PID file plus a `/proc/<pid>/cmdline` check, with a bounded SIGKILL wait (100 ms slices, 3 s cap). A process on the port that is not our `limServer` is never killed -- a clear error is reported instead (kill it or set `LIM_PORT`).
- **Stale-FIFO detection**: a server left reading a deleted or replaced FIFO node (manual `unlink` or `/tmp` cleanup) previously passed the port check, so browser output silently went nowhere and neither a relaunch nor `/reset` could recover it (the port was still bound). lim now verifies the running server's open fds (`/proc/<pid>/fd`) against the FIFO inode it writes to (`pipe_fd`), and replaces the stale server at startup and on `/reset`.
- **PID-file fallback**: server identification no longer depends solely on `/tmp/lim.server.pid` -- if the file is gone while the server lives, lim falls back to a `/proc` cmdline scan, so a surviving server is found and reused (or replaced) instead of being misreported as a foreign port occupant.
- **No more server zombies**: the double fork reparents the server to init, so a server that dies while lim is running is reaped by init instead of lingering as a zombie of lim.
- Fixed a check-then-watch race in the marker-file waiter: the inotify watch is now armed before the existing-file check, so a marker created in the gap can no longer be missed.
- The stale-FIFO inode comparison now uses `stat()` on the `/proc/<pid>/fd/N` magic links: `lstat` reports the link itself (always `S_IFLNK`), so it matched no fd of any process and every healthy server looked stale -- replaced, and the browser reloaded, at every restart.
- Server identification requires `argv[0]` to be a python interpreter in addition to the exact `<path>/limServer.py` cmdline arg, and the port-free stale-teardown only kills a candidate that holds the FIFO node lim writes to: an editor or tool that merely has the script open can no longer become a kill candidate.
- Removed the `TERM_PROGRAM=vscode` handshake: the extension's SSH terminal no longer exports it (plain `ssh -t -a`), and the startup browser prompt no longer branches on it. The VS Code viewer reset (**Ctrl+2** / **Ctrl+Shift+R**, `LIM: Reload Browser`) is documented in the README instead.

### Behavior Changes & Default Changes

| Change | Old | New |
|---|---|---|
| `limServer` lifecycle | Per-session child: killed by atexit on exit; orphaned on crash; browser page had to be reloaded on every lim restart | Persistent service: started detached (double-forked + `setsid`, reparented to init) by the first session that needs it, survives lim exit/crash, browser stays connected across sessions; verified against the live FIFO inode via `/proc/<pid>/fd` (PID file with `/proc` cmdline fallback); `/reset` replaces it if it stops or is on a stale FIFO node; stale server on a changed `LIM_PORT` is torn down with a bounded wait |

## v0.1.1 -- 2026-09-09

Release notes relative to v0.1.0.

### Highlights

- **Instant `/undo` for hybrid models** (Qwen3.5/3.6): a new recurrent-state checkpoint API in the bundled llama.cpp snapshots the model's recurrent state at every prompt boundary, so undoing a turn no longer requires re-decoding history. `--checkpoints` now also works with the in-session `/load` command and rebuilds checkpoints for all historical turns.
- **Fail-safe session restore**: restores pre-check context fit, reset to a fresh session cleanly on decode failure or interrupt, decode the git-HEAD note so users actually see it, and prune orphaned or truncated fast-cache entries.
- **Better web access**: Brave Search API fallback when SearXNG fails, automatic GitHub content fetching via the API (issues, PRs, discussions, actions, blobs), and smarter page fetches (targeted body extraction, SPA detection with API hints, stricter search quality gate).
- **Prompt-derived reasoning modes**: new `REASONING.md` guide, site-specific `localprompt` prepended to the system prompt, and a configurable `LIM_DUMMY_THOUGHT` stub for `LIM_THINKING=0`.
- **Browser viewer performance**: progressive rendering is now O(N) -- quadratic re-scans on every token are gone while commit decisions stay byte-identical.

### New Features

- Recurrent state checkpointing (llama.cpp patch) for instant `/undo` on hybrid models; tool-correction rollback is unified onto the same checkpoint stack.
- `/quit <path>` and `/exit <path>`: named save with fast cache, then exit.
- `/load <path> [--checkpoints]`: `--checkpoints` now works in-session (no exit/restart needed) and rebuilds recurrent checkpoints at each prompt boundary during decode.
- `localprompt`: an optional directory-specific file (falling back to `$LIM_CONFIG_DIR/localprompt`) is prepended to the system prompt -- use it for per-site instructions, e.g., Qwen reasoning-effort steering.
- `LIM_DUMMY_THOUGHT`: configurable pre-filled stub for `LIM_THINKING=0` (empty string emits the empty thinking block, Qwen 3.8's "no thinking" signal).
- `REASONING.md`: setup guide for prompt-derived reasoning modes (xhigh/low) with mode files, per-mode sampling env, and a login hook that keeps them consistent.
- `LIM_BRAVE_API_KEY`: Brave Search API fallback when SearXNG fails or returns no results.
- GitHub integration: auto-fetch issues, PRs, discussions, actions, and blobs via the GitHub API (bypasses SPA rendering); `GH_TOKEN` for higher rate limits; required User-Agent header added.
- Web fetch improvements: targeted body extraction, SPA detection with API hints, stricter search-quality gate.
- `search_file`: range search may now omit `begin` and/or `end`; line numbers are parsed forgivingly (leading integer prefix) so a missing closing tag or stray whitespace recovers the intended range.
- Model-dependent thinking tags; falls back to bare `thinking` tags when chat-template autodetection fails.
- Model loading: attempt a direct full-GPU load first, falling back to auto-fit (`common_fit_params`) only on OOM.
- `exec_shell` runs commands via a temp script instead of `bash -c`, fixing commands containing parentheses; now responsive to Ctrl+C.
- Missing `edit_file` path is auto-inferred from the last `search_file` call.
- Browser viewer: horizontal scrollbar in code blocks (whitespace preserved).
- Option B local launch: new `sudo -iu`-based `coder` shell function starts lim in the caller's directory and forwards arguments, so CLI restores with relative save-file paths work from a personal shell.
- Security: SSH agent forwarding is disabled (`ssh -a`) for LIM connections.
- Repo/docs: benchmark graph (cumulative time vs context length for all three `LIM_CHATBOT_MODE`s), new logo/screenshot/social preview, `.editorconfig`, README badges.

### Bug Fixes

**Crashes & corruption**

- Fixed a crash on 4-byte UTF-8 sequences decoding above U+10FFFF (`llama-utf8.patch`).
- Fixed KV-cache clobbering on tool-correction rollback: the recurrent checkpoint slot is saved right after `FUNC_START` and only the corrected call body is injected.
- Kept the token tracker and fast-restore cache in lockstep: turn-ending tokens (tool-call close, final EOG) are now decoded into the KV cache, and the `/undo` fallback re-decodes only the undo-target prefix instead of drifting one token per turn.
- Session restore: pre-check context fit before decoding, force a fresh-session reset on decode failure or interrupt, and prune orphaned or truncated fast-cache entries.
- Fixed stale tokens leaking into save files after a correction rollback (all_context_tokens resized correctly).

**Undo / restore / checkpoints**

- Slow-restore decode chunks now end exactly on prompt boundaries, so saved recurrent checkpoints match their exact undo positions instead of lagging up to one batch.
- Checkpoint matching on token-count suffix rather than full prompt string; long checkpoint labels truncated in the undo/restore prompts.
- Invalid-tool strike counter resets on `/continue` (no stale circuit breaker); two correction attempts allowed after injection, ejecting on third failure.
- Boundary checkpoint saved on in-session fast restore and after reincarnate; checkpoint-stack offset corrected after fast restore so `/undo` maps to the last restored turn.
- Readline history: chronological order preserved when restoring after an undo cancel or `/load` checkpoint prompt; `C` entries re-counted so they persist to `.lim_history` on the next flush.
- `/clear` promotes `C` entries to `A` and preserves them across `/load` restores; file cache cleared after undo and on reincarnate (prevents stale cache hits).

**Tool calling & session robustness**

- Whitespace-robust tool calling: no-whitespace protocol rule in the prompt, 0-match results echo the exact query, `search_file` simplified to a single exact-text search with number-free match headers.
- Newline in a `path` parameter is treated as an implicit end-of-parameter boundary; bare `<name>` tags accepted as fallback when the canonical `<parameter=name>` form is missing.
- Leaked thinking blocks are stripped from tool calls before parsing; oversized tool results are fed back as a compact error instead of ejecting to the prompt, so the LLM can narrow the request and the session survives.
- Silent-loop detector scoped to tool calls (replaces the removed standalone loop detector); silent-loop aborts route through the tool-correction cycle instead of dying.
- Tool correction reports why it was skipped when the correction prompt itself doesn't fit in context, instead of failing silently.
- `/clear` reloads the system prompt from disk, picking up edits and a fresh timestamp.
- Auto-resume on premature EOG after `/reincarnate`; empty and partial opening thinking tags no longer leak into the browser; speed-diagnostic leak fixed (single pipe_write); reconnecting browsers get the current context position; context position sent on first turn; browser status bar updated after `/load`.

**Browser viewer**

- Progressive rendering made O(N): the break/fence scan resumes from a per-wrapper high-water mark with position-based list/header guard state -- no more quadratic re-scans on every token, with byte-identical commit decisions.
- Fixed progressive-render stalls and premature thinking-block collapse: only line-start code fences count as fences, and a think block is never collapsed by a mid-stream force-render when prose mentions backticks.
- Think-tag close is depth-matched (O(N) incremental scan), so a spurious inner close tag no longer ends the thinking block early, collapsing it prematurely or leaking into prose.
- Nested thinking blocks: `segToHtml` emits only the think header and body, so a re-rendered think segment no longer injects a `.think-block` inside one.
- Export renders from a DOM clone so active thinking blocks are included without mutating the live session; think-block arrow transition removed (no flicker during streaming); Ctrl+J newlines render as `<br>` in the browser user-prompt display; HTML sentinel changed from `|` to `%` to avoid collisions with markdown table delimiters; markdown backslash consumption around `marked.parse()` fixed via placeholder protection.

**Build & environment**

- Makefile: fixed `$$` escaping for CUDA arch detection inside `$(shell)`; improved GPU architecture detection; `LIM_LLAMA_BUILD_DIR` is treated as in-tree only if it resolves to `./llama/build`.
- llama.cpp subrepo updated: recurrent-state checkpoint API, sampler penalties API fix, `llama-vocab-gcc15.patch` applied.
- llama debug log spam suppressed (set `LIM_LLAMA_DEBUG_LOG=1` to re-enable).
- VS Code extension: removed bash-only `export TERM_PROGRAM` for local terminals.
- Benchmark mode 1 (standard chatbot) re-feeds exact saved tokens instead of detokenizing/re-tokenizing, and no longer double-feeds the user message.

### Behavior Changes & Default Changes

| Change | Old | New |
|---|---|---|
| `LIM_TURN_TIMEOUT` default | `300` s | `3600` s -- the standalone loop detector is removed; runaway tool chains are bounded by the turn timeout, `LIM_MAX_AUTO_CONTINUE`, and the tool-call silent-loop detector |
| `LIM_EOG_RESAMPLE_MAX` default | `64` | `256` -- better recovery for models with frequent spurious EOGs |
| Sampling env var names | `LIM_TEMP`, `LIM_PENALTY_FREQ`, `LIM_PENALTY_PRESENT`, `LIM_PENALTY_REPEAT` | `LIM_TEMPERATURE`, `LIM_FREQUENCY_PENALTY`, `LIM_PRESENCE_PENALTY`, `LIM_REPETITION_PENALTY` (model-card/vLLM spelling). Legacy names still work; if both are set, the new name wins |
| Loop detector | Standalone `loop_detector.h` | Removed; replaced by the tool-call silent-loop detector + timeouts |
| Thinking tags | Legacy `THINK_START`/`THINK_END` | Model-dependent tags, with bare-tag fallback |
| Prompt loading | Legacy `~/prompt` fallback | Removed; site-specific `localprompt` + `~/.config/lim/prompt` only |
| SSH connections | Agent forwarding allowed | `ssh -a` (agent forwarding disabled) |
| README model examples | Qwen3.6-27B-UD-Q5_K_XL | Qwen3.8-27B-Q6_K |

### Internals

- Large DRY cleanup: duplicated service-startup, curl-fetch, search-result formatting, topology reads, stale-server handling, quote-stripping, token/log escaping, cache scans, save-header parsing, and tool-result turn building consolidated into shared helpers across all modules; dead code removed; duplicated save/restore, tool-rollback, session-announce, and diagnostic logic unified.
- All C++ sources reindented from 4-space to 2-space indentation.
- CLI restore injects `/load` verbatim, so CLI and in-session restore share one code path; `--checkpoints` is trailing-only everywhere (shared `strip_checkpoints_flag()` parser).
- New shared `LIM_DEFAULT_CTX` constant for the default context size.

