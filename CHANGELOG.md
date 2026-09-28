# Changelog

## v0.1.2 -- 2026-09-27

Release notes relative to v0.1.1.

### Highlights

- **MTP speculative decoding** (`LIM_MTP=1`): faster generation from draft-then-verify.
- **Automatic 90% context checkpoint**: adds an extra `/undo` target for launching a `/reincarnate` command.
- **Persistent browser server**: the viewer tab stays connected across lim restarts and crashes -- no reload.
- **New `/remind` command**: re-anchors the LLM to its full system prompt in one command.
- **Improved benchmark**: mode 2 (CACHED) re-tokenizes the full conversation each turn like a server with persistent caching and is now seen to be noticeably slower than LIM's incremental decoding.

### New Features

- MTP (multi-token-prediction) speculative decoding behind `LIM_MTP=1`: a second context mirroring the main KV cache proposes up to `LIM_MTP_DRAFT` tokens per round, and the main model verifies them all in a single batch decode, so the mirror's quantization costs acceptance rate only, never output correctness. Speculation pauses when the mirror can't be kept in sync (a main decode truncated by KV exhaustion, or a fast-restore from a cache saved without MTP) or after sustained low acceptance (it auto-disables itself; `/clear` re-enables it). MTP is not implemented for benchmark chatbot modes 1 and 2.
- `LIM_MTP_DRAFT` (default `4`, 1-32): number of tokens proposed per MTP round; must not exceed the model's recurrent rollback window.
- `LIM_MTP_BATCH` (default `128`, 1-512): cap on the MTP draft context's batch size; lowering it saves VRAM at the cost of a slightly slower one-time prefill mirror.
- `LIM_MTP_SIDECAR`: path to a separate MTP-only GGUF (only with `LIM_MTP=1`); the MTP draft head is loaded from this file instead of the main model, for models whose GGUF quantization omits the nextn tensors. The fitter reserves the sidecar's VRAM via a margin instead of loading the draft head standalone for measurement.
- MTP mirror KV cache is set to Q8_0 regardless of the main cache type -- the most compact type the MMA_F16 flash-attention kernel dequantizes natively, so no F16 pre-conversion / `f16_extra` cost -- and the fitter's VRAM estimate uses that.
- The draft mirror KV is stored in the fast-restore cache file, so a fast restore keeps MTP active; a pre-turn canary (silent 16-token probe with bounded re-load retries, a slow re-decode fallback, and EOG-truncation guards) validates the mirror before trusting a V1 cache restore.
- An extra mid-turn checkpoint labelled `/continue` is generated when the 90% context threshold is crossed.
- `/remind`: re-sends the full system prompt to the LLM.
- Persistent browser server: the Python server is a long-lived service started on first need by the first lim session that outlives lim sessions; lim attaches to the verified running server and tears down only stale ones -- dead, or left reading a deleted/replaced FIFO node, detected by comparing the FIFO inode lim writes to against the inodes the server holds open (`/proc/<pid>/fd`).
- Browser session dividers: each new session streams a dashed "Session N" rule.
- Browser status bar: both prefill and generation progress are now shown (t/s and position).
- Startup browser-connection prompt: when the viewer is not connected, lim prints the URL and waits before the first decode; the session proceeds as soon as the viewer loads the page. Ctrl+C gives up the wait: with `LIM_OUTPUT=3` it proceeds with both outputs, otherwise browser output is disabled for the session and re-enables itself if the viewer connects later.
- `/undo` saves the re-decode's R/S checkpoint, so a later undo to the same point is instant.
- Inline KaTeX hardening under `LIM_INLINE_LATEX` (default `1`): code regions are masked from inline math and `$...$` candidates are validated before KaTeX rendering (a math delimiter never abuts whitespace, backticks never appear in math, and the candidate must parse as KaTeX), so env vars stay literal and code blocks are LaTeX-free; tool labels are excluded from auto-render; block-level `$$...$$` math always renders; set to `0` to disable.
- `LIM_TEMPERATURE=0` now omits the wall-clock timestamp from the system prompt (the only session-varying prompt content), making runs byte-reproducible; the model does not know the current date/time in this mode.
- Prompt: new rule 10 and a 0-match nudge against spurious trailing newlines in tool calls; the tool-call protocol is clarified -- the reserved tags may appear in tool call content (the parser ignores them inside parameter values) but never in prose (use `FUNC_START`/`FUNC_END` placeholders in prose, rule 7), and the raw-text / no-escaping / any-length guarantee is promoted to rule 8.
- `LIM_SEARXNG_CMD` override is now actually implemented.
- `LIM_LLAMA_DEBUG_LOG` (default `0`) is now documented in the README env-var table: with `LIM_DEBUG=1`, set it to `1` to also print DEBUG-level llama.cpp/ggml log lines (e.g. "CUDA Graph id N reused" on every decode); default keeps INFO and above.
- Mode 2 (CACHED) benchmarking improved: each turn now re-tokenizes the full canonical conversation text before the prefix match, like llama-server. At 222041 context tokens, the default Mode 0 (LIM) is 1.8% faster than Mode 2 (CACHED) and 73% faster than Mode 1 (CHATBOT).

### Bug Fixes

**Crashes & corruption**

- Token tracker restored on any failed feed (interrupt, decode error, or truncated final batch: sync `n_past_` to the KV and re-attach the decoded prefix) so ghost KV rows can no longer poison V1 cache writes; the cache write is skipped when the live KV doesn't exactly match the save's tokens.
- GGML `LOG_ERROR` lines are emitted in `dummy_log_callback` instead of swallowed, so fatal conditions (e.g. the CUDA error string printed just before `GGML_ABORT`) are visible before the abort line; WARN/INFO/DEBUG/CONT still suppressed in normal mode.

**Undo / restore / checkpoints**

- `/undo` auto-resets the session via the shared `clear_session()` when the slow-path re-decode fails or is interrupted, and the reset is documented in the README.
- The `/undo` re-decode's R/S checkpoint is saved, so a later undo to the same point is instant.
- Every V1 cache rejection is now diagnosed instead of failing silently.
- Save-file checkpoint prompts are capped at the uint16 limit with a visible `...[truncated]` marker instead of a silent cut.
- Fixed the post-`/undo` turn-close and `/continue` state machine: position-based mid-turn detection, `/continue` is a no-op mid-turn, `tool_interrupt_pending` no longer cleared at turn end, interrupt checkpoint overwritten in place at turn end.
- Corrupt or truncated fast-cache entries are now detected and deleted on encounter (the payload must exactly match the header sizes, checked in both loader and writer) instead of being re-read and rejected on every restore.

**Tool calling & session robustness**

- One self-fix retry for tool calls missing required parameters before reverting to the correction cycle; failed tool correction ejects to the prompt without rollback.
- A missing closing tag in a `path` parameter is detected via parameter-tag bleed and routed through the correction cycle.
- Newlines in `path`/`paths` parameters route through the existing tool-correction cycle; leading/trailing newlines are stripped from `path` values so line-isolated paths execute instead of failing, while internal newlines still trigger the missing-`PARAM_END` correction.
- Tool-call arg-name matching is whitespace-tolerant and leading/trailing whitespace is trimmed from `path` values (newlines are still flagged as malformed).
- `search_file` `begin`/`end` args are trimmed of whitespace (shared `trim_chars` helper).
- EOG tokens inside an open thinking block are swallowed (capped) so reasoning no longer ends the turn early; the parser treats raw function tags inside thinking blocks as prose.
- `/continue` after a mid-think interrupt resumes in thinking mode so the viewer re-enters the thinking display.

**Browser viewer**

- Inline-math validation: the redundant all-uppercase env-var check is dropped so clean pairs like `$N$` and `$H_0$` typeset, while spaced env-var pairings like "$HOME $PATH" stay literal via the whitespace rule; code blocks stream as a single `<pre>`.

**Build & environment**

- Fixed the `log_diagnostic` browser leak, a forked-child `atexit` bug, and the `exec_shell` output tail drop; removed dead code, stale comments, and the phantom `.pic.o` dep target.
- VS Code extension: reads `LIM_AI_USER` (was `AI_USER`); `export TERM_PROGRAM` is no longer needed.
- llama.cpp subrepo updated: checkpoint and UTF-8 patches updated and new patches `llama-pr28243.patch` (qwen4exp MTP graph support, upstream PR #28243) and `llama-fattn-q80.patch` (native Q8_0 K/V dequant in the CUDA fattn MMA_F16 kernel) applied.
- `make install` now also builds and installs the VS Code extension; `install-all` is now an alias for `install`.
- Uninstall: removes cached SSL certs (`combined-ca.crt`, `cloudflare-chain.pem`, `ca-bundle-temp.crt`) and skips config cleanup when `LIM_CONFIG_DIR` is the repo root or an ancestor (dev-setup safety guard); `.gitignore` now ignores the config-dir runtime data (search cache, cert caches) that lands in-tree in dev setups.

### Behavior Changes & Default Changes

| Change | Old | New |
|---|---|---|
| Browser server lifetime | Per-session child process (died with lim) | Long-lived service surviving lim's exit and crashes; lim attaches to the verified running server; `/reset` replaces it only when dead or left on a stale FIFO node |
| 90% context crossing | Warn only: "Context approaching limit -- Type '/reincarnate' ... or '/clear'" | Turn is force-ended transparently at the crossing and auto-resumed (LLM-unaware, no pause), with a permanent `/continue`-labelled checkpoint at the crossing and the prompt-labelled checkpoint at the real turn end |
| Instruction re-anchoring | User had to prompt the LLM to re-read the system prompt | Built-in `/remind` re-sends the full system prompt unescaped directly into the KV-cache |
| `/undo` slow-path failure/interrupt | Session left in an inconsistent state | Session auto-resets to a fresh one via the shared `clear_session()` |
| System-prompt timestamp | Always included | Omitted at `LIM_TEMPERATURE=0` (byte-reproducible runs) |
| `make install` | Binary + config files | Binary + config files + VS Code extension (`install-all` now an alias for `install`) |
| Version source of truth | `VERSION` file + per-file `-DLIM_VERSION` define | `version.h` (`LIM_VERSION`); the Makefile extracts it and errors if the line is missing |
| Prompt tool-call rules | Rule 5: one tool call per response | Multiple tool calls may be invoked in sequence; new rules 6-10 (closing tags, reserved FUNC tags in prose, raw-text guarantee, no newlines in paths) |
| Benchmark modes 1 & 2 | Both modes worked on saved tokens (no re-tokenization): mode 1 re-fed them and re-decoded; mode 2 paid only the prefix comparison | Reworked around a maintained canonical conversation text: mode 1 re-tokenizes and re-decodes the full text each turn like a plain chatbot; mode 2 re-tokenizes, prefix-matches, and decodes only the delta like llama-server (one-time detokenize fallback, EOG turn-boundary drift fix) |
| Fast-restore cache format | Header-less raw KV blob | `LIM_CACHE_V1` header + main KV + optional MTP mirror KV; 0.1.1-era entries can't be loaded and are deleted on encounter -- the affected save pays one slow re-decode, after which a current-format entry is written |
| `--checkpoints` restore | Skipped the automatic fast-cache write after the re-decode | A full `--checkpoints` restore auto-writes the fast cache like any other slow restore; a partial (checkpoint-selected) restore does not |

### Internals

- New `mtp.cc`/`mtp.h` module: MTP draft context, verify loop, mirror sync and self-healing, acceptance tracking, and sidecar loading with shard-safe MTP detection.
- Large DRY cleanup: duplicated trim, context-position, speed-denominator, abort-feed, latch-reset, and pipe-reopen logic consolidated into shared helpers; duplicated log-opening, system-prompt loading, speed-diagnostic, git-SHA, command-matching, and checkpoint-prompt logic unified; the unreachable newline-splitting path in `extract_array_arg_bounded` removed (the read_files gate forbids internal newlines, so comma-split is the full contract).
- The `LIM_ESCAPE_CONTRACT` append moved into `load_system_prompt_text` so `/clear` re-feeds the contract along with startup, and it is always skipped when the base prompt file is empty.
- Version moved into `version.h` as the single source of truth; the per-file `-DLIM_VERSION` define and the `VERSION` file are dropped.
- `make` also builds the VS Code extension (requires Node.js and npm); `make lim` builds only the binary.
- README inconsistencies with the code fixed and implementation details that users don't need trimmed; `REASONING.md` updated.
- All C++ sources indented to 2 spaces.

---

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
