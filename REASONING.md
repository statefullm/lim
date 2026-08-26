
# LIM Reasoning Modes (think / instruct / quick)

Setup guide for Qwen 3.8 (or any model that steers reasoning effort via
prompt text). After this setup, every new shell (SSH, VS Code, `su`)
automatically restores the sampling env that matches your `localprompt`,
so the env is always consistent with what the model will read.

Example Modes:

| Mode     | localprompt content        | Sampling env                    | Use for                        |
|----------|----------------------------|---------------------------------|--------------------------------|
| `think`  | xhigh instructions + rules | temp 1.0, top_p 0.95, presence 0 | Hard problems, deep reasoning |
| `instruct` | rules only               | temp 0.7, top_p 0.8, presence 1.5 | Normal chat/coding          |
| `quick`  | low instructions + rules   | temp 0.7, top_p 0.8, presence 1.5 | Simple tasks, fast answers  |

`localprompt` is the single source of truth: the mode is derived from
its text (contains the xhigh text -> think, contains the low text ->
quick, otherwise -> instruct).

Assumes `LIM_CONFIG_DIR` is exported in `~/.bashrc` (default
`~/.config/lim`). All files below go
in `$LIM_CONFIG_DIR` unless stated otherwise.

---

## 1. Prompt text fragments

**`xhigh`** -- the xhigh reasoning-effort instruction (one line):

```
Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.
```

**`low`** -- the low reasoning-effort instruction (one line):

```
Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.
```

**`gpu`** -- your persistent rules, appended to every mode (content and
name are yours; this example is a GPU usage rule):

```
**CRITICAL RULE**
NEVER RUN ANYTHING ON THE GPU YOURSELF: ASK THE USER. NEVER KILL GPU processes!
```

## 2. Per-mode env files

**`think.env`**:

```bash
export LIM_TEMPERATURE=1.0
export LIM_TOP_P=0.95
export LIM_PRESENCE_PENALTY=0.0
```

**`instruct.env`** and **`quick.env`** (identical; they equal the
built-in defaults but are kept explicit):

```bash
export LIM_TEMPERATURE=0.7
export LIM_TOP_P=0.8
export LIM_PRESENCE_PENALTY=1.5
```

## 3. Mode files

Each mode file writes `localprompt` and sources its env file.

**`think`**:

```bash
cat "$LIM_CONFIG_DIR/xhigh" "$LIM_CONFIG_DIR/gpu" > "$LIM_CONFIG_DIR/localprompt"
source "$LIM_CONFIG_DIR/think.env"
```

**`instruct`**:

```bash
cat "$LIM_CONFIG_DIR/gpu" > "$LIM_CONFIG_DIR/localprompt"
source "$LIM_CONFIG_DIR/instruct.env"
```

**`quick`**:

```bash
cat "$LIM_CONFIG_DIR/low" "$LIM_CONFIG_DIR/gpu" > "$LIM_CONFIG_DIR/localprompt"
source "$LIM_CONFIG_DIR/quick.env"
```

## 4. Login hook

Create the hook directory (if you don't have the `.bashrc.d` loop in
`~/.bashrc`, just add `. ~/.bashrc.d/lim` near the end of `.bashrc`
instead):

```bash
mkdir -p ~/.bashrc.d
```

**`~/.bashrc.d/lim`**:

```bash
_lim_restore_mode() {
    local cfg="${LIM_CONFIG_DIR:-$HOME/.config/lim}"
    local lp="$cfg/localprompt"
    [ -f "$lp" ] || return 0

    local mode=instruct
    if [ -f "$cfg/xhigh" ] && grep -qFf "$cfg/xhigh" "$lp" 2>/dev/null; then
        mode=think
    elif [ -f "$cfg/low" ] && grep -qFf "$cfg/low" "$lp" 2>/dev/null; then
        mode=quick
    fi

    [ -f "$cfg/$mode.env" ] || return 0
    . "$cfg/$mode.env"
    echo "Restoring $mode mode"
}

# One-word mode switchers. Must run in the current shell, so they are
# aliases (or functions), not executable scripts: a child process
# cannot change this shell's environment.
alias think='source $LIM_CONFIG_DIR/think'
alias instruct='source $LIM_CONFIG_DIR/instruct'
alias quick='source $LIM_CONFIG_DIR/quick'

# Only in real terminals (SSH, su). lim's exec_shell sources .bashrc on
# every command to inject the env, so without this guard the hook would
# run (and print) inside every LLM tool call. VS Code terminals
# (TERM_PROGRAM=vscode) are re-exec'd shells whose original shell already
# restored the mode: skip to avoid a duplicate banner.
if [ -t 1 ] && [ "${TERM_PROGRAM:-}" != "vscode" ]; then
    _lim_restore_mode
fi
```

---

## Usage

- **Switch mode** (deliberate, rewrites `localprompt`):

  ```bash
  think                            # or instruct / quick
  coder                            # your launch alias
  ```

- **At login** (automatic): the hook prints `Restoring <mode> mode`
  and exports the matching env. `coder` is then always consistent with
  the `localprompt` that will be baked in.

## Caveat

- If you edit the text of `xhigh` or `low`, re-source the active mode
  so `localprompt` matches again; otherwise the next login falls back
  to `instruct` (env follows the text; your edited prompt is preserved).

