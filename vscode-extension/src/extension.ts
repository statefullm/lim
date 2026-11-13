import * as vscode from 'vscode';

// Track the viewer URL so reload can re-open it.
let currentViewerUrl: string | undefined;

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('lim.workspace.start', startWorkspace)
    );

    // Reload the Simple (integrated) Browser: re-read the tokenized viewer
    // URL (picking up a fresh token after a server replacement), close all
    // tabs, open a fresh one.
    context.subscriptions.push(
        vscode.commands.registerCommand('lim.workspace.reload', async () => {
            const limHost = process.env.LIM_HOST;
            const aiUser = process.env.LIM_AI_USER || 'ai';

            currentViewerUrl = await fetchViewerUrl(limHost, aiUser);
            await closeAllWebviewTabs();

            const activeTerminal = vscode.window.activeTerminal;
            vscode.commands.executeCommand('simpleBrowser.api.open', currentViewerUrl);
            if (activeTerminal) {
                setTimeout(() => activeTerminal.show(false), 200);
            }
        })
    );

    // Handle Ctrl+J in terminal: send file separator directly to the active terminal.
    context.subscriptions.push(
        vscode.commands.registerCommand('_lim.ctrlJ', () => {
            const terminal = vscode.window.activeTerminal;
            if (terminal) {
                terminal.sendText('\x1c', false);
            }
        })
    );

    // Add a status bar item with the rocket icon to quickly open the workspace.
    const statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBar.tooltip = 'Open LIM Workspace';
    statusBar.command = 'lim.workspace.start';
    statusBar.text = '$(rocket) LIM';
    statusBar.show();
    context.subscriptions.push(statusBar);
}

// Close Simple Browser tabs matching our viewer URL or title.
async function closeAllWebviewTabs(): Promise<void> {
    for (const group of vscode.window.tabGroups.all) {
        const toClose: vscode.Tab[] = [];
        for (const tab of group.tabs) {
            if (tab.label === 'LIM Viewer') {
                toClose.push(tab);
            }
        }
        for (const tab of toClose) {
            await vscode.window.tabGroups.close(tab);
        }
    }
}

function startWorkspace() {
    const config = vscode.workspace.getConfiguration('lim.workspace');
    const browserPort = config.get<number>('browserPort', 8765);
    const limHost = process.env.LIM_HOST;
    const aiUser = process.env.LIM_AI_USER || 'ai';

    // Create an integrated terminal for the LIM REPL and show it at the bottom.
    const terminal = vscode.window.createTerminal({
        name: 'LIM',
        cwd: process.env.HOME
    });

    if (limHost && limHost !== getHostname()) {
        // Plain interactive SSH: the remote side starts a login shell, which
        // loads the coder alias and LIM environment from the user's rc files.
        // The -L forward makes the loopback-only viewer reachable from the
        // local browser through this session (ssh binds the local side to
        // 127.0.0.1 by default, so no LAN exposure).
        terminal.sendText(`ssh -t -a -L ${browserPort}:localhost:${browserPort} ${aiUser}@${limHost}`);
    }

    terminal.show();

    // The server binds to loopback only, so the viewer is always reached via
    // localhost: directly on the machine, or remotely through the -L port
    // forward above. Wait for the server to be up, then open the tokenized
    // viewer URL (the bare page cannot connect to /ws without its token).
    // Uses an async fetch that resolves on first success -- not a busy loop,
    // just event-driven retries via setTimeout.
    waitForServer(browserPort).then(() => fetchViewerUrl(limHost, aiUser)).then(url => {
        currentViewerUrl = url;
        vscode.commands.executeCommand('simpleBrowser.api.open', url);
        setTimeout(() => terminal.show(false), 200);
    });
}

function localRead(path: string): string {
    const { readFileSync } = require('fs');
    return readFileSync(path, 'utf-8');
}

// Run `cat <path>` on the LIM host as aiUser and return the output.
function sshCat(limHost: string, aiUser: string, path: string): Promise<string> {
    const { execFile } = require('child_process');
    const quoted = `'${path.replace(/'/g, `'\\''`)}'`;
    return new Promise<string>((resolve, reject) => {
        execFile('ssh', ['-o', 'BatchMode=yes', `${aiUser}@${limHost}`, `cat ${quoted}`],
            { encoding: 'utf-8', timeout: 10000 },
            (err: Error | null, stdout: string) => (err ? reject(err) : resolve(stdout)));
    });
}

// Resolve the ai user's LIM_CONFIG_DIR: this process's own environment (if
// exported), else the `LIM_CONFIG_DIR=...` line in the ai user's ~/.bashrc
// (the README's documented override; the last assignment wins), else the
// default ~/.config/lim under the ai user's home.
async function resolveConfigDir(limHost: string | undefined, aiUser: string): Promise<string> {
    const def = `/home/${aiUser}/.config/lim`;
    const fromEnv = process.env.LIM_CONFIG_DIR;
    if (fromEnv && fromEnv.length > 0) return fromEnv;
    try {
        const rc = limHost && limHost !== getHostname()
            ? await sshCat(limHost, aiUser, `/home/${aiUser}/.bashrc`)
            : localRead(`/home/${aiUser}/.bashrc`);
        let value: string | undefined;
        for (const raw of rc.split('\n')) {
            const t = raw.trim().replace(/^export\s+/, '');
            if (!t.startsWith('LIM_CONFIG_DIR=')) continue;
            const v = t.slice('LIM_CONFIG_DIR='.length).trim();
            const q = v.match(/^(["'])([^"']*)\1/);
            value = q ? q[2] : v.split(/\s/)[0];
        }
        if (value && value.length > 0) return value;
    } catch {
        // rc file missing or unreadable: fall back to the default.
    }
    return def;
}

// The server writes the tokenized viewer URL (mode 0640, readable by the
// sharing user) to LIM_CONFIG_DIR/lim.viewer_url at startup. Read it:
// directly from disk when local, over SSH when remote. Falls back to the
// bare page, which tells the user to load the URL lim printed.
async function fetchViewerUrl(limHost: string | undefined, aiUser: string): Promise<string> {
    const config = vscode.workspace.getConfiguration('lim.workspace');
    const bare = `http://localhost:${config.get<number>('browserPort', 8765)}/viewer.html`;
    try {
        const file = `${await resolveConfigDir(limHost, aiUser)}/lim.viewer_url`;
        const content = limHost && limHost !== getHostname()
            ? await sshCat(limHost, aiUser, file)
            : localRead(file);
        const url = content.trim();
        if (url.includes('#token=')) return url;
    } catch {
        // File missing or unreadable: fall through to the bare page.
    }
    return bare;
}

function getHostname(): string {
    try {
        const { execSync } = require('child_process');
        return execSync('hostname', { encoding: 'utf-8' }).trim().split('.')[0];
    } catch {
        return 'localhost';
    }
}

export function deactivate() {}

// Wait for the server to respond via /status. Resolves on first success.
// localhost: the server binds to loopback only, so it is only reachable here
// on the machine itself or through an SSH port forward. Uses setTimeout
// between attempts -- event-driven, not a busy loop.
function waitForServer(port: number): Promise<void> {
    return new Promise((resolve) => {
        const url = `http://localhost:${port}/status`;
        const tryFetch = () => {
            fetch(url, { mode: 'cors' }).then(() => resolve()).catch(() => {
                setTimeout(tryFetch, 1000);
            });
        };
        tryFetch();
    });
}
