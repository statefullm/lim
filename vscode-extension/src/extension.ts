import * as vscode from 'vscode';

// Track the viewer URL so reload can re-open it.
let currentViewerUrl: string | undefined;

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('lim.workspace.start', startWorkspace)
    );

    // Reload the Simple (integrated) Browser: close all tabs, open a fresh one.
    context.subscriptions.push(
        vscode.commands.registerCommand('lim.workspace.reload', async () => {
            if (!currentViewerUrl) return;

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

    // The server binds to loopback only, so the viewer is always reached via
    // localhost: directly on the machine, or remotely through the -L port
    // forward in the SSH command below.
    const viewerUrl = `http://localhost:${browserPort}/viewer.html`;
    currentViewerUrl = viewerUrl;

    // Wait for the server to be up before opening the browser. Uses an async
    // fetch that resolves on first success -- not a busy loop, just event-driven
    // retries via setTimeout. Opens the browser only when viewer.html will load.
    waitForServer(browserPort).then(() => {
        vscode.commands.executeCommand('simpleBrowser.api.open', viewerUrl);
        setTimeout(() => terminal.show(false), 200);
    });

    // Create an integrated terminal for the LIM REPL and show it at the bottom.
    const limHost = process.env.LIM_HOST;
    const terminal = vscode.window.createTerminal({
        name: 'LIM',
        cwd: process.env.HOME
    });

    if (limHost && limHost !== getHostname()) {
        const aiUser = process.env.LIM_AI_USER || 'ai';
        // Plain interactive SSH: the remote side starts a login shell, which
        // loads the coder alias and LIM environment from the user's rc files.
        // The -L forward makes the loopback-only viewer reachable from the
        // local browser through this session (ssh binds the local side to
        // 127.0.0.1 by default, so no LAN exposure).
        terminal.sendText(`ssh -t -a -L ${browserPort}:localhost:${browserPort} ${aiUser}@${limHost}`);
    }

    terminal.show();
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
