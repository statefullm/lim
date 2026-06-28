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

    const host = process.env.LIM_HOST || getHostname();
    const viewerUrl = `http://${host}:${browserPort}/viewer.html`;
    currentViewerUrl = viewerUrl;

    // Wait for the server to be up before opening the browser. Uses an async
    // fetch that resolves on first success -- not a busy loop, just event-driven
    // retries via setTimeout. Opens the browser only when viewer.html will load.
    waitForServer(host, browserPort).then(() => {
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
        const aiUser = process.env.AI_USER || 'ai';
        terminal.sendText(`ssh -t ${aiUser}@${limHost}`);
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
// Uses setTimeout between attempts -- event-driven, not a busy loop.
function waitForServer(host: string, port: number): Promise<void> {
    return new Promise((resolve) => {
        const url = `http://${host}:${port}/status`;
        const tryFetch = () => {
            fetch(url, { mode: 'cors' }).then(() => resolve()).catch(() => {
                setTimeout(tryFetch, 1000);
            });
        };
        tryFetch();
    });
}
