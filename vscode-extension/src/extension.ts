import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('lllm.workspace.start', startWorkspace)
    );

    // Handle Ctrl+J in terminal: send file separator directly to the active terminal.
    context.subscriptions.push(
        vscode.commands.registerCommand('_lllm.ctrlJ', () => {
            const terminal = vscode.window.activeTerminal;
            if (terminal) {
                terminal.sendText('\x1c', false);
            }
        })
    );
}

function startWorkspace() {
    const config = vscode.workspace.getConfiguration('lllm.workspace');
    const browserPort = config.get<number>('browserPort', 8765);
    const modelPath = config.get<string>('modelPath', '');

    const host = process.env.LLLM_HOST || getHostname();
    const viewerUrl = `http://${host}:${browserPort}/viewer.html`;

    // Wait for the server to be up before opening the browser. Uses an async
    // fetch that resolves on first success — not a busy loop, just event-driven
    // retries via setTimeout. Opens the browser only when viewer.html will load.
    waitForServer(host, browserPort).then(() => {
        vscode.commands.executeCommand('simpleBrowser.api.open', viewerUrl);
    });

    // Create an integrated terminal for the LLLM REPL and show it at the bottom.
    const lllmHost = process.env.LLLM_HOST;
    const terminal = vscode.window.createTerminal({
        name: 'LLLM',
        cwd: process.env.HOME
    });

    if (lllmHost && lllmHost !== getHostname()) {
        terminal.sendText(`ssh -t ai@${lllmHost} 'LLLM_VSCODE=1 exec bash --login'`);
    } else {
        terminal.sendText('export LLLM_VSCODE=1');
        if (modelPath) {
            terminal.sendText(`sudo -u ai -E /home/ai/bin/lllm ${modelPath}`);
        }
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
// Uses setTimeout between attempts — event-driven, not a busy loop.
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
