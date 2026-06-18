import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('lllm.workspace.start', startWorkspace)
    );
}

function startWorkspace() {
    const config = vscode.workspace.getConfiguration('lllm.workspace');
    const browserPort = config.get<number>('browserPort', 8765);
    const modelPath = config.get<string>('modelPath', '');

    // Get the actual hostname so the viewer's WebSocket connects correctly.
    const host = process.env.LLLM_HOST || getHostname();
    const viewerUrl = `http://${host}:${browserPort}/viewer.html`;

    // Open in VS Code's integrated browser with the viewer URL.
    vscode.commands.executeCommand('simpleBrowser.show', viewerUrl);

    // Create an integrated terminal for the LLLM REPL and show it at the bottom.
    const terminal = vscode.window.createTerminal({
        name: 'LLLM',
        cwd: '/home/ai/lllm'
    });

    if (modelPath) {
        terminal.sendText(`sudo -u ai -E /home/ai/bin/lllm ${modelPath}`);
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
