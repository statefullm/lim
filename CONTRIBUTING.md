# Contributing to LIM

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/statefullm/lim.git
cd lim
make
./lim --version
```

See [README.md](README.md) for full build and user setup instructions.

## Versioning

- The binary's version lives in `version.h` (`LIM_VERSION`) and the VS Code extension's version in `vscode-extension/package.json` -- bump the extension version only when the extension itself changes, and `make` never rewrites either file.

## Pull Requests

1. Fork the repository and create your branch from `main`.
2. Make focused, incremental changes.
3. Ensure your code compiles cleanly with `make`.
4. Include a clear commit message describing what changed and why.
5. Open a PR against `main` with a description of your changes.

## Code Style

- Follow the existing C++ style (2-space indentation, Allman braces).
- Use meaningful variable and function names.
- Add comments for non-obvious logic, especially around KV-cache management and tool dispatch.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include your GPU model, CUDA version, and LIM version when reporting bugs.

## Areas We Need Help With

- **Testing**: More benchmarking scripts and edge-case tests.
- **Documentation**: Examples, tutorials, and FAQ entries.
- **Tool ecosystem**: New tools or improvements to existing ones.

