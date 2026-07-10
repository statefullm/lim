# Contributing to LIM

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/statefullm/lim.git
cd lim
make
./lim --help
```

See [README.md](README.md) for full build and user setup instructions.

## Pull Requests

1. Fork the repository and create your branch from `main`.
2. Make focused, incremental changes.
3. Ensure your code compiles cleanly with `make`.
4. Include a clear commit message describing what changed and why.
5. Open a PR against `main` with a description of your changes.

## Code Style

- Follow the existing C++ style (4-space indentation, Allman braces).
- Use meaningful variable and function names.
- Add comments for non-obvious logic, especially around KV-cache management and tool dispatch.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include your GPU model, CUDA version, and LIM version when reporting bugs.

## Areas We Need Help With

- **Testing**: More benchmarking scripts and edge-case tests.
- **Documentation**: Examples, tutorials, and FAQ entries.
- **Tool ecosystem**: New tools or improvements to existing ones.

