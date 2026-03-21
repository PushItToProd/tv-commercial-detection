---
description: "Diagnose and fix failing unit tests in the server test suite"
agent: "agent"
argument-hint: "Optional: specific test file, test name, or error message"
tools: [vscode, execute, read, agent, edit, search, web, browser, 'pylance-mcp-server/*', vscode.mermaid-chat-features/renderMermaidDiagram, ms-azuretools.vscode-containers/containerToolsConfig, ms-python.python/getPythonEnvironmentInfo, ms-python.python/getPythonExecutableCommand, ms-python.python/installPythonPackage, ms-python.python/configurePythonEnvironment, ms-toolsai.jupyter/configureNotebook, ms-toolsai.jupyter/listNotebookPackages, ms-toolsai.jupyter/installNotebookPackages, todo]
---

Run the failing tests, diagnose the root cause, and fix them.

## Steps

1. **Run the tests** to capture current failures:
   ```
   cd server && uv run pytest tests/ -m "not integration" -x --tb=short 2>&1
   ```
   If the user specified a particular test file or name, scope the run to that target.

2. **Diagnose** each failure:
   - Read the full traceback
   - Read the relevant source file(s) and test file(s) to understand the expected vs. actual behaviour
   - Determine whether the bug is in the **implementation** or the **test** itself

3. **Fix** the root cause:
   - Prefer fixing implementation bugs over updating tests to match broken behaviour
   - Only update a test assertion if the test expectation was genuinely wrong (e.g. the spec changed)
   - Keep changes minimal — don't refactor surrounding code

4. **Verify** the fix:
   ```
   cd server && uv run pytest tests/ -m "not integration" --tb=short 2>&1
   ```
   Confirm all previously-failing tests now pass and no new failures were introduced.

## Project notes

- Package manager: `uv` (never `pip`)
- Test runner: `pytest` with marker `not integration` for unit tests
- Test directory: `server/tests/`
- Source root: `server/src/tv_commercial_detector/`
