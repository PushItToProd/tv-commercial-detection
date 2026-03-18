# Plan: Package Restructure, Linting & Tests

**TL;DR**: Convert `server/` from a flat pile of modules into a proper `src`-layout Python package named `tv_commercial_detector`, add `ruff` for linting/formatting, and add a `pytest` suite covering pure logic, OpenCV, HTTP routes, and (mocked) LLM classification — with no real broadcast images in the repo.

---

## Phase 1 — Package Restructure

**1. Create the package skeleton**
- Create `server/src/tv_commercial_detector/` with an `__init__.py`
- Create `server/src/tv_commercial_detector/routes/` and `server/src/tv_commercial_detector/classification/` (both already have `__init__.py`s; bring them along)

**2. Move source modules into the package**
Move these into `server/src/tv_commercial_detector/`:
- Flat modules: `classify.py`, `config.py`, `state.py`, `matrix.py`, `metrics.py`, `frame_saver.py`, `main.py`
- Sub-packages: `routes/`, `classification/`
- Resource directories: `templates/`, `prompt/` (including all logos) — they move *inside* the package so `Path(__file__)` anchoring still works

**3. Update all internal imports to relative syntax** — *depends on step 2*

Every bare cross-module import in every file:

| File | Before | After |
|---|---|---|
| `main.py` | `import classify` | `from . import classify` |
| `main.py` | `from config import app_config` | `from .config import app_config` |
| `main.py` | `from routes.receive import …` | `from .routes.receive import …` |
| `classify.py` | `from classification import logo_match` | `from .classification import logo_match` |
| `matrix.py`, `frame_saver.py` | `from config import …` / `from state import …` | `from .config import …` / `from .state import …` |
| `routes/receive.py` | `from classify import classify_image` | `from ..classify import classify_image` |
| `routes/receive.py` | `from routes.status import …` | `from .status import …` |
| `routes/trigger_matrix.py` | `import matrix` | `from .. import matrix` |
| `routes/status.py`, `routes/review.py` | `from config import …` | `from ..config import …` |

**4. Update consumers outside the package** — *depends on step 3*
- `server/check_classification.py`: `from classify import …` → `from tv_commercial_detector.classify import …`
- `server/scripts/get_image_descriptions.py`: check and update if it imports from the server modules

**5. Update `pyproject.toml`** — *parallel with step 2*
- Replace `[tool.setuptools]` `py-modules` + `packages` with autodiscovery:
  ```toml
  [tool.setuptools.packages.find]
  where = ["src"]

  [tool.setuptools.package-data]
  "tv_commercial_detector" = ["templates/*", "prompt/*", "prompt/logos/*"]
  ```
- `Path(__file__)` references in `routes/status.py`, `routes/review.py`, and `classify.py` that point to `templates/` or `prompt/` need updating to reflect the new depth (e.g. `Jinja2Templates(directory=Path(__file__).parent.parent / "templates")` becomes `Path(__file__).parent.parent / "templates"` — still valid since `routes/` is one level inside the package)

**6. Update the entry point** — *depends on step 3*
- `AGENTS.md`, `server/Dockerfile`, `docker-compose.yml`: `uvicorn main:create_app` → `uvicorn tv_commercial_detector.main:create_app`

---

## Phase 2 — Ruff (lint + format)

**7. Add ruff as a dev dependency**
- `uv add --dev ruff` (adds to `[dependency-groups].dev` in `pyproject.toml`)

**8. Configure ruff in `pyproject.toml`**
- `[tool.ruff]` — `src = ["src"]`, `target-version = "py314"`, `line-length = 88`
- `[tool.ruff.lint]` — `select = ["E", "W", "F", "I", "UP"]` (errors, warnings, pyflakes, isort, pyupgrade)
- `[tool.ruff.format]` — defaults are fine

**9. Apply to existing code**
- `uv run ruff check --fix src/` — auto-fix safe violations
- `uv run ruff format src/` — reformat all files
- Manually resolve any remaining reported issues

**10. Document lint/format commands in `AGENTS.md`**

---

## Phase 3 — Tests

**11. Add test dependencies**
- `uv add --dev pytest pytest-mock httpx`

**12. Configure pytest in `pyproject.toml`**
- `[tool.pytest.ini_options]` — `testpaths = ["tests"]`; `addopts = "-v"`; define a `integration` mark

**13. Create `server/tests/` structure** — *parallel with steps 14–17*
```
server/tests/
  conftest.py            # shared fixtures: test app, tmp dirs
  test_config.py
  test_state.py
  test_classify.py
  classification/
    test_logo_match.py
    test_rectangle_match.py
  routes/
    test_receive.py
    test_status.py
    test_review.py
    test_trigger_matrix.py
  integration/
    test_check_classification.py
```

**14. Pure logic tests** — *depends on step 13*
- `test_config.py`: Load config from a JSON file; verify env-var overrides behave correctly
- `test_state.py`: State transitions (`is_ad` flipping), debounce timer logic, SSE queue management

**15. OpenCV classification tests (synthetic images)** — *depends on step 13*
- Use `PIL` to programmatically generate test images:
  - For `test_logo_match.py`: blank frame with the logo template pasted at a known position; assert match; blank frame without logo; assert no match
  - For `test_rectangle_match.py`: frame with a drawn rectangle matching the ad-bar pattern; assert detection
- **No real broadcast images** — all images generated in-test or placed as small synthetic fixtures

**16. HTTP route tests** — *depends on step 13*
- Use FastAPI's `TestClient` (via `httpx`)
- `conftest.py` creates the test app with `classify_image` patched via `pytest-mock`
- `test_receive.py`: POST a synthetic JPEG, verify state change + matrix call mocked
- `test_status.py`: Check SSE endpoint returns well-formed event stream
- `test_review.py`: Test labeling endpoints, file serving
- `test_trigger_matrix.py`: Verify manual switch calls the mocked matrix helper

**17. Mocked LLM classification test** — *depends on step 13*
- `test_classify.py`: Patch `openai.OpenAI` to return a fixed response; verify `classify_image` parses `"ad"`, `"content"`, `"unknown"` correctly and handles malformed responses

**18. Integration test scaffold for `check_classification.py`** — *depends on step 13*
- `test_check_classification.py` marked `@pytest.mark.integration` (skipped in CI by default)
- Reads a configurable directory of labeled frames (set via env var `TEST_FRAMES_DIR`)
- Requires a live llama.cpp server; skip automatically if not reachable
- Reports per-label accuracy — fail if it drops below a configurable threshold

---

## Relevant Files

- [server/pyproject.toml](server/pyproject.toml) — build system, deps, ruff + pytest config
- [server/main.py](server/main.py) — app factory, all imports updated
- [server/classify.py](server/classify.py) — classification imports + `__main__` entry
- [server/routes/receive.py](server/routes/receive.py) — most complex import chain
- [server/routes/trigger_matrix.py](server/routes/trigger_matrix.py) — `import matrix` pattern
- [server/routes/status.py](server/routes/status.py) — Jinja2 template path anchor
- [server/routes/review.py](server/routes/review.py) — Jinja2 template path anchor
- [server/check_classification.py](server/check_classification.py) — consumer outside the package
- [server/Dockerfile](server/Dockerfile) — entry point command
- [docker-compose.yml](docker-compose.yml) — service definition
- [AGENTS.md](AGENTS.md) — run commands

---

## Verification

1. `cd server && uv run uvicorn tv_commercial_detector.main:create_app --factory` starts without import errors
2. `cd server && uv run ruff check src/` — zero violations
3. `cd server && uv run pytest tests/ -v` — all unit tests pass (integration tests skipped)
4. `cd server && uv run python -m tv_commercial_detector.classify --help` — standalone classification script still works
5. `cd server && docker build .` — Docker image builds with new entry point
6. Manual smoke test: load the browser extension, send a frame, check `/status` SSE stream

---

## Decisions

- **Templates and prompt files** move *into* the package source tree so `Path(__file__)` references keep working; `package-data` is configured in `pyproject.toml` so they're included in the installable package
- **No real broadcast images** anywhere in the test suite — OpenCV tests use programmatically generated images; LLM tests use mocked API responses
- **`check_classification.py`** stays as a top-level script (not deleted) but gains a parallel pytest integration test form; images for integration testing are loaded from a local directory set by an environment variable and are never committed to the repo
- **Out of scope for this plan**: output directory reorganization (`frames/`, etc.), pydantic-settings migration, multi-broadcast prompt presets — these are listed in `todo.md` and remain separate tasks

---

## Further Considerations

1. **`scripts/get_image_descriptions.py`** — needs a quick read to verify if it imports any server modules before moving files; if it does it becomes another consumer to update
2. **Pyright/type checking** — the project already uses Pyright via VS Code; ruff's `UP` rules (pyupgrade) will modernise some type annotations. Worth deciding if `pyright` should be configured in `pyproject.toml` (`[tool.pyright]`) at the same time as ruff, or deferred
3. **`classify.py` as a runnable script** — it currently uses `argparse` and can be run directly; after the move, the recommended invocation becomes `uv run python -m tv_commercial_detector.classify`. The `if __name__ == "__main__":` block is fine as-is with relative imports
