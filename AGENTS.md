# Repository Guidelines

## Project Structure & Module Organization
- Current layout is minimal: `model/` contains YOLOv8 weight files (`yolov8n.pt`, `yolov8n_1.pt`). Keep new model artifacts in this folder or subfolders named by task (e.g., `model/detect/`).
- When adding code, place reusable logic under `src/` and entry points or scripts under `scripts/`. Add dataset samples (if needed) under `data/` with README notes; avoid committing large/raw datasets.
- Put experiments in `notebooks/` and keep them lightweight (strip outputs before committing). Keep configuration in versioned YAML/JSON under `configs/`.

## Build, Test, and Development Commands
- Create a virtual environment before development: `python -m venv .venv` then activate and `pip install -r requirements.txt` (add the file alongside new code).
- Run static checks when available: `ruff .` and `black .` (configure in `pyproject.toml`).
- Execute tests with `pytest` from the repo root: `pytest -q` for quick runs, `pytest --maxfail=1 --ff` to debug failures.
- If you add training/inference scripts, provide a usage example in their module docstring plus a short `scripts/README.md`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants. Prefer type hints and docstrings for public functions.
- Keep modules small and focused; avoid mixing training, data prep, and inference utilities in the same file.
- Use descriptive filenames (`loader_coco.py`, `augmentations.py`) and keep configuration-driven values out of code by reading from YAML/JSON.

## Testing Guidelines
- Use `pytest` with files named `test_*.py`; mirror package structure (e.g., `tests/model/test_loader.py` for `src/model/loader.py`).
- Provide small, deterministic fixtures; avoid loading full model weights in unit tests—stub or mock instead. Add integration tests that load weights only when necessary.
- Aim for coverage of new logic; include regression tests when fixing bugs.

## Commit & Pull Request Guidelines
- No existing history—adopt Conventional Commits (`feat: add data loader`, `fix: handle empty labels`). Keep commits small and focused.
- PRs should include: purpose, key changes, test commands/output, and any performance or size impacts (e.g., new model weights). Link issues/tasks when available.
- Before opening a PR, run lint + tests locally and ensure large artifacts are tracked with Git LFS or excluded via `.gitignore`.

## Security & Data Handling
- Do not commit private datasets, credentials, or API keys. Prefer environment variables for secrets.
- Large binaries (new weights/checkpoints) should use Git LFS or be referenced via download instructions rather than committed directly when possible.
