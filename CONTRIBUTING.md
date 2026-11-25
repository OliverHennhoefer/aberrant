# Contributing to ABERRANT

Thanks for contributing.

## Development setup

```bash
uv sync --extra dev --extra docs
```

Install local Git hooks (including pre-push checks):

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Quality gates

Run these before opening a PR:

```bash
uv run python -m ruff check .
uv run python -m mypy aberrant
uv run python -m pytest -q
uv build
```

To run integration tests directly:

```bash
uv run python -m pytest tests/integration -q
```

They check each detector's ROC-AUC and PR-AUC against floors in
`tests/integration/test_detection_quality.py`. When a change moves a detector
on purpose, measure it over the seeds 42 and 0 to 3 and set the floors 0.02
(ROC-AUC) and 0.05 (PR-AUC) below the lowest value.

Slow checks (more seeds, shuffled COVER, comparisons with river) are marked
`benchmark` and run nightly. To run them locally:

```bash
uv sync --extra dev --extra benchmark
uv run python -m pytest tests/integration -m benchmark -q
```

## PR expectations

- Keep changes scoped and reviewable.
- Add or update tests for behavior changes.
- Update docs for user-facing API changes.
- Add a changelog entry under `Unreleased` in `CHANGELOG.md`.

## Optional dependency policy

- Base test suite must pass without optional extras.
- Tests requiring optional dependencies must skip when dependency is absent.

## Commit guidance

- Use clear, imperative commit messages.
- Avoid mixing unrelated refactors with bug fixes.
