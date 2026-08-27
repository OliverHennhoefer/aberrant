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
