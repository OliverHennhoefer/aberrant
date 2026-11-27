# Contributing to ABERRANT

Thanks for contributing.

## Development setup

```bash
uv sync --group docs
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
uv run python -m pytest -n auto -q
uv build
```

`-n auto` runs the tests on every CPU core with pytest-xdist. Leave it out to
debug a single test. To see which lines the unit tests miss:

```bash
uv run python -m pytest tests --ignore=tests/integration -n auto --cov=aberrant --cov-report=term-missing:skip-covered -q
```

To run integration tests directly:

```bash
uv run python -m pytest tests/integration -n auto -q
```

They check each detector's ROC-AUC and PR-AUC against floors in
`tests/integration/test_detection_quality.py`. When a change moves a detector
on purpose, measure it over the seeds 42 and 0 to 3 and set the floors 0.02
(ROC-AUC) and 0.05 (PR-AUC) below the lowest value.

Slow checks (more seeds, shuffled COVER, comparisons with river) are marked
`benchmark` and run nightly. To run them locally:

```bash
uv sync --group benchmark
uv run python -m pytest tests/integration -m benchmark -q
```

## Performance work

`benchmarks/detectors.py` measures every catalog detector: events per
second, per-event latency, pickled state size, the event at which it becomes
ready, ROC-AUC, PR-AUC and a fingerprint of its exact scores. To judge a
change, edit the working tree and compare it with the last commit:

```bash
uv run python benchmarks/detectors.py --cases loda --against HEAD \
    --require-same-scores --forbid-slower --log .benchmarks/results.tsv \
    --note "what the change does"
```

`--against` alternates the two versions over several rounds and calls a case
faster or slower only when a bootstrap interval of the median speed ratio
excludes no change. `--require-same-scores` fails when any score sequence
changed, which a pure speed or memory change must not do; the exact score
fixtures in `tests/data/score_sequences/` check the same at the unit level.
Detectors whose parameters the benchmark overrides also run at their
defaults (`<id>@defaults`), and `--memory` adds what the live model retains,
caches included.

Timings are noisy on shared machines: run nothing else meanwhile, and treat
`within noise` as no gain. `--profile --cases loda` prints the hottest
functions, but cProfile inflates code that makes many small Python calls.
The repository needs uv 0.12.18 or later.

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
