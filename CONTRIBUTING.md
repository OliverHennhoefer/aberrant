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
second, per-event latency, pickled state size, traced memory, the event at
which it becomes ready, ROC-AUC, PR-AUC and a fingerprint of its exact
scores. A kept change must be a Pareto improvement: better on at least one
of these axes and worse on none, across every case it can affect. To judge
one, edit the working tree and compare it with the last commit:

```bash
uv run python benchmarks/detectors.py --cases loda --against HEAD \
    --pareto --require-same-scores --log .benchmarks/results.tsv \
    --note "what the change does"
```

`--against` starts a fresh worker process per version each round and runs
each case on both back to back, so both halves of a pair see the same
machine load; each round times a case for at least `--min-seconds` (0.25 s).
A timing counts as faster or slower only when a distribution-free 95%
interval of the median ratio excludes no change, the median moves by at least `--min-gain`
(5%) for speed or `--latency-tolerance` (25%) for p99 latency, the noisier
of the two, and as many fresh rounds agree on their own. Sizes count as
changed beyond both 2% and 4 KiB; quality counts any change in ROC-AUC or
PR-AUC; readiness counts any change in the event at which the model becomes
ready. `--pareto` exits 0 when some axis got better and none got worse, 1
when any axis of any case got worse, and 3 when nothing moved beyond the
noise. It implies `--memory`, which traces one more run per side and case.

A change that improves one axis at another's cost is a trade-off, and it is
rejected unless it is a deliberate one: pass
`--accept-tradeoff "why it is a net benefit"`, which the log records, and
spell the trade-off out in the pull request, with the numbers, for review.

`--cases` keeps the loop fast, but a change outside one detector's own
module can move every detector that imports it, so run the final check
without `--cases` (about 25 minutes for the whole catalog).
`--require-same-scores` fails when any score sequence changed, which a speed
or memory change must not do; the exact score fixtures in
`tests/data/score_sequences/` check the same at the unit level. Detectors
whose parameters the benchmark overrides also run at their defaults
(`<id>@defaults`), so a gain in one regime cannot hide a loss in the other.

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
