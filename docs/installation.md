# Installation

ABERRANT requires Python 3.10 or newer and is continuously tested on CPython
3.10, 3.11, and 3.12, on Windows and Linux. Installation on other Python or
operating-system versions depends on wheels for the required dependencies and
any selected optional extras; it is not part of the current CI matrix.

## Install the base package

=== "pip"

    ```bash
    python -m pip install aberrant
    ```

=== "uv"

    ```bash
    uv pip install aberrant
    ```

The base installation includes NumPy, SciPy, tqdm, and the dataset cache's file
locking dependency. It is enough for all NumPy-backed detectors, transforms,
drift detectors, and dataset streaming.

## Add optional capabilities

| Extra | Install when you need | Added dependency |
| --- | --- | --- |
| `eval` | The evaluation examples and scikit-learn metrics | `scikit-learn` |
| `dl` | The user-supplied PyTorch `Autoencoder` | `torch` |
| `faiss` | `FaissSimilaritySearchEngine`, commonly used with `KNN` | `faiss-cpu` |

Extras can be combined in one installation:

=== "pip"

    ```bash
    python -m pip install "aberrant[eval,faiss]"
    ```

=== "uv"

    ```bash
    uv pip install "aberrant[eval,faiss]"
    ```

!!! important "Optional imports are explicit"

    `Autoencoder` and the neural-network `Architecture` base require the `dl`
    extra. The FAISS engine requires the `faiss` extra. Core package imports do
    not import either optional dependency.

## Verify the installation

This check uses no optional dependency:

```python
import aberrant
from aberrant.model import ThresholdModel

detector = ThresholdModel(ceiling={"temperature": 80.0})

assert detector.score_one({"temperature": 72.0}) == 0.0
assert detector.score_one({"temperature": 91.0}) == 1.0
print(f"ABERRANT {aberrant.__version__} is installed")
```

## Set up a development checkout

The `dev`, `docs`, `benchmark`, and `all` extras are contributor toolchains,
not runtime requirements for library users.

```bash
git clone https://github.com/OliverHennhoefer/aberrant.git
cd aberrant
uv sync --extra dev --extra docs
uv run python -m pytest -q
uv run zensical build
```

The documentation extra includes Zensical and mkdocstrings; the development
extra includes the formatter, linter, type checker, test framework, PyTorch,
and scikit-learn. See the
[contribution guide](https://github.com/OliverHennhoefer/aberrant/blob/main/CONTRIBUTING.md)
for the complete quality gates.
