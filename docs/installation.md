# Installation

aberrant requires Python 3.11 or newer and is continuously tested on CPython
3.11 to 3.14, on Windows and Linux. Installation on other Python or
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

The base installation depends on NumPy alone. It is enough for all
NumPy-backed detectors, transforms, drift detectors, calibration and the
benchmark datasets.

## Add optional capabilities

| Extra | Install when you need | Added dependency |
| --- | --- | --- |
| `eval` | The evaluation examples and scikit-learn metrics | `scikit-learn` |
| `dl` | The user-supplied PyTorch `Autoencoder` | `torch` |
| `faiss` | the FAISS backend of `KNN` | `faiss-cpu` |
| `all` | Every optional capability above | all of the above |

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
    extra. `KNN(backend="faiss")` requires the `faiss` extra. Core package imports do
    not import either optional dependency.

## Verify the installation

This check uses no optional dependency:

```python
import aberrant
from aberrant.model import ThresholdModel

detector = ThresholdModel(ceiling={"temperature": 80.0})

assert detector.score_one({"temperature": 72.0}) == 0.0
assert detector.score_one({"temperature": 91.0}) == 1.0
print(f"aberrant {aberrant.__version__} is installed")
```

## Set up a development checkout

Contributor tools are dependency groups, not extras, so they never reach a
library installation. `uv sync` installs the `dev` group by default; add
`--group docs`, `--group benchmark` or `--group security` as needed.

```bash
git clone https://github.com/OliverHennhoefer/aberrant.git
cd aberrant
uv sync --group docs
uv run python -m pytest -q
uv run zensical build
```

The `docs` group includes Zensical and mkdocstrings; the `dev` group includes
the formatter, linter, type checker, test framework, PyTorch, and
scikit-learn. See the
[contribution guide](https://github.com/OliverHennhoefer/aberrant/blob/main/CONTRIBUTING.md)
for the complete quality gates.
