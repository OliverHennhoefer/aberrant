"""Every aberrant import shown in the README and the documentation resolves."""

import ast
import importlib
import re
from pathlib import Path

import pytest

from aberrant.exceptions import MissingOptionalDependencyError

ROOT = Path(__file__).parents[1]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
BLOCK = re.compile(r"^( *)```python[^\n]*\n(.*?)^\1```", re.DOTALL | re.MULTILINE)


def _imports(page: Path) -> list[tuple[str, str]]:
    found = []
    for indent, code in BLOCK.findall(page.read_text(encoding="utf-8")):
        lines = [line[len(indent) :] for line in code.splitlines()]
        for node in ast.walk(ast.parse("\n".join(lines))):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.split(".")[0] == "aberrant"
            ):
                found.extend((node.module, alias.name) for alias in node.names)
    return found


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(ROOT)))
def test_documented_imports_resolve(page: Path) -> None:
    for module, name in _imports(page):
        try:
            getattr(importlib.import_module(module), name)
        except MissingOptionalDependencyError:
            continue
        except (ImportError, AttributeError):
            pytest.fail(f"`from {module} import {name}` does not resolve")
