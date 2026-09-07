"""Repository policy checks for reproducible GitHub Actions workflows."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from aberrant import __version__

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
ACTION_USE = re.compile(r"^\s*uses:\s*[^@\s]+@([^\s#]+)", re.MULTILINE)
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def _workflow_texts() -> list[str]:
    return [path.read_text(encoding="utf-8") for path in WORKFLOWS.glob("*.yml")]


def test_all_third_party_actions_are_pinned_to_commit_shas() -> None:
    references = [
        reference
        for workflow in _workflow_texts()
        for reference in ACTION_USE.findall(workflow)
    ]

    assert references
    assert all(COMMIT_SHA.fullmatch(reference) for reference in references)


def test_all_uv_sync_commands_use_the_lockfile() -> None:
    sync_commands = [
        line.strip()
        for workflow in _workflow_texts()
        for line in workflow.splitlines()
        if "uv sync" in line
    ]

    assert sync_commands
    assert all("--locked" in command for command in sync_commands)


def _run_release_version_check(tag: str) -> subprocess.CompletedProcess[str]:
    workflow = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    check_step = workflow.split("- name: Validate release version\n", 1)[1].split(
        "- name:", 1
    )[0]
    assert "if: startsWith(github.ref, 'refs/tags/')" in check_step
    assert workflow.index("name: Validate release version") < workflow.index(
        "name: Build distributions"
    )
    script_match = re.search(
        r"<<'PY'\n(.*?)^\s+PY$", check_step, re.MULTILINE | re.DOTALL
    )
    assert script_match is not None
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script_match.group(1))],
        cwd=WORKFLOWS.parents[1],
        env={**os.environ, "GITHUB_REF_NAME": tag},
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_accepts_the_package_version_tag() -> None:
    result = _run_release_version_check(f"v{__version__}")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("tag", ["v0.0.0", __version__, f"v{__version__}rc1"])
def test_release_rejects_mismatched_version_tags(tag: str) -> None:
    result = _run_release_version_check(tag)

    assert result.returncode != 0
    assert "does not match package version" in result.stderr
    assert f"expected 'v{__version__}'" in result.stderr
