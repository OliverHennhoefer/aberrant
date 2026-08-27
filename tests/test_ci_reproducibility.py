"""Repository policy checks for reproducible GitHub Actions workflows."""

import re
from pathlib import Path

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
