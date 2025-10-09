"""Tests for Claude Code recursive listing behavior.

These tests verify that when treating CWD as a prefix, list(recursive=True)
aggregates logs from all project folders whose names start with the current
project prefix, and that recursive=False restricts to the exact project.

Both modes use filenames only (no relative paths) for log URIs to maintain
consistency.
"""

from pathlib import Path
from unittest.mock import patch

from src.agents.claude_code.core import ClaudeLogStore


def _write_jsonl(file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        "\n".join(
            [
                '{"timestamp":"2025-01-01T00:00:00Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Hi"}]}}',
                '{"timestamp":"2025-01-01T00:00:01Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}}',
            ]
        )
    )


def test_claude_recursive_listing(tmp_path: Path) -> None:
    # Arrange projects root
    projects_root = tmp_path / ".claude" / "projects"

    # Base project is '...-python'; sub project is '...-python-utils'
    cwd_python = Path("/home/user/projects/myproject/python")
    base = str(cwd_python).replace("/", "-")
    proj_base = projects_root / base
    proj_utils = projects_root / f"{base}-utils"

    f1 = proj_base / "log1.jsonl"
    f2 = proj_utils / "log2.jsonl"
    _write_jsonl(f1)
    _write_jsonl(f2)

    # When recursive=True (default), both logs are listed using filenames only
    with patch.object(Path, "home", return_value=tmp_path), \
         patch.object(Path, "cwd", return_value=cwd_python):
        store = ClaudeLogStore()
        logs = store.list()  # default recursive

    uris = {uri for uri, _ in logs}
    # Both files should be listed with filename only (no relative path)
    assert "log1.jsonl" in uris
    assert "log2.jsonl" in uris
    # Verify no path separators in URIs (consistent with non-recursive mode)
    for uri, _ in logs:
        assert "/" not in uri and "\\" not in uri


def test_claude_nonrecursive_listing(tmp_path: Path) -> None:
    # Arrange projects root and a specific sub-project (python)
    projects_root = tmp_path / ".claude" / "projects"
    cwd_python = Path("/home/user/projects/myproject/python")
    base = str(cwd_python).replace("/", "-")
    proj_python = projects_root / base

    f1 = proj_python / "only.jsonl"
    _write_jsonl(f1)

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.object(Path, "cwd", return_value=cwd_python):
        store = ClaudeLogStore()
        logs = store.list(recursive=False)

    # Non-recursive returns filenames (no path separators)
    assert logs, "Expected at least one log in non-recursive listing"
    for uri, _ in logs:
        assert "/" not in uri and "\\" not in uri
        assert uri == f1.name
