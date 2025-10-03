"""CLI smoke tests for cligent command behavior.

These tests exercise non-interactive subcommands using the Codex mock
home to ensure deterministic output without requiring real sessions.
"""

from pathlib import Path
from unittest.mock import patch

from src.cli import main


def test_cli_list_codex(monkeypatch):
    mock_home = Path(__file__).parent / "mock_codex_home"
    mock_cwd = Path("/workspace/sample/project")
    with patch.object(Path, "home", return_value=mock_home), \
         patch.object(Path, "cwd", return_value=mock_cwd):
        # Should return exit code 0 and print at least one URI
        code = main(["--agent", "codex", "list"])
        assert code == 0


def test_cli_parse_codex(monkeypatch, capsys):
    mock_home = Path(__file__).parent / "mock_codex_home"
    mock_cwd = Path("/workspace/sample/project")
    uri = "2025/10/01/rollout-2025-10-01T12-00-00-abcdef.jsonl"
    with patch.object(Path, "home", return_value=mock_home), \
         patch.object(Path, "cwd", return_value=mock_cwd):
        code = main(["--agent", "codex", "parse", uri])
    captured = capsys.readouterr()
    assert code == 0
    assert "Parsed 2 messages" in captured.out

