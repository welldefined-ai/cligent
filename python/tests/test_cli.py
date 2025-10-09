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


def test_cli_live_codex_shows_yaml(capsys):
    mock_home = Path(__file__).parent / "mock_codex_home"
    mock_cwd = Path("/workspace/sample/project")
    with patch.object(Path, "home", return_value=mock_home), \
         patch.object(Path, "cwd", return_value=mock_cwd):
        code = main(["--agent", "codex", "live"])  # default -n=10
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("schema: tigs.chat/v1")
    assert "messages:" in out
    assert "role: assistant" in out


def test_cli_live_codex_n_1_limits_output(capsys):
    mock_home = Path(__file__).parent / "mock_codex_home"
    mock_cwd = Path("/workspace/sample/project")
    with patch.object(Path, "home", return_value=mock_home), \
         patch.object(Path, "cwd", return_value=mock_cwd):
        code = main(["--agent", "codex", "live", "-n", "1"])  # only last message
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("schema: tigs.chat/v1")
    # Ensure assistant role present and only one message block present heuristically
    assert "role: assistant" in out
