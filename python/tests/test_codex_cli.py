"""Unit tests for Codex CLI agent implementation."""

import json
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

import pytest

from src.agents.codex_cli.core import CodexRecord, CodexLogFile, CodexLogStore, CodexCligent
from src.core.models import Role, Chat
from src import ChatParser


class TestCodexRecord:
    """Tests for CodexRecord behaviour."""

    def test_extracts_user_message(self):
        json_data = {
            "timestamp": "2025-10-01T12:00:01.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello Codex"}
                ],
            },
        }
        record = CodexRecord.load(json.dumps(json_data))

        message = record.extract_message()

        assert message is not None
        assert message.role == Role.USER
        assert message.content == "Hello Codex"
        assert message.provider == "codex-cli"
        assert message.timestamp == datetime.fromisoformat("2025-10-01T12:00:01+00:00")

    def test_skips_non_message_payloads(self):
        json_data = {
            "timestamp": "2025-10-01T12:00:01.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "shell",
                'arguments': '{"command":["ls"]}',
            },
        }
        record = CodexRecord.load(json.dumps(json_data))

        assert record.extract_message() is None


class TestCodexLogFile:
    """Tests for CodexLogFile parsing."""

    @pytest.fixture
    def test_data_path(self):
        return Path(__file__).parent / "test_data" / "codex_cli"

    def test_parses_simple_chat(self, test_data_path):
        log_file = CodexLogFile(file_path=test_data_path / "simple_chat.jsonl")
        log_file.load()
        chat = log_file.to_chat(log_uri="simple_chat.jsonl")

        assert isinstance(chat, Chat)
        assert len(chat.messages) == 2
        assert chat.messages[0].role == Role.USER
        assert chat.messages[1].role == Role.ASSISTANT
        assert "hello world" in chat.messages[1].content.lower()

    def test_ignores_non_message_entries(self, test_data_path):
        log_file = CodexLogFile(file_path=test_data_path / "mixed_entries.jsonl")
        log_file.load()
        chat = log_file.to_chat(log_uri="mixed_entries.jsonl")

        assert len(chat.messages) == 2
        assert chat.messages[0].role == Role.USER
        assert chat.messages[1].role == Role.ASSISTANT


class TestCodexLogStore:
    """Tests for CodexLogStore listing and resolution."""

    @pytest.fixture
    def mock_home(self):
        return Path(__file__).parent / "mock_codex_home"

    def test_lists_nested_session_logs(self, mock_home):
        with patch.object(Path, "home", return_value=mock_home):
            store = CodexLogStore()
            logs = store.list()

        assert len(logs) >= 2
        uris = [uri for uri, _ in logs]
        assert any("2025/10/01" in uri for uri in uris)
        assert any("2025/10/02" in uri for uri in uris)

    def test_resolves_relative_uri(self, mock_home):
        with patch.object(Path, "home", return_value=mock_home):
            store = CodexLogStore()
            uri = "2025/10/02/rollout-2025-10-02T09-30-00-uvwxyz.jsonl"
            resolved = store._resolve_log_path(uri)

        assert resolved.exists()
        assert resolved.name == "rollout-2025-10-02T09-30-00-uvwxyz.jsonl"


class TestCodexCligent:
    """Integration-style tests for the Codex agent."""

    @pytest.fixture
    def mock_home(self):
        return Path(__file__).parent / "mock_codex_home"

    def test_parse_with_relative_uri(self, mock_home):
        mock_cwd = Path("/workspace/sample/project")
        with patch.object(Path, "home", return_value=mock_home), \
             patch.object(Path, "cwd", return_value=mock_cwd):
            parser = ChatParser("codex")
            uri = "2025/10/01/rollout-2025-10-01T12-00-00-abcdef.jsonl"
            chat = parser.parse(uri)

        assert isinstance(chat, Chat)
        assert len(chat.messages) == 2
        assert chat.messages[0].role == Role.USER
        assert chat.messages[1].role == Role.ASSISTANT

    def test_live_log_returns_latest(self, mock_home):
        mock_cwd = Path("/workspace/sample/project")
        with patch.object(Path, "home", return_value=mock_home), \
             patch.object(Path, "cwd", return_value=mock_cwd):
            parser = ChatParser("codex")
            chat = parser.parse()

        assert chat is not None
        assert any(msg.role == Role.ASSISTANT for msg in chat.messages)
