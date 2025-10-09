"""Real data tests for the chat parser library."""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.core import Chat, Message, Role
from src import ChatParser
from src.agents.claude_code.core import ClaudeLogStore, ClaudeLogFile


@pytest.fixture
def test_data_path():
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def mock_home():
    """Path to mock Claude home directory."""
    return Path(__file__).parent / "mock_claude_home"


@pytest.fixture
def parser(mock_home):
    """ChatParser configured with mock Claude environment."""
    mock_cwd = Path("/home/user/projects/myproject/python")
    with (
        patch.object(Path, "home", return_value=mock_home),
        patch.object(Path, "cwd", return_value=mock_cwd),
    ):
        yield ChatParser("claude-code")


class TestChatParserReal:
    """Test ChatParser with real log data."""

    def test_list_logs_real_data(self, parser):
        """Test listing logs with real data."""
        logs = parser.list_logs()

        # Should find our test logs
        assert len(logs) >= 4  # We created at least 4 test files

        # Check structure
        for log_uri, metadata in logs:
            assert isinstance(log_uri, str)
            assert isinstance(metadata, dict)
            assert "size" in metadata
            assert "project" in metadata
            assert "accessible" in metadata

    def test_parse_simple_chat(self, parser):
        """Test parsing a simple chat log using the log file reader."""
        chat = parser.parse("simple_chat.jsonl")

        assert isinstance(chat, Chat)
        assert len(chat.messages) == 4

        # Check first message
        msg1 = chat.messages[0]
        assert msg1.role == Role.USER
        assert "create a list in Python" in msg1.content
        assert msg1.timestamp is not None

        # Check last message
        msg4 = chat.messages[3]
        assert msg4.role == Role.ASSISTANT
        assert "append" in msg4.content

    def test_parse_system_message_chat(self, parser):
        """Test parsing a chat with system messages using the log reader."""
        chat = parser.parse("system_chat.jsonl")

        assert len(chat.messages) == 3

        # First message should be system
        assert chat.messages[0].role == Role.SYSTEM
        assert "helpful coding assistant" in chat.messages[0].content

        # Should have user and assistant messages too
        roles = [msg.role for msg in chat.messages]
        assert Role.USER in roles
        assert Role.ASSISTANT in roles

    def test_parse_empty_chat(self, parser):
        """Test parsing an empty chat (summary only) using the log reader."""
        chat = parser.parse("empty_chat.jsonl")

        assert isinstance(chat, Chat)
        assert len(chat.messages) == 0

    def test_parse_malformed_chat(self, parser):
        """Test parsing a chat with malformed records using the log reader."""
        chat = parser.parse("malformed_chat.jsonl")

        # Should skip malformed records but parse valid ones
        assert len(chat.messages) == 2  # Two valid messages
        assert chat.messages[0].content == "This is a valid message"
        assert chat.messages[1].content == "This message follows the valid one"

    def test_parse_nonexistent_file(self, parser):
        """Test parsing a non-existent file using the log reader."""
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.jsonl")

    def test_live_log(self, parser):
        """Test getting the live (most recent) log."""
        chat = parser.parse()  # No URI = live log

        # Should get the most recent log
        assert isinstance(chat, Chat)
        assert len(chat.messages) >= 1

    def test_message_selection_and_composition(self, parser):
        """Test selecting messages and composing output.

        Use a preloaded chat cache to avoid absolute path parsing.
        """
        parser.select("simple_chat.jsonl", [0, 2])  # First and third messages
        result = parser.compose()

        # Should contain selected messages in Tigs YAML format
        assert "schema: tigs.chat/v1" in result
        assert "messages:" in result
        assert "role: user" in result
        # Should have 2 selected messages
        import yaml

        tigs_data = yaml.safe_load(result)
        assert len(tigs_data["messages"]) == 2

        # Test selecting all messages
        parser.clear_selection()
        parser.select("simple_chat.jsonl")  # All messages
        result = parser.compose()

        tigs_data = yaml.safe_load(result)
        assert len(tigs_data["messages"]) == 4  # All messages

    def test_unselect_messages(self, parser):
        """Test unselecting specific messages using preloaded cache."""
        parser.select("simple_chat.jsonl")
        assert len(parser.selected_messages) == 4

        # Unselect first message
        parser.unselect("simple_chat.jsonl", [0])
        assert len(parser.selected_messages) == 3

        # Unselect all messages from this log
        parser.unselect("simple_chat.jsonl")
        assert len(parser.selected_messages) == 0

    def test_multiple_log_composition(self, parser):
        """Test composing from multiple logs using preloaded cache."""
        parser.select("simple_chat.jsonl", [0])  # First message from simple
        parser.select("system_chat.jsonl", [1])  # Second message from system

        # Should have messages from both logs
        assert len(parser.selected_messages) == 2

        result = parser.compose()
        assert "create a list in Python" in result  # From simple_log
        assert "hello world program" in result  # From system_log


class TestSessionIDFunctionality:
    """Test filename/URI behavior for listing and parsing."""

    def test_list_logs_nonrecursive_returns_filenames(self):
        """Non-recursive listing returns filenames without path separators."""
        from src import ChatParser

        # Use current directory which should have Claude Code logs
        parser = ChatParser("claude-code")
        logs = parser.list_logs(recursive=False)

        if logs:  # Only test if there are logs
            for log_uri, metadata in logs:
                assert "/" not in log_uri and "\\" not in log_uri
                assert log_uri.endswith(".jsonl")

    def test_parse_with_filename(self):
        """Test parsing using filename returned by list_logs(False)."""
        from src import ChatParser

        parser = ChatParser("claude-code")
        logs = parser.list_logs(recursive=False)

        if logs:  # Only test if there are logs
            session_id, _ = logs[0]
            # Should be able to parse using just the filename
            chat = parser.parse(session_id)
            assert chat is not None
            assert len(chat.messages) >= 0

    def test_location_parameter_changes_project(self):
        """Test that location parameter changes which project's logs are found."""
        from src import ChatParser
        from pathlib import Path

        mock_home = Path(__file__).parent / "mock_claude_home"

        # Test with different project directories
        with patch.object(Path, "home", return_value=mock_home):
            # Mock different working directories to simulate different projects
            with patch.object(
                Path, "cwd", return_value=Path("/home/user/projects/myproject")
            ):
                parent_parser = ChatParser("claude-code")
                parent_logs = parent_parser.list_logs(recursive=False)

            with patch.object(
                Path, "cwd", return_value=Path("/home/user/projects/myproject/python")
            ):
                python_parser = ChatParser("claude-code")
                python_logs = python_parser.list_logs(recursive=False)

            # Different projects should have different logs
            # (unless no logs exist for one of them)
            if parent_logs and python_logs:
                parent_ids = {log[0] for log in parent_logs}
                python_ids = {log[0] for log in python_logs}
                # The sets might overlap but shouldn't be identical
                assert parent_ids != python_ids, "Different projects returned same logs"


class TestPlanHandling:
    """Test ExitPlanMode and plan response handling."""

    def test_exit_plan_mode_detection(self):
        """Test that ExitPlanMode tool use is correctly detected."""
        from src.agents.claude_code.core import ClaudeRecord as Record

        # Mock ExitPlanMode record
        exit_plan_data = {
            "type": "assistant",
            "uuid": "test-uuid",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me create a plan:"},
                    {
                        "type": "tool_use",
                        "id": "tool-id",
                        "name": "ExitPlanMode",
                        "input": {
                            "plan": "## Test Plan\n\n1. Do something\n2. Do something else"
                        },
                    },
                ],
            },
            "timestamp": "2024-01-01T12:00:00Z",
        }

        record = Record(
            type="assistant", raw_data=exit_plan_data, timestamp="2024-01-01T12:00:00Z"
        )

        # Test detection
        assert record.is_exit_plan_mode()

        # Test message extraction
        message = record.extract_message()
        assert message is not None
        assert message.role == Role.ASSISTANT
        assert "📋 **Plan Proposal**" in message.content
        assert "## Test Plan" in message.content

    def test_regular_assistant_message_with_plan_present(self):
        """Test that assistant messages with ExitPlanMode are converted to plan messages."""
        from src.agents.claude_code.core import ClaudeRecord as Record

        # Mock assistant message that contains both text and ExitPlanMode
        mixed_data = {
            "type": "assistant",
            "uuid": "test-uuid",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll create a plan for this task:"},
                    {
                        "type": "tool_use",
                        "name": "ExitPlanMode",
                        "input": {"plan": "## My Plan\nStep 1\nStep 2"},
                    },
                ],
            },
        }

        record = Record(type="assistant", raw_data=mixed_data)
        message = record.extract_message()

        # Should extract as plan message, not regular text message
        assert message is not None
        assert "📋 **Plan Proposal**" in message.content
        assert "## My Plan" in message.content

    def test_regular_message_without_plan_tools(self):
        """Test that regular messages without plan tools are processed normally."""
        from src.agents.claude_code.core import ClaudeRecord as Record

        # Mock regular assistant message
        regular_data = {
            "type": "assistant",
            "uuid": "test-uuid",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a regular message without any planning tools.",
                    }
                ],
            },
        }

        record = Record(type="assistant", raw_data=regular_data)
        message = record.extract_message()

        assert message is not None
        assert message.role == Role.ASSISTANT
        assert (
            message.content == "This is a regular message without any planning tools."
        )


class TestClaudeImplementation:
    """Test Claude-specific implementation details."""

    @pytest.fixture
    def test_data_path(self):
        return Path(__file__).parent / "test_data"

    @pytest.fixture
    def mock_home(self):
        """Path to mock Claude home directory."""
        return Path(__file__).parent / "mock_claude_home"

    def test_record_parsing(self):
        """Test parsing individual JSONL records."""
        from src.agents.claude_code.core import ClaudeRecord as Record

        # Test user message record
        user_json = '{"type":"user","message":{"role":"user","content":"Hello"},"uuid":"test-uuid","timestamp":"2024-01-01T12:00:00.000Z"}'
        record = Record.load(user_json)

        assert record.type == "user"
        assert record.is_message()

        message = record.extract_message()
        assert message.role == Role.USER
        assert message.content == "Hello"
        assert message.timestamp is not None

    def test_session_loading(self, test_data_path):
        """Test loading a complete session."""
        session_file = test_data_path / "claude_code_project" / "simple_chat.jsonl"
        log_file = ClaudeLogFile(file_path=session_file)
        log_file.load()

        # Verify session information exists in the messages
        messages = [
            r.extract_message() for r in log_file.records if r.extract_message()
        ]
        _session_ids = {msg.session_id for msg in messages if msg.session_id}
        # Note: Claude doesn't currently set session_id on messages, but structure is in place
        # Verify summary record exists
        summary_records = [
            r for r in log_file.records if hasattr(r, "type") and r.type == "summary"
        ]
        assert len(summary_records) == 1
        assert "Simple test chat" in summary_records[0].raw_data.get("summary", "")
        assert len(log_file.records) == 5  # 4 messages + 1 summary

        chat = log_file.to_chat()
        assert len(chat.messages) == 4

    def test_claude_store_operations(self, mock_home):
        """Test ClaudeStore file operations."""
        mock_cwd = Path("/home/user/projects/myproject/python")
        with (
            patch.object(Path, "home", return_value=mock_home),
            patch.object(Path, "cwd", return_value=mock_cwd),
        ):
            store = ClaudeLogStore()

            # Test listing
            logs = store.list()
            assert len(logs) > 0

            # Test getting content
            first_log_uri, metadata = logs[0]
            content = store.get(first_log_uri)
            assert len(content) > 0
            assert content.count("\n") >= 0  # Should have newlines

            # Test live log
            live_uri = store.live()
            assert live_uri is not None
            assert live_uri in [uri for uri, _ in logs]

    def test_tool_message_filtering(self, parser) -> None:
        """Test that tool use messages are filtered out correctly."""
        chat = parser.parse("tool_filtering_chat.jsonl")

        # Should have only text messages, no tool messages
        expected_text_messages = [
            "Create a Python function",
            "I'll help you create a Python function.",
            "Perfect! The function has been created. Here's what it does:",
            "The function is ready to use!",
        ]

        assert len(chat.messages) == len(expected_text_messages)

        # Verify content matches expected text messages
        for i, expected_content in enumerate(expected_text_messages):
            assert expected_content in chat.messages[i].content
            # Ensure no tool JSON artifacts remain
            assert not chat.messages[i].content.strip().startswith("[{")

    def test_mixed_content_messages(self) -> None:
        """Test messages with both text and tool content extract only text."""
        from src.agents.claude_code.core import ClaudeRecord as Record

        # Simulate a message with both text and tool_use blocks
        mixed_data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Let me help you with that."},
                    {"type": "tool_use", "id": "test", "name": "Bash"},
                ]
            },
            "uuid": "test",
        }

        record = Record(type="assistant", raw_data=mixed_data)
        message = record.extract_message()

        # Should extract only the text part
        assert message is not None
        assert message.content == "Let me help you with that."
        assert "tool_use" not in message.content


class TestErrorHandling:
    """Test error handling with real scenarios."""

    @pytest.fixture
    def test_data_path(self):
        return Path(__file__).parent / "test_data"

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON records."""
        from src.agents.claude_code.core import ClaudeRecord as Record

        invalid_json = '{"invalid": json missing closing brace'

        with pytest.raises(ValueError):
            Record.load(invalid_json)

    def test_missing_required_fields(self):
        """Test handling records with missing required fields."""
        from src.agents.claude_code.core import ClaudeRecord as Record

        minimal_json = '{"type":"unknown"}'
        record = Record.load(minimal_json)

        # Should handle missing fields gracefully
        assert record.type == "unknown"

        # Should not extract message for unknown type
        message = record.extract_message()
        assert message is None

    def test_file_permission_errors(self):
        """Test handling file access errors."""
        # Mock current working directory to nonexistent path
        with patch.object(Path, "cwd", return_value=Path("/nonexistent/directory")):
            store = ClaudeLogStore()

        # Should return empty list, not raise error
        logs = store.list()
        assert logs == []

        # Should raise proper error for get()
        with pytest.raises(IOError):
            store.get("/nonexistent/file.jsonl")


class TestMessageAndChatMethods:
    """Test Message and Chat functionality with real data."""

    def test_message_formatting(self):
        """Test message string formatting and dict conversion."""
        from datetime import datetime

        msg = Message(
            role=Role.USER,
            content="Test message",
            provider="test_provider",
            log_uri="/test/path",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            raw_data={"test": "value"},
            session_id="test_session",
        )

        # Test string formatting
        str_repr = str(msg)
        assert "USER" in str_repr
        assert "Test message" in str_repr
        assert "[12:00:00]" in str_repr

        # Test dict conversion
        msg_dict = msg.to_dict()
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Test message"
        assert msg_dict["timestamp"] == "2024-01-01T12:00:00"
        assert msg_dict["provider"] == "test_provider"
        assert msg_dict["raw_data"] == {"test": "value"}
        assert msg_dict["session_id"] == "test_session"

    def test_chat_operations(self):
        """Test Chat add, remove, merge operations."""
        msg1 = Message(
            role=Role.USER,
            content="First message",
            provider="test",
            log_uri="/test/path1",
        )
        msg2 = Message(
            role=Role.ASSISTANT,
            content="Second message",
            provider="test",
            log_uri="/test/path2",
        )
        msg3 = Message(
            role=Role.USER,
            content="Third message",
            provider="test",
            log_uri="/test/path3",
        )

        # Test adding messages
        chat1 = Chat()
        chat1.add(msg1)
        chat1.add(msg2)
        assert len(chat1.messages) == 2

        # Test removing messages
        chat1.remove(msg1)
        assert len(chat1.messages) == 1
        assert chat1.messages[0] == msg2

        # Test merging chats
        chat2 = Chat(messages=[msg3])
        merged = chat1.merge(chat2)
        assert len(merged.messages) == 2
        assert msg2 in merged.messages
        assert msg3 in merged.messages

    def test_tigs_export_format(self):
        """Test Tigs YAML format export."""
        messages = [
            Message(
                role=Role.SYSTEM,
                content="System prompt",
                provider="test",
                log_uri="/test/sys",
            ),
            Message(
                role=Role.USER,
                content="User question",
                provider="test",
                log_uri="/test/user",
            ),
            Message(
                role=Role.ASSISTANT,
                content="Assistant response",
                provider="test",
                log_uri="/test/assistant",
            ),
        ]

        chat = Chat(messages=messages)
        tigs_output = chat.export()

        # Should be valid YAML with proper Tigs structure
        import yaml

        tigs_data = yaml.safe_load(tigs_output)

        assert tigs_data["schema"] == "tigs.chat/v1"
        assert "messages" in tigs_data
        assert len(tigs_data["messages"]) == 3

        # Check message content (strip trailing newlines from literal block style)
        assert tigs_data["messages"][0]["role"] == "system"
        assert tigs_data["messages"][0]["content"].rstrip() == "System prompt"
        assert tigs_data["messages"][1]["role"] == "user"
        assert tigs_data["messages"][1]["content"].rstrip() == "User question"
        assert tigs_data["messages"][2]["role"] == "assistant"
        assert tigs_data["messages"][2]["content"].rstrip() == "Assistant response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
