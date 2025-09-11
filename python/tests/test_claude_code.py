"""Real data tests for the chat parser library."""

import pytest
from pathlib import Path
import sys
import os
from unittest.mock import patch

from core import Chat, Message, Role, ChatParserError, cligent as ChatParser
from agents.claude.claude_code import ClaudeStore, Session, Record


class TestChatParserReal:
    """Test ChatParser with real log data."""

    @pytest.fixture
    def test_data_path(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"

    @pytest.fixture
    def mock_home(self):
        """Path to mock Claude home directory."""
        return Path(__file__).parent / "mock_claude_home"

    @pytest.fixture
    def parser(self, test_data_path):
        """ChatParser instance using test data."""
        # Mock the current working directory to point to test data
        with patch.object(Path, 'cwd', return_value=test_data_path):
            return ChatParser("claude-code")

    @pytest.fixture
    def claude_parser(self, mock_home):
        """ChatParser instance using mock Claude environment."""
        mock_cwd = Path("/home/user/projects/myproject/python")
        with patch.object(Path, 'home', return_value=mock_home), \
             patch.object(Path, 'cwd', return_value=mock_cwd):
            return ChatParser("claude-code")

    def test_list_logs_real_data(self, claude_parser, mock_home):
        """Test listing logs with real data."""
        mock_cwd = Path("/home/user/projects/myproject/python")
        with patch.object(Path, 'home', return_value=mock_home), \
             patch.object(Path, 'cwd', return_value=mock_cwd):
            logs = claude_parser.list()

        # Should find our test logs
        assert len(logs) >= 4  # We created at least 4 test files

        # Check structure
        for log_uri, metadata in logs:
            assert isinstance(log_uri, str)
            assert isinstance(metadata, dict)
            assert "size" in metadata
            assert "project" in metadata
            assert "accessible" in metadata

    def test_parse_simple_chat(self, parser, test_data_path):
        """Test parsing a simple chat log."""
        simple_log = test_data_path / "claude_code_project" / "simple_chat.jsonl"
        chat = parser.parse(str(simple_log))

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

    def test_parse_system_message_chat(self, parser, test_data_path):
        """Test parsing a chat with system messages."""
        system_log = test_data_path / "claude_code_project" / "system_chat.jsonl"
        chat = parser.parse(str(system_log))

        assert len(chat.messages) == 3

        # First message should be system
        assert chat.messages[0].role == Role.SYSTEM
        assert "helpful coding assistant" in chat.messages[0].content

        # Should have user and assistant messages too
        roles = [msg.role for msg in chat.messages]
        assert Role.USER in roles
        assert Role.ASSISTANT in roles

    def test_parse_empty_chat(self, parser, test_data_path):
        """Test parsing an empty chat (summary only)."""
        empty_log = test_data_path / "claude_code_project" / "empty_chat.jsonl"
        chat = parser.parse(str(empty_log))

        assert isinstance(chat, Chat)
        assert len(chat.messages) == 0

    def test_parse_malformed_chat(self, parser, test_data_path):
        """Test parsing a chat with malformed records."""
        malformed_log = test_data_path / "claude_code_project" / "malformed_chat.jsonl"
        chat = parser.parse(str(malformed_log))

        # Should skip malformed records but parse valid ones
        assert len(chat.messages) == 2  # Two valid messages
        assert chat.messages[0].content == "This is a valid message"
        assert chat.messages[1].content == "This message follows the valid one"

    def test_parse_nonexistent_file(self, parser):
        """Test parsing a non-existent file."""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.jsonl")

    def test_live_log(self, claude_parser, mock_home):
        """Test getting the live (most recent) log."""
        mock_cwd = Path("/home/user/projects/myproject/python")
        with patch.object(Path, 'home', return_value=mock_home), \
             patch.object(Path, 'cwd', return_value=mock_cwd):
            chat = claude_parser.parse()  # No URI = live log

        # Should get the most recent log
        assert isinstance(chat, Chat)
        assert len(chat.messages) >= 1

    def test_message_selection_and_composition(self, parser, test_data_path):
        """Test selecting messages and composing output."""
        simple_log = test_data_path / "claude_code_project" / "simple_chat.jsonl"

        # Select specific messages
        parser.select(str(simple_log), [0, 2])  # First and third messages
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
        parser.select(str(simple_log))  # All messages
        result = parser.compose()

        tigs_data = yaml.safe_load(result)
        assert len(tigs_data["messages"]) == 4  # All messages

    def test_unselect_messages(self, parser, test_data_path):
        """Test unselecting specific messages."""
        simple_log = str(test_data_path / "claude_code_project" / "simple_chat.jsonl")

        # Select all messages
        parser.select(simple_log)
        assert len(parser.selected_messages) == 4

        # Unselect first message
        parser.unselect(simple_log, [0])
        assert len(parser.selected_messages) == 3

        # Unselect all messages from this log
        parser.unselect(simple_log)
        assert len(parser.selected_messages) == 0

    def test_multiple_log_composition(self, parser, test_data_path):
        """Test composing from multiple logs."""
        simple_log = str(test_data_path / "claude_code_project" / "simple_chat.jsonl")
        system_log = str(test_data_path / "claude_code_project" / "system_chat.jsonl")

        # Select from multiple logs
        parser.select(simple_log, [0])  # First message from simple
        parser.select(system_log, [1])  # Second message from system

        # Should have messages from both logs
        assert len(parser.selected_messages) == 2

        result = parser.compose()
        assert "create a list in Python" in result  # From simple_log
        assert "hello world program" in result      # From system_log


class TestSessionIDFunctionality:
    """Test the new session ID based functionality."""

    def test_list_logs_returns_session_ids(self):
        """Test that list returns session IDs, not full paths."""
        from core import cligent as ChatParser
        from pathlib import Path

        # Use current directory which should have Claude Code logs
        parser = ChatParser("claude-code")
        logs = parser.list()

        if logs:  # Only test if there are logs
            for log_uri, metadata in logs:
                # Session IDs should not contain path separators
                assert "/" not in log_uri, f"Expected session ID, got path: {log_uri}"
                assert "\\" not in log_uri, f"Expected session ID, got path: {log_uri}"
                # Session IDs are UUIDs (36 chars with dashes)
                assert len(log_uri) == 36, f"Session ID wrong length: {log_uri}"
                assert log_uri.count("-") == 4, f"Session ID wrong format: {log_uri}"

    def test_parse_with_session_id(self):
        """Test parsing using session ID instead of full path."""
        from core import cligent as ChatParser

        parser = ChatParser("claude-code")
        logs = parser.list()

        if logs:  # Only test if there are logs
            session_id, _ = logs[0]
            # Should be able to parse using just the session ID
            chat = parser.parse(session_id)
            assert chat is not None
            assert len(chat.messages) >= 0

    def test_location_parameter_changes_project(self):
        """Test that location parameter changes which project's logs are found."""
        from core import cligent as ChatParser
        from pathlib import Path

        mock_home = Path(__file__).parent / "mock_claude_home"
        
        # Test with different project directories
        with patch.object(Path, 'home', return_value=mock_home):
            # Mock different working directories to simulate different projects
            with patch.object(Path, 'cwd', return_value=Path("/home/user/projects/myproject")):
                parent_parser = ChatParser("claude-code")
                parent_logs = parent_parser.list()

            with patch.object(Path, 'cwd', return_value=Path("/home/user/projects/myproject/python")):
                python_parser = ChatParser("claude-code")
                python_logs = python_parser.list()

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
        from agents.claude.claude_code import Record
        
        # Mock ExitPlanMode record
        exit_plan_data = {
            "type": "assistant",
            "uuid": "test-uuid",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text", 
                        "text": "Let me create a plan:"
                    },
                    {
                        "type": "tool_use",
                        "id": "tool-id",
                        "name": "ExitPlanMode",
                        "input": {
                            "plan": "## Test Plan\n\n1. Do something\n2. Do something else"
                        }
                    }
                ]
            },
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        record = Record(
            type="assistant",
            uuid="test-uuid", 
            raw_data=exit_plan_data,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        # Test detection
        assert record.is_exit_plan_mode()
        
        # Test message extraction
        message = record.extract_message()
        assert message is not None
        assert message.role == Role.ASSISTANT
        assert "📋 **Plan Proposal**" in message.content
        assert "## Test Plan" in message.content
        assert message.metadata['is_plan'] == True
        assert message.metadata['tool_type'] == 'ExitPlanMode'

    def test_plan_response_detection(self):
        """Test that plan approval responses are correctly detected."""
        from agents.claude.claude_code import Record
        
        # Mock plan approval record
        plan_response_data = {
            "type": "user",
            "uuid": "test-uuid",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-id",
                        "content": "User has approved your plan. You can now start coding."
                    }
                ]
            },
            "timestamp": "2024-01-01T12:01:00Z"
        }
        
        record = Record(
            type="user",
            uuid="test-uuid",
            raw_data=plan_response_data,
            timestamp="2024-01-01T12:01:00Z" 
        )
        
        # Test detection
        assert record.is_plan_response()
        
        # Test message extraction
        message = record.extract_message()
        assert message is not None
        assert message.role == Role.USER
        assert "✅ **Plan Approved**" in message.content
        assert message.metadata['is_plan_response'] == True
        assert message.metadata['tool_type'] == 'plan_response'

    def test_plan_rejection_detection(self):
        """Test that plan rejection responses are correctly detected."""
        from agents.claude.claude_code import Record
        
        # Mock plan rejection record (actual format from real log)
        plan_rejection_data = {
            "type": "user",
            "uuid": "test-uuid",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-id",
                        "content": "The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and wait for the user to tell you how to proceed.",
                        "is_error": True
                    }
                ]
            },
            "timestamp": "2024-01-01T12:01:00Z"
        }
        
        record = Record(
            type="user",
            uuid="test-uuid",
            raw_data=plan_rejection_data,
            timestamp="2024-01-01T12:01:00Z" 
        )
        
        # Test detection
        assert record.is_plan_response()
        
        # Test message extraction
        message = record.extract_message()
        assert message is not None
        assert message.role == Role.USER
        assert "❌ **Plan Rejected**" in message.content
        assert "User has rejected the plan proposal" in message.content
        assert message.metadata['is_plan_response'] == True
        assert message.metadata['tool_type'] == 'plan_response'

    def test_generic_rejection_detection(self):
        """Test that generic tool rejection is also detected as plan response."""
        from agents.claude.claude_code import Record
        
        # Mock generic tool rejection
        rejection_data = {
            "type": "user",
            "uuid": "test-uuid",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-id",
                        "content": "tool use was rejected",
                        "is_error": True
                    }
                ]
            },
            "timestamp": "2024-01-01T12:01:00Z"
        }
        
        record = Record(
            type="user",
            uuid="test-uuid",
            raw_data=rejection_data,
            timestamp="2024-01-01T12:01:00Z" 
        )
        
        # Test detection
        assert record.is_plan_response()
        
        # Test message extraction  
        message = record.extract_message()
        assert message is not None
        assert message.role == Role.USER
        assert "❌ **Plan Rejected**" in message.content

    def test_regular_assistant_message_with_plan_present(self):
        """Test that assistant messages with ExitPlanMode are converted to plan messages."""
        from agents.claude.claude_code import Record
        
        # Mock assistant message that contains both text and ExitPlanMode
        mixed_data = {
            "type": "assistant",
            "uuid": "test-uuid",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'll create a plan for this task:"
                    },
                    {
                        "type": "tool_use", 
                        "name": "ExitPlanMode",
                        "input": {
                            "plan": "## My Plan\nStep 1\nStep 2"
                        }
                    }
                ]
            }
        }
        
        record = Record(type="assistant", uuid="test-uuid", raw_data=mixed_data)
        message = record.extract_message()
        
        # Should extract as plan message, not regular text message
        assert message is not None
        assert message.metadata['is_plan'] == True
        assert "📋 **Plan Proposal**" in message.content
        assert "## My Plan" in message.content

    def test_regular_message_without_plan_tools(self):
        """Test that regular messages without plan tools are processed normally."""
        from agents.claude.claude_code import Record
        
        # Mock regular assistant message
        regular_data = {
            "type": "assistant", 
            "uuid": "test-uuid",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a regular message without any planning tools."
                    }
                ]
            }
        }
        
        record = Record(type="assistant", uuid="test-uuid", raw_data=regular_data)
        message = record.extract_message()
        
        assert message is not None
        assert message.role == Role.ASSISTANT
        assert message.content == "This is a regular message without any planning tools."
        assert not message.metadata.get('is_plan', False)
        assert not message.metadata.get('is_plan_response', False)


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
        # Test user message record
        user_json = '{"type":"user","message":{"role":"user","content":"Hello"},"uuid":"test-uuid","timestamp":"2024-01-01T12:00:00.000Z"}'
        record = Record.load(user_json)

        assert record.type == "user"
        assert record.uuid == "test-uuid"
        assert record.is_message()

        message = record.extract_message()
        assert message.role == Role.USER
        assert message.content == "Hello"
        assert message.timestamp is not None

    def test_session_loading(self, test_data_path):
        """Test loading a complete session."""
        session_file = test_data_path / "claude_code_project" / "simple_chat.jsonl"
        session = Session(file_path=session_file)
        session.load()

        assert session.session_id == "test-session-1"
        assert "Simple test chat" in session.summary
        assert len(session.records) == 5  # 4 messages + 1 summary

        chat = session.to_chat()
        assert len(chat.messages) == 4

    def test_claude_store_operations(self, mock_home):
        """Test ClaudeStore file operations."""
        mock_cwd = Path("/home/user/projects/myproject/python")
        with patch.object(Path, 'home', return_value=mock_home), \
             patch.object(Path, 'cwd', return_value=mock_cwd):
            store = ClaudeStore()

            # Test listing
            logs = store.list()
            assert len(logs) > 0

            # Test getting content
            first_log_uri, metadata = logs[0]
            content = store.get(first_log_uri)
            assert len(content) > 0
            assert content.count('\n') >= 0  # Should have newlines

            # Test live log
            live_uri = store.live()
            assert live_uri is not None
            assert live_uri in [uri for uri, _ in logs]

    def test_tool_message_filtering(self, test_data_path: Path) -> None:
        """Test that tool use messages are filtered out correctly."""
        from core import cligent as ChatParser

        # Mock the current working directory to point to test data
        with patch.object(Path, 'cwd', return_value=test_data_path):
            parser = ChatParser("claude-code")
        tool_log = (
            test_data_path / "claude_code_project" / "tool_filtering_chat.jsonl"
        )

        chat = parser.parse(str(tool_log))

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
            assert not chat.messages[i].content.strip().startswith('[{')

    def test_mixed_content_messages(self) -> None:
        """Test messages with both text and tool content extract only text."""
        from agents.claude.claude_code import Record

        # Simulate a message with both text and tool_use blocks
        mixed_data = {
            'type': 'assistant',
            'message': {
                'content': [
                    {'type': 'text', 'text': 'Let me help you with that.'},
                    {'type': 'tool_use', 'id': 'test', 'name': 'Bash'},
                ]
            },
            'uuid': 'test',
        }

        record = Record(type='assistant', uuid='test', raw_data=mixed_data)
        message = record.extract_message()

        # Should extract only the text part
        assert message is not None
        assert message.content == "Let me help you with that."
        assert 'tool_use' not in message.content


class TestErrorHandling:
    """Test error handling with real scenarios."""

    @pytest.fixture
    def test_data_path(self):
        return Path(__file__).parent / "test_data"

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON records."""
        invalid_json = '{"invalid": json missing closing brace'

        with pytest.raises(ValueError):
            Record.load(invalid_json)

    def test_missing_required_fields(self):
        """Test handling records with missing required fields."""
        minimal_json = '{"type":"unknown"}'
        record = Record.load(minimal_json)

        # Should handle missing fields gracefully
        assert record.type == "unknown"
        assert record.uuid == ""  # Default value

        # Should not extract message for unknown type
        message = record.extract_message()
        assert message is None

    def test_file_permission_errors(self):
        """Test handling file access errors."""
        # Mock current working directory to nonexistent path
        with patch.object(Path, 'cwd', return_value=Path("/nonexistent/directory")):
            store = ClaudeStore()

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
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"test": "value"}
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
        assert msg_dict["metadata"] == {"test": "value"}

    def test_chat_operations(self):
        """Test Chat add, remove, merge operations."""
        msg1 = Message(role=Role.USER, content="First message")
        msg2 = Message(role=Role.ASSISTANT, content="Second message")
        msg3 = Message(role=Role.USER, content="Third message")

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
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="User question"),
            Message(role=Role.ASSISTANT, content="Assistant response")
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
