"""Real data tests for the chat parser library."""

import pytest
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chat_parser import ChatParser, Chat, Message, Role
from chat_parser.claude.claude_code import ClaudeStore, Session, Record
from chat_parser.errors import ChatParserError


class TestChatParserReal:
    """Test ChatParser with real log data."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"
    
    @pytest.fixture 
    def parser(self, test_data_path):
        """ChatParser instance using test data."""
        return ChatParser("claude-code", location=str(test_data_path))
    
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
    
    def test_parse_simple_chat(self, parser, test_data_path):
        """Test parsing a simple chat log."""
        simple_log = test_data_path / "test_project" / "simple_chat.jsonl"
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
        system_log = test_data_path / "test_project" / "system_chat.jsonl" 
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
        empty_log = test_data_path / "test_project" / "empty_chat.jsonl"
        chat = parser.parse(str(empty_log))
        
        assert isinstance(chat, Chat)
        assert len(chat.messages) == 0
    
    def test_parse_malformed_chat(self, parser, test_data_path):
        """Test parsing a chat with malformed records."""
        malformed_log = test_data_path / "test_project" / "malformed_chat.jsonl"
        chat = parser.parse(str(malformed_log))
        
        # Should skip malformed records but parse valid ones
        assert len(chat.messages) == 2  # Two valid messages
        assert chat.messages[0].content == "This is a valid message"
        assert chat.messages[1].content == "This message follows the valid one"
    
    def test_parse_nonexistent_file(self, parser):
        """Test parsing a non-existent file."""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.jsonl")
    
    def test_live_log(self, parser):
        """Test getting the live (most recent) log."""
        chat = parser.parse()  # No URI = live log
        
        # Should get the most recent log (recent_chat.jsonl based on timestamp)
        assert isinstance(chat, Chat)
        assert len(chat.messages) >= 1
    
    def test_message_selection_and_composition(self, parser, test_data_path):
        """Test selecting messages and composing output."""
        simple_log = test_data_path / "test_project" / "simple_chat.jsonl"
        
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
        simple_log = str(test_data_path / "test_project" / "simple_chat.jsonl")
        
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
        simple_log = str(test_data_path / "test_project" / "simple_chat.jsonl")
        system_log = str(test_data_path / "test_project" / "system_chat.jsonl")
        
        # Select from multiple logs
        parser.select(simple_log, [0])  # First message from simple
        parser.select(system_log, [1])  # Second message from system
        
        # Should have messages from both logs
        assert len(parser.selected_messages) == 2
        
        result = parser.compose()
        assert "create a list in Python" in result  # From simple_log
        assert "hello world program" in result      # From system_log


class TestClaudeImplementation:
    """Test Claude-specific implementation details."""
    
    @pytest.fixture
    def test_data_path(self):
        return Path(__file__).parent / "test_data"
    
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
        session_file = test_data_path / "test_project" / "simple_chat.jsonl"
        session = Session(file_path=session_file)
        session.load()
        
        assert session.session_id == "test-session-1"
        assert "Simple test chat" in session.summary
        assert len(session.records) == 5  # 4 messages + 1 summary
        
        chat = session.to_chat()
        assert len(chat.messages) == 4
    
    def test_claude_store_operations(self, test_data_path):
        """Test ClaudeStore file operations."""
        store = ClaudeStore(location=str(test_data_path))
        
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
        store = ClaudeStore(location="/nonexistent/directory")
        
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
        
        # Check message content
        assert tigs_data["messages"][0]["role"] == "system"
        assert tigs_data["messages"][0]["content"] == "System prompt"
        assert tigs_data["messages"][1]["role"] == "user"
        assert tigs_data["messages"][1]["content"] == "User question"
        assert tigs_data["messages"][2]["role"] == "assistant"
        assert tigs_data["messages"][2]["content"] == "Assistant response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])