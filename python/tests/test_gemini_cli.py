"""Unit tests for Gemini CLI code agent implementation."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

from chat_parser.gemini.gemini_cli import (
    GeminiRecord, 
    GeminiSession, 
    GeminiStore, 
    GeminiCliAgent
)
from chat_parser.models import Role, Chat


class TestGeminiRecord:
    """Test GeminiRecord class functionality."""

    def test_load_basic_record(self):
        """Test loading a basic Gemini record."""
        json_data = {
            "type": "user",
            "role": "user",
            "content": "Hello, Gemini!",
            "timestamp": "2024-01-01T10:00:00Z",
            "session_id": "test-session-123"
        }
        json_string = json.dumps(json_data)
        
        record = GeminiRecord.load(json_string)
        
        assert record.type == "user"
        assert record.role == "user"
        assert record.content == "Hello, Gemini!"
        assert record.timestamp == "2024-01-01T10:00:00Z"
        assert record.session_id == "test-session-123"
        assert record.raw_data == json_data

    def test_load_with_alternative_fields(self):
        """Test loading record with alternative field names."""
        json_data = {
            "type": "assistant",
            "sender": "model",
            "text": "Hello, human!",
            "time": "1704103200",
            "conversation_id": "conv-456"
        }
        json_string = json.dumps(json_data)
        
        record = GeminiRecord.load(json_string)
        
        assert record.type == "assistant"
        assert record.role == "model"
        assert record.content == "Hello, human!"
        assert record.timestamp == "1704103200"
        assert record.session_id == "conv-456"

    def test_load_with_message_field(self):
        """Test loading record with 'message' field for content."""
        json_data = {
            "type": "system",
            "role": "system",
            "message": "System initialized",
            "created_at": "2024-01-01T12:00:00+00:00"
        }
        json_string = json.dumps(json_data)
        
        record = GeminiRecord.load(json_string)
        
        assert record.content == "System initialized"
        assert record.timestamp == "2024-01-01T12:00:00+00:00"

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON record"):
            GeminiRecord.load("invalid json")

    def test_extract_message_user_role(self):
        """Test extracting message from user record."""
        record = GeminiRecord(
            type="user",
            role="user",
            content="Test message",
            timestamp="2024-01-01T10:00:00Z"
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert message.role == Role.USER
        assert message.content == "Test message"
        assert message.timestamp == datetime.fromisoformat("2024-01-01T10:00:00+00:00")

    def test_extract_message_model_role(self):
        """Test extracting message with 'model' role mapped to assistant."""
        record = GeminiRecord(
            type="assistant",
            role="model",
            content="Assistant response",
            timestamp="1704103200"  # Unix timestamp
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert message.role == Role.ASSISTANT
        assert message.content == "Assistant response"
        assert message.timestamp == datetime.fromtimestamp(1704103200)

    def test_extract_message_list_content(self):
        """Test extracting message with list content."""
        record = GeminiRecord(
            type="assistant",
            role="assistant",
            content=[
                {"text": "First part"},
                "Second part",
                {"content": "Third part"}
            ]
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert message.content == "First part\nSecond part\nThird part"

    def test_extract_message_dict_content(self):
        """Test extracting message with dict content."""
        record = GeminiRecord(
            type="user",
            role="user",
            content={"text": "Message from dict"}
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert message.content == "Message from dict"

    def test_extract_message_empty_content(self):
        """Test extracting message with empty content returns None."""
        record = GeminiRecord(
            type="user",
            role="user",
            content=""
        )
        
        message = record.extract_message()
        assert message is None

    def test_extract_message_invalid_timestamp(self):
        """Test extracting message with invalid timestamp."""
        record = GeminiRecord(
            type="user",
            role="user",
            content="Test",
            timestamp="invalid-timestamp"
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert message.timestamp is None

    def test_is_message_by_type(self):
        """Test is_message detection by record type."""
        message_types = ['user', 'assistant', 'system', 'human', 'ai', 'model', 'message']
        
        for msg_type in message_types:
            record = GeminiRecord(type=msg_type)
            assert record.is_message()
        
        record = GeminiRecord(type="tool_use")
        assert not record.is_message()

    def test_is_message_by_role(self):
        """Test is_message detection by role field."""
        message_roles = ['user', 'assistant', 'system', 'human', 'ai', 'model']
        
        for role in message_roles:
            record = GeminiRecord(type="unknown", role=role)
            assert record.is_message()
        
        record = GeminiRecord(type="unknown", role="tool")
        assert not record.is_message()

    def test_role_mapping_variations(self):
        """Test various role mapping scenarios."""
        role_tests = [
            ("user", Role.USER),
            ("human", Role.USER),
            ("assistant", Role.ASSISTANT),
            ("model", Role.ASSISTANT),
            ("ai", Role.ASSISTANT),
            ("system", Role.SYSTEM),
            ("unknown", Role.ASSISTANT),  # Default fallback
        ]
        
        for gemini_role, expected_role in role_tests:
            record = GeminiRecord(type="message", role=gemini_role, content="test")
            message = record.extract_message()
            assert message.role == expected_role


class TestGeminiSession:
    """Test GeminiSession class functionality."""

    @pytest.fixture
    def temp_jsonl_file(self, tmp_path):
        """Create a temporary JSONL file for testing."""
        file_path = tmp_path / "test_session.jsonl"
        
        records = [
            {"type": "user", "role": "user", "content": "Hello", "session_id": "test-123"},
            {"type": "assistant", "role": "model", "content": "Hi there!"},
            {"type": "user", "role": "user", "content": "How are you?"},
            {"type": "assistant", "role": "model", "content": "I'm doing well!"}
        ]
        
        with open(file_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        return file_path

    @pytest.fixture
    def malformed_jsonl_file(self, tmp_path):
        """Create a JSONL file with some malformed lines."""
        file_path = tmp_path / "malformed_session.jsonl"
        
        with open(file_path, 'w') as f:
            f.write('{"type": "user", "content": "Good line"}\n')
            f.write('invalid json line\n')
            f.write('{"type": "assistant", "content": "Another good line"}\n')
        
        return file_path

    def test_load_valid_file(self, temp_jsonl_file):
        """Test loading a valid JSONL file."""
        session = GeminiSession(file_path=temp_jsonl_file)
        session.load()
        
        assert len(session.records) == 4
        assert session.session_id == "test-123"
        assert session.records[0].content == "Hello"
        assert session.records[1].role == "model"

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        session = GeminiSession(file_path=Path("/nonexistent/file.jsonl"))
        
        with pytest.raises(FileNotFoundError, match="Log file not found"):
            session.load()

    def test_load_malformed_file(self, malformed_jsonl_file, capsys):
        """Test loading file with malformed JSON lines."""
        session = GeminiSession(file_path=malformed_jsonl_file)
        session.load()
        
        # Should have 2 valid records despite malformed line
        assert len(session.records) == 2
        
        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning: Skipped invalid record" in captured.out

    def test_to_chat(self, temp_jsonl_file):
        """Test converting session to Chat object."""
        session = GeminiSession(file_path=temp_jsonl_file)
        session.load()
        
        chat = session.to_chat()
        
        assert isinstance(chat, Chat)
        assert len(chat.messages) == 4
        assert chat.messages[0].role == Role.USER
        assert chat.messages[1].role == Role.ASSISTANT
        assert chat.messages[0].content == "Hello"

    def test_empty_file(self, tmp_path):
        """Test loading empty file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.touch()
        
        session = GeminiSession(file_path=empty_file)
        session.load()
        
        assert len(session.records) == 0
        chat = session.to_chat()
        assert len(chat.messages) == 0


class TestGeminiStore:
    """Test GeminiStore class functionality."""

    @pytest.fixture
    def mock_home_dir(self, tmp_path):
        """Create mock home directory structure."""
        home_dir = tmp_path / "home"
        gemini_dir = home_dir / ".gemini" / "tmp"
        gemini_dir.mkdir(parents=True)
        
        # Create some test log files
        (gemini_dir / "session1.jsonl").write_text('{"content": "test1"}')
        (gemini_dir / "session2.jsonl").write_text('{"content": "test2"}')
        
        return home_dir

    def test_init_with_default_location(self):
        """Test initializing store with default location."""
        store = GeminiStore()
        assert store.agent == "gemini-cli"

    def test_init_with_custom_location(self):
        """Test initializing store with custom location."""
        custom_path = Path("/custom/path")
        store = GeminiStore(location=custom_path)
        assert store.agent == "gemini-cli"

    def test_list_logs(self, mock_home_dir):
        """Test listing available logs."""
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = GeminiStore()
            logs = store.list()
        
        assert len(logs) == 2
        session_ids = [log[0] for log in logs]
        assert "session1" in session_ids
        assert "session2" in session_ids
        
        # Check metadata structure
        metadata = logs[0][1]
        assert "size" in metadata
        assert "modified" in metadata
        assert "accessible" in metadata

    def test_list_logs_no_directory(self, tmp_path):
        """Test listing logs when directory doesn't exist."""
        fake_home = tmp_path / "fake_home"
        with patch('pathlib.Path.home', return_value=fake_home):
            store = GeminiStore()
            logs = store.list()
        
        assert logs == []

    def test_get_log_by_session_id(self, mock_home_dir):
        """Test retrieving log content by session ID."""
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = GeminiStore()
            content = store.get("session1")
        
        assert content == '{"content": "test1"}'

    def test_get_log_by_full_path(self, mock_home_dir):
        """Test retrieving log content by full path."""
        log_path = mock_home_dir / ".gemini" / "tmp" / "session1.jsonl"
        
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = GeminiStore()
            content = store.get(str(log_path))
        
        assert content == '{"content": "test1"}'

    def test_get_nonexistent_log(self, mock_home_dir):
        """Test retrieving non-existent log raises FileNotFoundError."""
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = GeminiStore()
            
            with pytest.raises(FileNotFoundError, match="Log file not found"):
                store.get("nonexistent")

    def test_live_log(self, mock_home_dir):
        """Test getting most recent log."""
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = GeminiStore()
            live_log = store.live()
        
        # Should return one of the sessions (most recent)
        assert live_log in ["session1", "session2"]

    def test_live_log_no_logs(self, tmp_path):
        """Test getting live log when no logs exist."""
        fake_home = tmp_path / "fake_home"
        with patch('pathlib.Path.home', return_value=fake_home):
            store = GeminiStore()
            live_log = store.live()
        
        assert live_log is None

    def test_fallback_directories(self, tmp_path):
        """Test fallback to different log directories."""
        home_dir = tmp_path / "home"
        
        # Test fallback to logs directory when tmp doesn't exist
        logs_dir = home_dir / ".gemini" / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "test.jsonl").write_text('{"test": "data"}')
        
        with patch('pathlib.Path.home', return_value=home_dir):
            store = GeminiStore()
            assert store._logs_dir == logs_dir
            logs = store.list()
            assert len(logs) == 1

    def test_directory_selection_logic(self, tmp_path):
        """Test directory selection follows expected priority."""
        home_dir = tmp_path / "home"
        gemini_base = home_dir / ".gemini"
        
        # Test that when logs directory exists, it's selected over sessions
        logs_dir = gemini_base / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "test.jsonl").write_text('{"test": "data"}')
        
        with patch('pathlib.Path.home', return_value=home_dir):
            store = GeminiStore()
            assert store._logs_dir == logs_dir
            logs = store.list()
            assert len(logs) == 1


class TestGeminiCliAgent:
    """Test GeminiCliAgent class functionality."""

    def test_config_properties(self):
        """Test agent configuration properties."""
        agent = GeminiCliAgent()
        config = agent.config
        
        assert config.name == "gemini-cli"
        assert config.display_name == "Gemini CLI"
        assert config.log_extensions == [".jsonl", ".json"]
        assert config.requires_session_id is True
        assert config.metadata["log_format"] == "jsonl"
        assert config.metadata["base_dir"] == "~/.gemini/"
        assert config.metadata["supports_tools"] is True

    def test_create_store(self):
        """Test creating store instance."""
        agent = GeminiCliAgent()
        store = agent.create_store()
        
        assert isinstance(store, GeminiStore)

    def test_create_store_with_location(self):
        """Test creating store with custom location."""
        agent = GeminiCliAgent()
        store = agent.create_store(location="/custom/path")
        
        assert isinstance(store, GeminiStore)

    def test_parse_content_with_session_id(self, tmp_path):
        """Test parsing content using session ID."""
        # Create a temporary JSONL file
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_file = logs_dir / "test-session.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"type": "user", "role": "user", "content": "Hello"}\n')
            f.write('{"type": "assistant", "role": "model", "content": "Hi!"}\n')
        
        # Mock the store's _logs_dir
        agent = GeminiCliAgent()
        store = agent.create_store()
        store._logs_dir = logs_dir
        
        content = store.get("test-session")
        chat = agent.parse_content(content, "test-session", store)
        
        assert isinstance(chat, Chat)
        assert len(chat.messages) == 2
        assert chat.messages[0].role == Role.USER
        assert chat.messages[0].content == "Hello"

    def test_parse_content_with_full_path(self, tmp_path):
        """Test parsing content using full file path."""
        log_file = tmp_path / "full_path_test.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"type": "user", "role": "user", "content": "Test message"}\n')
        
        agent = GeminiCliAgent()
        store = agent.create_store()
        
        content = '{"type": "user", "role": "user", "content": "Test message"}\n'
        chat = agent.parse_content(content, str(log_file), store)
        
        assert isinstance(chat, Chat)
        assert len(chat.messages) == 1
        assert chat.messages[0].content == "Test message"

    def test_detect_agent_jsonl_file(self, tmp_path):
        """Test detecting Gemini CLI logs from JSONL file."""
        log_file = tmp_path / "gemini_log.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"role": "user", "content": "Hello Gemini"}\n')
            f.write('{"role": "model", "content": "Hi there!"}\n')
        
        agent = GeminiCliAgent()
        assert agent.detect_agent(log_file) is True

    def test_detect_agent_with_gemini_indicators(self, tmp_path):
        """Test detecting logs with Gemini-specific indicators."""
        log_file = tmp_path / "test.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"model": "gemini-pro", "role": "user", "text": "Test"}\n')
        
        agent = GeminiCliAgent()
        assert agent.detect_agent(log_file) is True

    def test_detect_agent_with_session_id(self, tmp_path):
        """Test detecting logs with session ID indicators."""
        log_file = tmp_path / "test.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"session_id": "123", "content": "Test"}\n')
        
        agent = GeminiCliAgent()
        assert agent.detect_agent(log_file) is True

    def test_detect_agent_wrong_extension(self, tmp_path):
        """Test detection fails for wrong file extension."""
        log_file = tmp_path / "test.txt"
        log_file.write_text('{"role": "user", "content": "Test"}')
        
        agent = GeminiCliAgent()
        assert agent.detect_agent(log_file) is False

    def test_detect_agent_non_gemini_content(self, tmp_path):
        """Test detection fails for non-Gemini content."""
        log_file = tmp_path / "other.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"tool": "other", "data": "Not Gemini"}\n')
        
        agent = GeminiCliAgent()
        assert agent.detect_agent(log_file) is False

    def test_detect_agent_malformed_json(self, tmp_path):
        """Test detection handles malformed JSON gracefully."""
        log_file = tmp_path / "malformed.jsonl"
        log_file.write_text('invalid json content')
        
        agent = GeminiCliAgent()
        assert agent.detect_agent(log_file) is False

    def test_detect_agent_empty_file(self, tmp_path):
        """Test detection handles empty files gracefully."""
        log_file = tmp_path / "empty.jsonl"
        log_file.touch()
        
        agent = GeminiCliAgent()
        assert agent.detect_agent(log_file) is False


class TestGeminiIntegration:
    """Integration tests for Gemini CLI components."""

    @pytest.fixture
    def test_data_path(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"

    @pytest.fixture
    def gemini_test_data_path(self, test_data_path):
        """Path to Gemini test data directory."""
        return test_data_path / "gemini_project"

    @pytest.fixture
    def mock_gemini_home(self):
        """Path to mock Gemini home directory."""
        return Path(__file__).parent / "mock_gemini_home"

    @pytest.fixture
    def gemini_parser(self, mock_gemini_home):
        """ChatParser instance using mock Gemini environment."""
        with patch('pathlib.Path.home', return_value=mock_gemini_home):
            from cligent import ChatParser
            return ChatParser("gemini-cli")

    @pytest.fixture
    def gemini_parser_with_test_data(self, gemini_test_data_path):
        """ChatParser instance using test data files."""
        import tempfile
        import shutil
        
        # Create temporary home directory structure
        temp_home = Path(tempfile.mkdtemp())
        gemini_dir = temp_home / ".gemini" / "tmp"
        gemini_dir.mkdir(parents=True)
        
        try:
            # Copy test data files to the temporary structure
            for test_file in gemini_test_data_path.glob("*.jsonl"):
                if test_file.name != "empty_gemini_chat.jsonl":  # Skip empty file
                    # Use test file name as session name
                    session_name = test_file.stem.replace("_gemini_chat", "")
                    dest_file = gemini_dir / f"{session_name}.jsonl"
                    shutil.copy2(test_file, dest_file)
            
            with patch('pathlib.Path.home', return_value=temp_home):
                from cligent import ChatParser
                yield ChatParser("gemini-cli")
        finally:
            # Cleanup
            shutil.rmtree(temp_home, ignore_errors=True)

    @pytest.fixture
    def sample_gemini_data(self, tmp_path):
        """Create sample Gemini CLI data structure."""
        home_dir = tmp_path / "home"
        gemini_dir = home_dir / ".gemini" / "tmp"
        gemini_dir.mkdir(parents=True)
        
        # Create a realistic Gemini conversation
        conversation = [
            {
                "type": "user",
                "role": "user",
                "content": "What's the weather like?",
                "timestamp": "2024-01-01T10:00:00Z",
                "session_id": "chat-session-123"
            },
            {
                "type": "assistant",
                "role": "model",
                "content": [
                    {"text": "I don't have access to real-time weather data, "},
                    {"text": "but I can help you find weather information."}
                ],
                "timestamp": "2024-01-01T10:00:05Z",
                "session_id": "chat-session-123"
            },
            {
                "type": "user",
                "role": "user",
                "content": "How can I check the weather?",
                "timestamp": "2024-01-01T10:01:00Z",
                "session_id": "chat-session-123"
            },
            {
                "type": "assistant",
                "role": "model",
                "content": {
                    "text": "You can check weather using weather apps or websites like Weather.com."
                },
                "timestamp": "2024-01-01T10:01:10Z",
                "session_id": "chat-session-123"
            }
        ]
        
        log_file = gemini_dir / "chat-session-123.jsonl"
        with open(log_file, 'w') as f:
            for record in conversation:
                f.write(json.dumps(record) + '\n')
        
        return home_dir, "chat-session-123"

    def test_end_to_end_parsing(self, sample_gemini_data):
        """Test complete end-to-end parsing workflow."""
        home_dir, session_id = sample_gemini_data
        
        with patch('pathlib.Path.home', return_value=home_dir):
            agent = GeminiCliAgent()
            store = agent.create_store()
            
            # List available logs
            logs = store.list()
            assert len(logs) == 1
            assert logs[0][0] == session_id
            
            # Get log content
            content = store.get(session_id)
            assert content is not None
            
            # Parse content to Chat
            chat = agent.parse_content(content, session_id, store)
            
            assert isinstance(chat, Chat)
            assert len(chat.messages) == 4
            
            # Check message details
            assert chat.messages[0].role == Role.USER
            assert chat.messages[0].content == "What's the weather like?"
            
            assert chat.messages[1].role == Role.ASSISTANT
            assert "I don't have access to real-time weather data" in chat.messages[1].content
            assert "but I can help you find weather information." in chat.messages[1].content
            
            assert chat.messages[3].role == Role.ASSISTANT
            assert "You can check weather using weather apps" in chat.messages[3].content

    def test_agent_detection_integration(self, sample_gemini_data):
        """Test agent detection with realistic data."""
        home_dir, session_id = sample_gemini_data
        
        log_file = home_dir / ".gemini" / "tmp" / f"{session_id}.jsonl"
        
        agent = GeminiCliAgent()
        assert agent.detect_agent(log_file) is True

    def test_list_logs_with_test_data(self, gemini_parser_with_test_data):
        """Test listing logs using test data files."""
        logs = gemini_parser_with_test_data.list_logs()
        
        # Should find our test data files (excluding empty file)
        assert len(logs) >= 2
        
        log_uris = [log[0] for log in logs]
        assert "simple" in log_uris
        assert "complex" in log_uris
        
        # Check metadata structure
        for log_uri, metadata in logs:
            assert isinstance(log_uri, str)
            assert isinstance(metadata, dict)
            assert "size" in metadata
            assert "modified" in metadata
            assert "accessible" in metadata

    def test_parse_simple_gemini_chat(self, gemini_test_data_path, gemini_parser_with_test_data):
        """Test parsing simple Gemini chat from test data."""
        chat = gemini_parser_with_test_data.parse("simple")
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        assert len(chat.messages) == 4
        
        # Check first message
        assert chat.messages[0].role.value == "user"
        assert "Hello, Gemini! Can you help me with Python?" in chat.messages[0].content
        
        # Check assistant response
        assert chat.messages[1].role.value == "assistant"
        assert "I'd be happy to help you with Python" in chat.messages[1].content
        
        # Check list comprehension question
        assert chat.messages[2].role.value == "user"
        assert "How do I create a list comprehension?" in chat.messages[2].content
        
        # Check assistant response with list content
        assert chat.messages[3].role.value == "assistant"
        assert "List comprehensions in Python provide a concise way" in chat.messages[3].content
        assert "[expression for item in iterable if condition]" in chat.messages[3].content

    def test_parse_complex_gemini_chat(self, gemini_parser_with_test_data):
        """Test parsing complex Gemini chat with various field formats."""
        chat = gemini_parser_with_test_data.parse("complex")
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        assert len(chat.messages) >= 3  # At least 3 non-system messages
        
        # Should handle different role names and timestamp formats
        role_values = {msg.role.value for msg in chat.messages}
        assert "user" in role_values
        assert "assistant" in role_values
        
        # Check content from different field names (text, content, message)
        contents = [msg.content for msg in chat.messages]
        assert any("machine learning" in content.lower() for content in contents)
        assert any("main types" in content.lower() for content in contents)

    def test_parse_malformed_gemini_chat(self, gemini_parser_with_test_data, capsys):
        """Test parsing malformed Gemini chat handles errors gracefully."""
        # This should not crash, but may produce warnings
        chat = gemini_parser_with_test_data.parse("malformed")
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        # Should have some valid messages despite malformed lines
        assert len(chat.messages) >= 2
        
        # Check that we got valid messages
        assert chat.messages[0].content == "This is a valid line"
        assert "Another valid line after malformed ones" in [msg.content for msg in chat.messages]

    def test_direct_file_parsing(self, gemini_test_data_path):
        """Test parsing Gemini files directly using file paths."""
        simple_file = gemini_test_data_path / "simple_gemini_chat.jsonl"
        
        # Test agent detection
        agent = GeminiCliAgent()
        assert agent.detect_agent(simple_file) is True
        
        # Test session parsing
        session = GeminiSession(file_path=simple_file)
        session.load()
        
        assert len(session.records) == 4
        assert session.session_id == "gemini-session-001"
        
        # Test chat conversion
        chat = session.to_chat()
        assert len(chat.messages) == 4
        assert all(msg.content.strip() for msg in chat.messages)  # All messages have content

    def test_empty_file_handling(self, gemini_test_data_path):
        """Test handling of empty Gemini chat file."""
        empty_file = gemini_test_data_path / "empty_gemini_chat.jsonl"
        
        session = GeminiSession(file_path=empty_file)
        session.load()
        
        assert len(session.records) == 0
        
        chat = session.to_chat()
        assert len(chat.messages) == 0

    def test_composition_with_test_data(self, gemini_parser_with_test_data):
        """Test message selection and composition with test data."""
        # Select messages from simple chat
        gemini_parser_with_test_data.select("simple", [0, 1])  # First two messages
        
        # Create composition
        composition = gemini_parser_with_test_data.compose()
        
        assert "Hello, Gemini! Can you help me with Python?" in composition
        assert "I'd be happy to help you with Python" in composition
        
        # Clear and select different messages
        gemini_parser_with_test_data.clear_selection()
        gemini_parser_with_test_data.select("complex", [0])  # First message from complex chat
        
        composition2 = gemini_parser_with_test_data.compose()
        assert "machine learning" in composition2.lower()

    def test_multiple_files_integration(self, gemini_parser_with_test_data):
        """Test working with multiple Gemini chat files."""
        logs = gemini_parser_with_test_data.list_logs()
        
        # Parse all available logs
        all_chats = []
        for log_uri, _ in logs:
            chat = gemini_parser_with_test_data.parse(log_uri)
            all_chats.append(chat)
        
        # Should have parsed multiple chats
        assert len(all_chats) >= 2
        
        # All should be valid Chat objects
        for chat in all_chats:
            assert chat is not None
            assert hasattr(chat, 'messages')
            assert len(chat.messages) >= 1