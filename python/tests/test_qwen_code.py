"""Unit tests for Qwen Code agent implementation."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

from chat_parser.qwen.qwen_code import (
    QwenRecord, 
    QwenSession, 
    QwenStore, 
    QwenCodeAgent
)
from chat_parser.models import Role, Chat


class TestQwenRecord:
    """Test QwenRecord class functionality."""

    def test_load_basic_record(self):
        """Test loading a basic Qwen record."""
        json_data = {
            "type": "user",
            "role": "user",
            "content": "Hello, Qwen!",
            "timestamp": "2024-01-01T10:00:00Z",
            "session_id": "test-session-123",
            "model": "qwen-coder"
        }
        json_string = json.dumps(json_data)
        
        record = QwenRecord.load(json_string)
        
        assert record.type == "user"
        assert record.role == "user"
        assert record.content == "Hello, Qwen!"
        assert record.timestamp == "2024-01-01T10:00:00Z"
        assert record.session_id == "test-session-123"
        assert record.model == "qwen-coder"
        assert record.raw_data == json_data

    def test_load_with_alternative_fields(self):
        """Test loading record with alternative field names."""
        json_data = {
            "messageType": "assistant",
            "sender": "qwen",
            "text": "Hello, human!",
            "time": "1704103200",
            "conversationId": "conv-456",
            "modelName": "qwen-turbo"
        }
        json_string = json.dumps(json_data)
        
        record = QwenRecord.load(json_string)
        
        assert record.type == "assistant"
        assert record.role == "qwen"
        assert record.content == "Hello, human!"
        assert record.timestamp == "1704103200"
        assert record.session_id == "conv-456"
        assert record.model == "qwen-turbo"

    def test_load_with_checkpoint_fields(self):
        """Test loading record with checkpoint-specific fields."""
        json_data = {
            "type": "checkpoint",
            "checkpoint_tag": "checkpoint_001",
            "checkpointTag": "alternate_name",  # Should use checkpoint_tag over checkpointTag
            "timestamp": "2024-01-01T12:00:00+08:00",
            "session_id": "checkpoint-session"
        }
        json_string = json.dumps(json_data)
        
        record = QwenRecord.load(json_string)
        
        assert record.type == "checkpoint"
        assert record.checkpoint_tag == "checkpoint_001"  # Should prioritize checkpoint_tag
        assert record.timestamp == "2024-01-01T12:00:00+08:00"

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON record"):
            QwenRecord.load("invalid json")

    def test_extract_message_user_role(self):
        """Test extracting message from user record."""
        record = QwenRecord(
            type="user",
            role="user",
            content="Test message",
            timestamp="2024-01-01T10:00:00Z",
            session_id="test-session"
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert message.role.value == "user"
        assert message.content == "Test message"
        assert message.timestamp == datetime.fromisoformat("2024-01-01T10:00:00+00:00")
        assert message.metadata["session_id"] == "test-session"

    def test_extract_message_qwen_role(self):
        """Test extracting message with 'qwen' role mapped to assistant."""
        record = QwenRecord(
            type="assistant",
            role="qwen",
            content="Assistant response",
            timestamp="1704103200",  # Unix timestamp
            model="qwen-coder"
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert message.role.value == "assistant"
        assert message.content == "Assistant response"
        assert message.timestamp == datetime.fromtimestamp(1704103200)
        assert message.metadata["model"] == "qwen-coder"

    def test_extract_message_list_content(self):
        """Test extracting message with list content."""
        record = QwenRecord(
            type="assistant",
            role="assistant",
            content=[
                {"type": "text", "text": "First part"},
                {"text": "Second part"},
                "Third part string"
            ]
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert message.content == "First part\nSecond part\nThird part string"

    def test_extract_message_dict_content(self):
        """Test extracting message with dict content."""
        record = QwenRecord(
            type="user",
            role="user",
            content={"text": "Message from dict"}
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert message.content == "Message from dict"

    def test_extract_message_filters_tool_blocks(self):
        """Test extracting message filters out tool blocks."""
        record = QwenRecord(
            type="assistant",
            role="assistant",
            content=[
                {"type": "text", "text": "Here's some code:"},
                {"tool_use": "# This should be filtered"},  # This format triggers the filter
                {"text": "And here's the explanation."}
            ]
        )
        
        message = record.extract_message()
        
        assert message is not None
        assert "Here's some code:" in message.content
        assert "And here's the explanation." in message.content
        assert "This should be filtered" not in message.content

    def test_extract_message_empty_content(self):
        """Test extracting message with empty content returns None."""
        record = QwenRecord(
            type="user",
            role="user",
            content=""
        )
        
        message = record.extract_message()
        assert message is None

    def test_extract_message_invalid_timestamp(self):
        """Test extracting message with invalid timestamp."""
        record = QwenRecord(
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
        message_types = ['user', 'assistant', 'system', 'human', 'ai', 'qwen', 'message']
        
        for msg_type in message_types:
            record = QwenRecord(type=msg_type)
            assert record.is_message()
        
        non_message_types = ['tool_use', 'tool_result', 'checkpoint']
        for msg_type in non_message_types:
            record = QwenRecord(type=msg_type)
            assert not record.is_message()

    def test_is_message_by_role(self):
        """Test is_message detection by role field."""
        message_roles = ['user', 'assistant', 'system', 'human', 'ai', 'qwen']
        
        for role in message_roles:
            record = QwenRecord(type="unknown", role=role)
            assert record.is_message()
        
        record = QwenRecord(type="unknown", role="tool")
        assert not record.is_message()

    def test_role_mapping_variations(self):
        """Test various role mapping scenarios."""
        role_tests = [
            ("user", "user"),
            ("human", "user"),
            ("assistant", "assistant"),
            ("qwen", "assistant"),
            ("ai", "assistant"),
            ("model", "assistant"),
            ("system", "system"),
            ("unknown", "assistant"),  # Default fallback
        ]
        
        for qwen_role, expected_role in role_tests:
            record = QwenRecord(type="message", role=qwen_role, content="test")
            message = record.extract_message()
            assert message.role.value == expected_role


class TestQwenSession:
    """Test QwenSession class functionality."""

    @pytest.fixture
    def temp_jsonl_file(self, tmp_path):
        """Create a temporary JSONL file for testing."""
        file_path = tmp_path / "test_session.jsonl"
        
        records = [
            {"type": "user", "role": "user", "content": "Hello", "session_id": "test-123"},
            {"type": "checkpoint", "checkpoint_tag": "start", "session_id": "test-123"},
            {"type": "assistant", "role": "qwen", "content": "Hi there!", "session_id": "test-123"},
            {"type": "user", "role": "user", "content": "How are you?", "session_id": "test-123"},
            {"type": "assistant", "role": "qwen", "content": "I'm doing well!", "session_id": "test-123"}
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
        session = QwenSession(file_path=temp_jsonl_file)
        session.load()
        
        assert len(session.records) == 5
        assert session.session_id == "test-123"
        assert "start" in session.checkpoint_tags
        assert session.records[0].content == "Hello"
        assert session.records[2].role == "qwen"

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        session = QwenSession(file_path=Path("/nonexistent/file.jsonl"))
        
        with pytest.raises(FileNotFoundError, match="Log file not found"):
            session.load()

    def test_load_malformed_file(self, malformed_jsonl_file, capsys):
        """Test loading file with malformed JSON lines."""
        session = QwenSession(file_path=malformed_jsonl_file)
        session.load()
        
        # Should have 2 valid records despite malformed line
        assert len(session.records) == 2
        
        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning: Skipped invalid record" in captured.out

    def test_to_chat(self, temp_jsonl_file):
        """Test converting session to Chat object."""
        session = QwenSession(file_path=temp_jsonl_file)
        session.load()
        
        chat = session.to_chat()
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        assert len(chat.messages) == 4  # Excludes checkpoint records
        assert chat.messages[0].role.value == "user"
        assert chat.messages[1].role.value == "assistant"
        assert chat.messages[0].content == "Hello"

    def test_empty_file(self, tmp_path):
        """Test loading empty file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.touch()
        
        session = QwenSession(file_path=empty_file)
        session.load()
        
        assert len(session.records) == 0
        assert len(session.checkpoint_tags) == 0
        chat = session.to_chat()
        assert len(chat.messages) == 0


class TestQwenStore:
    """Test QwenStore class functionality."""

    @pytest.fixture
    def mock_home_dir(self, tmp_path):
        """Create mock home directory structure."""
        home_dir = tmp_path / "home"
        qwen_dir = home_dir / ".qwen" / "logs"
        qwen_dir.mkdir(parents=True)
        
        # Create some test log files
        (qwen_dir / "session1.jsonl").write_text('{"content": "test1", "model": "qwen"}')
        (qwen_dir / "session2.jsonl").write_text('{"content": "test2", "model": "qwen"}')
        
        return home_dir

    def test_init_with_default_location(self):
        """Test initializing store with default location."""
        store = QwenStore()
        assert store.agent == "qwen-code"

    def test_init_without_location(self):
        """Test initializing store (location not supported for Qwen)."""
        store = QwenStore()
        assert store.agent == "qwen-code"

    def test_list_logs(self, mock_home_dir):
        """Test listing available logs."""
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = QwenStore()
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
            store = QwenStore()
            logs = store.list()
        
        assert logs == []

    def test_get_log_by_session_id(self, mock_home_dir):
        """Test retrieving log content by session ID."""
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = QwenStore()
            content = store.get("session1")
        
        assert content == '{"content": "test1", "model": "qwen"}'

    def test_get_log_by_full_path(self, mock_home_dir):
        """Test retrieving log content by full path."""
        log_path = mock_home_dir / ".qwen" / "logs" / "session1.jsonl"
        
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = QwenStore()
            content = store.get(str(log_path))
        
        assert content == '{"content": "test1", "model": "qwen"}'

    def test_get_nonexistent_log(self, mock_home_dir):
        """Test retrieving non-existent log raises FileNotFoundError."""
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = QwenStore()
            
            with pytest.raises(FileNotFoundError, match="Log file not found"):
                store.get("nonexistent")

    def test_live_log(self, mock_home_dir):
        """Test getting most recent log."""
        with patch('pathlib.Path.home', return_value=mock_home_dir):
            store = QwenStore()
            live_log = store.live()
        
        # Should return one of the sessions (most recent)
        assert live_log in ["session1", "session2"]

    def test_live_log_no_logs(self, tmp_path):
        """Test getting live log when no logs exist."""
        fake_home = tmp_path / "fake_home"
        with patch('pathlib.Path.home', return_value=fake_home):
            store = QwenStore()
            live_log = store.live()
        
        assert live_log is None

    def test_directory_search_order(self, tmp_path):
        """Test directory search follows priority order."""
        home_dir = tmp_path / "home"
        qwen_base = home_dir / ".qwen"
        
        # Create multiple directories with different priorities
        sessions_dir = qwen_base / "sessions"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "session.jsonl").write_text('{"sessions": "data"}')
        
        conversations_dir = qwen_base / "conversations"  
        conversations_dir.mkdir(parents=True)
        (conversations_dir / "conv.jsonl").write_text('{"conversations": "data"}')
        
        # logs should have highest priority
        logs_dir = qwen_base / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "log.jsonl").write_text('{"logs": "data"}')
        
        with patch('pathlib.Path.home', return_value=home_dir):
            store = QwenStore()
            assert store._logs_dir == logs_dir  # Should choose logs over others
            logs = store.list()
            assert len(logs) == 1
            assert logs[0][0] == "log"


class TestQwenCodeAgent:
    """Test QwenCodeAgent class functionality."""

    def test_config_properties(self):
        """Test agent configuration properties."""
        agent = QwenCodeAgent()
        config = agent.config
        
        assert config.name == "qwen-code"
        assert config.display_name == "Qwen Code"
        assert config.log_extensions == [".jsonl", ".json"]
        assert config.requires_session_id is True
        assert config.metadata["log_format"] == "jsonl"
        assert config.metadata["base_dir"] == "~/.qwen/"
        assert config.metadata["supports_checkpoints"] is True
        assert config.metadata["supports_tools"] is True
        assert config.metadata["based_on"] == "gemini-cli"

    def test_create_store(self):
        """Test creating store instance."""
        agent = QwenCodeAgent()
        store = agent.create_store()
        
        assert isinstance(store, QwenStore)

    def test_create_store_with_location(self):
        """Test creating store with custom location."""
        agent = QwenCodeAgent()
        store = agent.create_store(location="/custom/path")
        
        assert isinstance(store, QwenStore)

    def test_parse_content_with_session_id(self, tmp_path):
        """Test parsing content using session ID."""
        # Create a temporary JSONL file
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_file = logs_dir / "test-session.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"type": "user", "role": "user", "content": "Hello"}\n')
            f.write('{"type": "assistant", "role": "qwen", "content": "Hi!"}\n')
        
        # Mock the store's _logs_dir
        agent = QwenCodeAgent()
        store = agent.create_store()
        store._logs_dir = logs_dir
        
        content = store.get("test-session")
        chat = agent.parse_content(content, "test-session", store)
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        assert len(chat.messages) == 2
        assert chat.messages[0].role.value == "user"
        assert chat.messages[0].content == "Hello"

    def test_parse_content_with_full_path(self, tmp_path):
        """Test parsing content using full file path."""
        log_file = tmp_path / "full_path_test.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"type": "user", "role": "user", "content": "Test message"}\n')
        
        agent = QwenCodeAgent()
        store = agent.create_store()
        
        content = '{"type": "user", "role": "user", "content": "Test message"}\n'
        chat = agent.parse_content(content, str(log_file), store)
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        assert len(chat.messages) == 1
        assert chat.messages[0].content == "Test message"

    def test_detect_agent_jsonl_file(self, tmp_path):
        """Test detecting Qwen logs from JSONL file."""
        log_file = tmp_path / "qwen_log.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"role": "user", "content": "Hello Qwen"}\n')
            f.write('{"role": "qwen", "content": "Hi there!"}\n')
        
        agent = QwenCodeAgent()
        assert agent.detect_agent(log_file) is True

    def test_detect_agent_with_qwen_indicators(self, tmp_path):
        """Test detecting logs with Qwen-specific indicators."""
        test_cases = [
            # Model name starts with qwen
            '{"model": "qwen-coder", "role": "user", "text": "Test"}',
            # Contains qwen in data
            '{"role": "user", "content": "Using qwen model"}',
            # Role is qwen  
            '{"role": "qwen", "content": "Response"}',
            # Has checkpoint fields
            '{"checkpoint": "test", "content": "Test"}',
            # References QWEN.md with contextFileName
            '{"contextFileName": "QWEN.md", "content": "Test"}'
        ]
        
        agent = QwenCodeAgent()
        
        for i, test_case in enumerate(test_cases):
            log_file = tmp_path / f"test_{i}.jsonl"
            log_file.write_text(test_case)
            assert agent.detect_agent(log_file) is True

    def test_detect_agent_wrong_extension(self, tmp_path):
        """Test detection fails for wrong file extension."""
        log_file = tmp_path / "test.txt"
        log_file.write_text('{"role": "qwen", "content": "Test"}')
        
        agent = QwenCodeAgent()
        assert agent.detect_agent(log_file) is False

    def test_detect_agent_non_qwen_content(self, tmp_path):
        """Test detection fails for non-Qwen content."""
        log_file = tmp_path / "other.jsonl"
        
        with open(log_file, 'w') as f:
            f.write('{"tool": "other", "data": "Not AI", "role": "tool"}\n')
        
        agent = QwenCodeAgent()
        assert agent.detect_agent(log_file) is False

    def test_detect_agent_malformed_json(self, tmp_path):
        """Test detection handles malformed JSON gracefully."""
        log_file = tmp_path / "malformed.jsonl"
        log_file.write_text('invalid json content')
        
        agent = QwenCodeAgent()
        assert agent.detect_agent(log_file) is False

    def test_detect_agent_empty_file(self, tmp_path):
        """Test detection handles empty files gracefully."""
        log_file = tmp_path / "empty.jsonl"
        log_file.touch()
        
        agent = QwenCodeAgent()
        assert agent.detect_agent(log_file) is False


class TestQwenIntegration:
    """Integration tests for Qwen Code components."""

    @pytest.fixture
    def test_data_path(self):
        """Path to test data directory."""
        return Path(__file__).parent / "test_data"

    @pytest.fixture
    def qwen_test_data_path(self, test_data_path):
        """Path to Qwen test data directory."""
        return test_data_path / "qwen_project"

    @pytest.fixture
    def mock_qwen_home(self):
        """Path to mock Qwen home directory."""
        return Path(__file__).parent / "mock_qwen_home"

    @pytest.fixture
    def qwen_parser(self, mock_qwen_home):
        """ChatParser instance using mock Qwen environment."""
        with patch('pathlib.Path.home', return_value=mock_qwen_home):
            from cligent import ChatParser
            return ChatParser("qwen-code")

    @pytest.fixture
    def qwen_parser_with_test_data(self, qwen_test_data_path):
        """ChatParser instance using test data files."""
        import tempfile
        import shutil
        
        # Create temporary home directory structure
        temp_home = Path(tempfile.mkdtemp())
        qwen_dir = temp_home / ".qwen" / "logs"
        qwen_dir.mkdir(parents=True)
        
        try:
            # Copy test data files to the temporary structure
            for test_file in qwen_test_data_path.glob("*.jsonl"):
                if test_file.name != "empty_qwen_chat.jsonl":  # Skip empty file
                    # Use test file name as session name
                    session_name = test_file.stem.replace("_qwen_chat", "")
                    dest_file = qwen_dir / f"{session_name}.jsonl"
                    shutil.copy2(test_file, dest_file)
            
            with patch('pathlib.Path.home', return_value=temp_home):
                from cligent import ChatParser
                yield ChatParser("qwen-code")
        finally:
            # Cleanup
            shutil.rmtree(temp_home, ignore_errors=True)

    @pytest.fixture
    def sample_qwen_data(self, tmp_path):
        """Create sample Qwen data structure."""
        home_dir = tmp_path / "home"
        qwen_dir = home_dir / ".qwen" / "logs"
        qwen_dir.mkdir(parents=True)
        
        # Create a realistic Qwen conversation with checkpoints
        conversation = [
            {
                "type": "user",
                "role": "user",
                "content": "Can you help me with algorithms?",
                "timestamp": "2024-01-01T10:00:00Z",
                "session_id": "algo-session-123",
                "model": "qwen-coder"
            },
            {
                "type": "checkpoint",
                "checkpoint_tag": "session_start",
                "timestamp": "2024-01-01T10:00:01Z",
                "session_id": "algo-session-123"
            },
            {
                "type": "assistant",
                "role": "qwen",
                "content": [
                    {"type": "text", "text": "Absolutely! I'd be happy to help you with algorithms. "},
                    {"type": "text", "text": "What specific algorithm topic would you like to explore?"}
                ],
                "timestamp": "2024-01-01T10:00:05Z",
                "session_id": "algo-session-123",
                "model": "qwen-coder"
            },
            {
                "type": "user",
                "role": "user",
                "content": "How does quicksort work?",
                "timestamp": "2024-01-01T10:01:00Z",
                "session_id": "algo-session-123"
            },
            {
                "type": "checkpoint",
                "checkpoint_tag": "quicksort_discussion",
                "timestamp": "2024-01-01T10:01:01Z",
                "session_id": "algo-session-123"
            },
            {
                "type": "assistant",
                "role": "qwen",
                "content": {
                    "text": "Quicksort is a divide-and-conquer algorithm that works by selecting a pivot element and partitioning the array around it."
                },
                "timestamp": "2024-01-01T10:01:10Z",
                "session_id": "algo-session-123"
            }
        ]
        
        log_file = qwen_dir / "algo-session-123.jsonl"
        with open(log_file, 'w') as f:
            for record in conversation:
                f.write(json.dumps(record) + '\n')
        
        return home_dir, "algo-session-123"

    def test_chatparser_integration_with_mock_home(self, qwen_parser, mock_qwen_home):
        """Test ChatParser integration using mock home directory."""
        with patch('pathlib.Path.home', return_value=mock_qwen_home):
            logs = qwen_parser.list_logs()
            
        # Should find our mock log files
        assert len(logs) >= 2
        
        log_uris = [log[0] for log in logs]
        assert "session-20240101-001" in log_uris
        assert "session-20240101-002" in log_uris
        
        # Check metadata structure
        for log_uri, metadata in logs:
            assert isinstance(log_uri, str)
            assert isinstance(metadata, dict)
            assert "size" in metadata
            assert "modified" in metadata
            assert "accessible" in metadata

    def test_chatparser_parse_qwen_logs(self, qwen_parser, mock_qwen_home):
        """Test parsing actual Qwen logs through ChatParser."""
        with patch('pathlib.Path.home', return_value=mock_qwen_home):
            logs = qwen_parser.list_logs()
            
            if logs:
                # Parse the first available log
                log_uri = logs[0][0]
                chat = qwen_parser.parse(log_uri)
                
                assert chat is not None
                assert hasattr(chat, 'messages')
                assert len(chat.messages) >= 1
                
                # Verify message structure
                for message in chat.messages:
                    assert message.role.value in ["user", "assistant", "system"]
                    assert isinstance(message.content, str)
                    assert len(message.content.strip()) > 0

    def test_qwen_alias_integration(self):
        """Test that 'qwen' alias works with ChatParser."""
        from cligent import ChatParser
        
        # Test both full name and alias
        parser1 = ChatParser("qwen-code")
        parser2 = ChatParser("qwen")
        
        assert parser1.agent.config.name == "qwen-code"
        assert parser2.agent.config.name == "qwen-code"
        assert parser1.agent.config.display_name == parser2.agent.config.display_name

    def test_list_logs_with_test_data(self, qwen_parser_with_test_data):
        """Test listing logs using test data files."""
        logs = qwen_parser_with_test_data.list_logs()
        
        # Should find our test data files (excluding empty file)
        assert len(logs) >= 3
        
        log_uris = [log[0] for log in logs]
        assert "simple" in log_uris
        assert "complex" in log_uris
        assert "checkpoint" in log_uris
        
        # Check metadata structure
        for log_uri, metadata in logs:
            assert isinstance(log_uri, str)
            assert isinstance(metadata, dict)
            assert "size" in metadata
            assert "modified" in metadata
            assert "accessible" in metadata

    def test_parse_simple_qwen_chat(self, qwen_parser_with_test_data):
        """Test parsing simple Qwen chat from test data."""
        chat = qwen_parser_with_test_data.parse("simple")
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        assert len(chat.messages) == 4
        
        # Check first message
        assert chat.messages[0].role.value == "user"
        assert "Hello, Qwen! Can you help me with Python programming?" in chat.messages[0].content
        
        # Check assistant response with qwen role mapping
        assert chat.messages[1].role.value == "assistant"
        assert "I'm Qwen, and I'd be delighted to help you" in chat.messages[1].content
        
        # Check algorithm question
        assert chat.messages[2].role.value == "user"
        assert "How do I implement a binary search algorithm?" in chat.messages[2].content
        
        # Check code response
        assert chat.messages[3].role.value == "assistant"
        assert "def binary_search" in chat.messages[3].content

    def test_parse_complex_qwen_chat(self, qwen_parser_with_test_data):
        """Test parsing complex Qwen chat with various field formats."""
        chat = qwen_parser_with_test_data.parse("complex")
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        assert len(chat.messages) >= 3  # At least 3 non-checkpoint/system messages
        
        # Should handle different role names and timestamp formats
        role_values = {msg.role.value for msg in chat.messages}
        assert "user" in role_values
        assert "assistant" in role_values
        
        # Check content from different field names (text, content, message)
        contents = [msg.content for msg in chat.messages]
        assert any("machine learning" in content.lower() for content in contents)
        assert any("supervised" in content.lower() for content in contents)

    def test_parse_checkpoint_chat(self, qwen_parser_with_test_data):
        """Test parsing chat with checkpoint records."""
        chat = qwen_parser_with_test_data.parse("checkpoint")
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        # Should have messages but not checkpoint records
        assert len(chat.messages) == 4  # User and assistant messages only
        
        # Verify no checkpoint records in messages
        for message in chat.messages:
            assert "checkpoint" not in message.metadata.get("type", "")

    def test_parse_malformed_qwen_chat(self, qwen_parser_with_test_data, capsys):
        """Test parsing malformed Qwen chat handles errors gracefully."""
        # This should not crash, but may produce warnings
        chat = qwen_parser_with_test_data.parse("malformed")
        
        assert chat is not None
        assert hasattr(chat, 'messages')
        # Should have some valid messages despite malformed lines
        assert len(chat.messages) >= 2
        
        # Check that we got valid messages
        assert chat.messages[0].content == "This is a valid line"
        assert "Another valid line after malformed ones" in [msg.content for msg in chat.messages]

    def test_direct_file_parsing(self, qwen_test_data_path):
        """Test parsing Qwen files directly using file paths."""
        simple_file = qwen_test_data_path / "simple_qwen_chat.jsonl"
        
        # Test agent detection
        agent = QwenCodeAgent()
        assert agent.detect_agent(simple_file) is True
        
        # Test session parsing
        session = QwenSession(file_path=simple_file)
        session.load()
        
        assert len(session.records) == 4
        assert session.session_id == "qwen-session-001"
        
        # Test chat conversion
        chat = session.to_chat()
        assert len(chat.messages) == 4
        assert all(msg.content.strip() for msg in chat.messages)  # All messages have content

    def test_empty_file_handling(self, qwen_test_data_path):
        """Test handling of empty Qwen chat file."""
        empty_file = qwen_test_data_path / "empty_qwen_chat.jsonl"
        
        session = QwenSession(file_path=empty_file)
        session.load()
        
        assert len(session.records) == 0
        assert len(session.checkpoint_tags) == 0
        
        chat = session.to_chat()
        assert len(chat.messages) == 0

    def test_composition_with_test_data(self, qwen_parser_with_test_data):
        """Test message selection and composition with test data."""
        # Select messages from simple chat
        qwen_parser_with_test_data.select("simple", [0, 1])  # First two messages
        
        # Create composition
        composition = qwen_parser_with_test_data.compose()
        
        assert "Hello, Qwen! Can you help me with Python programming?" in composition
        assert "I'm Qwen, and I'd be delighted to help you" in composition
        
        # Clear and select different messages
        qwen_parser_with_test_data.clear_selection()
        qwen_parser_with_test_data.select("complex", [0])  # First message from complex chat
        
        composition2 = qwen_parser_with_test_data.compose()
        assert "machine learning" in composition2.lower()

    def test_end_to_end_parsing(self, sample_qwen_data):
        """Test complete end-to-end parsing workflow with checkpoints."""
        home_dir, session_id = sample_qwen_data
        
        with patch('pathlib.Path.home', return_value=home_dir):
            agent = QwenCodeAgent()
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
            
            assert chat is not None
            assert hasattr(chat, 'messages')
            assert len(chat.messages) == 4  # Excludes checkpoint records
            
            # Check message details
            assert chat.messages[0].role.value == "user"
            assert chat.messages[0].content == "Can you help me with algorithms?"
            
            assert chat.messages[1].role.value == "assistant"
            assert "Absolutely! I'd be happy to help you with algorithms" in chat.messages[1].content
            assert "What specific algorithm topic would you like to explore?" in chat.messages[1].content
            
            assert chat.messages[3].role.value == "assistant"
            assert "Quicksort is a divide-and-conquer algorithm" in chat.messages[3].content

    def test_multiple_files_integration(self, qwen_parser_with_test_data):
        """Test working with multiple Qwen chat files."""
        logs = qwen_parser_with_test_data.list_logs()
        
        # Parse all available logs
        all_chats = []
        for log_uri, _ in logs:
            chat = qwen_parser_with_test_data.parse(log_uri)
            all_chats.append(chat)
        
        # Should have parsed multiple chats
        assert len(all_chats) >= 3
        
        # All should be valid Chat objects
        for chat in all_chats:
            assert chat is not None
            assert hasattr(chat, 'messages')
            assert len(chat.messages) >= 1