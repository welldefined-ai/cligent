"""Basic integration tests for the chat parser framework."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from chat_parser import ChatParser, Chat, Message, Role


class TestChatParser:
    """Test the main ChatParser interface."""
    
    def test_parser_initialization(self):
        """Test ChatParser can be initialized with different agents."""
        parser = ChatParser("claude-code")
        assert parser.agent_name == "claude-code"
        assert parser.store is not None
        
    def test_parser_initialization_with_location(self):
        """Test ChatParser can be initialized with custom location."""
        custom_path = Path("/custom/path")
        parser = ChatParser("claude-code", location=custom_path)
        assert parser.agent_name == "claude-code"
        
    def test_unsupported_agent_raises_error(self):
        """Test that unsupported agent names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported agent"):
            ChatParser("unsupported-agent")
    
    @patch('chat_parser.parser.ChatParser._create_store')
    def test_list_logs_delegates_to_store(self, mock_create_store):
        """Test that list_logs delegates to the store."""
        mock_store = Mock()
        mock_store.list.return_value = [{"path": "/test.jsonl", "size": 100}]
        mock_create_store.return_value = mock_store
        
        parser = ChatParser("claude-code")
        result = parser.list_logs()
        
        mock_store.list.assert_called_once()
        assert result == [{"path": "/test.jsonl", "size": 100}]
    
    @patch('chat_parser.parser.ChatParser._create_store')
    @patch('chat_parser.parser.ChatParser._parse_content')
    def test_parse_specific_log(self, mock_parse_content, mock_create_store):
        """Test parsing a specific log file."""
        mock_chat = Mock(spec=Chat)
        mock_store = Mock()
        mock_store.get.return_value = "raw log content"
        mock_create_store.return_value = mock_store
        mock_parse_content.return_value = mock_chat
        
        parser = ChatParser("claude-code")
        result = parser.parse("/path/to/log.jsonl")
        
        mock_store.get.assert_called_once_with("/path/to/log.jsonl")
        mock_parse_content.assert_called_once_with("raw log content", "/path/to/log.jsonl")
        assert result == mock_chat
    
    @patch('chat_parser.parser.ChatParser._create_store')
    @patch('chat_parser.parser.ChatParser._parse_content')
    def test_parse_live_log(self, mock_parse_content, mock_create_store):
        """Test parsing the live log."""
        mock_chat = Mock(spec=Chat)
        mock_store = Mock()
        mock_store.live.return_value = "/live/log.jsonl"
        mock_store.get.return_value = "live log content"
        mock_create_store.return_value = mock_store
        mock_parse_content.return_value = mock_chat
        
        parser = ChatParser("claude-code")
        result = parser.parse()  # No path = live
        
        mock_store.live.assert_called_once()
        mock_store.get.assert_called_once_with("/live/log.jsonl")
        mock_parse_content.assert_called_once_with("live log content", "/live/log.jsonl")
        assert result == mock_chat
    
    def test_compose_with_messages(self):
        """Test composing with direct messages."""
        parser = ChatParser("claude-code")
        
        # Create mock messages
        msg1 = Mock(spec=Message)
        msg2 = Mock(spec=Message)
        
        # Create mock chat that these messages would create
        mock_chat = Mock(spec=Chat)
        mock_chat.export.return_value = "Test output"
        
        # Mock Chat constructor
        with patch('chat_parser.parser.Chat') as mock_chat_class:
            mock_chat_class.return_value = mock_chat
            
            result = parser.compose(msg1, msg2)
            
            mock_chat_class.assert_called_once_with(messages=[msg1, msg2])
            mock_chat.export.assert_called_once()
            assert result == "Test output"
    
    @patch('chat_parser.parser.ChatParser.parse')
    def test_compose_with_selected_messages(self, mock_parse):
        """Test composing with selected messages."""
        parser = ChatParser("claude-code")
        
        # Create mock messages and chat
        msg1, msg2 = Mock(spec=Message), Mock(spec=Message)
        mock_chat = Mock(spec=Chat)
        mock_chat.messages = [msg1, msg2, Mock(spec=Message)]
        mock_parse.return_value = mock_chat
        
        # Create result chat
        result_chat = Mock(spec=Chat)
        result_chat.export.return_value = "Selected output"
        
        with patch('chat_parser.parser.Chat') as mock_chat_class:
            mock_chat_class.return_value = result_chat
            
            # Select messages and compose
            parser.select("/path/to/log.jsonl", [0, 1])
            result = parser.compose()
            
            mock_chat_class.assert_called_once_with(messages=[msg1, msg2])
            assert result == "Selected output"
    
    @patch('chat_parser.parser.ChatParser.parse')
    def test_selection_management(self, mock_parse):
        """Test message selection management with log paths."""
        parser = ChatParser("claude-code")
        
        # Setup mock chat
        mock_chat = Mock(spec=Chat)
        msg1, msg2 = Mock(spec=Message), Mock(spec=Message)
        mock_chat.messages = [msg1, msg2]
        mock_parse.return_value = mock_chat
        
        log_uri = "/path/to/log.jsonl"
        
        # Test specific message selection
        parser.select(log_uri, [0, 1])
        assert parser.selected_messages == [msg1, msg2]
        
        # Test unselect specific messages
        parser.unselect(log_uri, [1])
        assert parser.selected_messages == [msg1]
        
        # Test full chat selection (all messages)
        parser.clear_selection()
        parser.select(log_uri)  # No indices = all messages
        assert parser.selected_messages == [msg1, msg2]
        
        # Test clear
        parser.clear_selection()
        assert parser.selected_messages == []
    
    def test_compose_without_selection_raises_error(self):
        """Test that composing without selection raises error."""
        parser = ChatParser("claude-code")
        
        with pytest.raises(ValueError, match="No messages selected"):
            parser.compose()
    
    def test_compose_mixed_types_raises_error(self):
        """Test that composing mixed message/chat types raises error."""
        parser = ChatParser("claude-code")
        msg = Mock(spec=Message)
        chat = Mock(spec=Chat)
        
        with pytest.raises(ValueError, match="Mixed types"):
            parser.compose(msg, chat)


class TestModels:
    """Test basic model functionality."""
    
    def test_message_creation(self):
        """Test Message can be created with required fields."""
        msg = Message(
            role=Role.USER,
            content="Hello world"
        )
        assert msg.role == Role.USER
        assert msg.content == "Hello world"
        assert msg.timestamp is None
        assert msg.metadata == {}
    
    def test_chat_creation(self):
        """Test Chat can be created with messages."""
        msg1 = Message(role=Role.USER, content="Hello")
        msg2 = Message(role=Role.ASSISTANT, content="Hi")
        
        chat = Chat(messages=[msg1, msg2])
        assert len(chat.messages) == 2
        assert chat.messages[0] == msg1
        assert chat.messages[1] == msg2




class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('chat_parser.parser.ChatParser._create_store')
    def test_store_error_propagation(self, mock_create_store):
        """Test that store errors are propagated properly."""
        mock_store = Mock()
        mock_store.list.side_effect = Exception("Store error")
        mock_create_store.return_value = mock_store
        
        parser = ChatParser("claude-code")
        
        with pytest.raises(Exception, match="Store error"):
            parser.list_logs()
    
    @patch('chat_parser.parser.ChatParser.parse')
    def test_invalid_message_indices(self, mock_parse):
        """Test selection with invalid indices is handled gracefully."""
        parser = ChatParser("claude-code")
        
        mock_chat = Mock(spec=Chat)
        msg1 = Mock(spec=Message)
        mock_chat.messages = [msg1]  # Only one message
        mock_parse.return_value = mock_chat
        
        # Select indices including invalid ones
        parser.select("/path/to/log.jsonl", [0, 1, 5, -1])
        
        # Should only select valid index
        assert len(parser.selected_messages) == 1
        assert parser.selected_messages[0] == msg1