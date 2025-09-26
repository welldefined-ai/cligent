from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import yaml
from datetime import datetime
from pathlib import Path
from .core.models import Chat, Message
from .core.errors import ErrorCollector

if TYPE_CHECKING:
    from .core.models import LogStore

class Cligent(ABC):
    """Abstract base class for all agent implementations."""

    def __init__(self):
        """Initialize agent backend."""
        self.error_collector = ErrorCollector()
        self.selected_messages: List[Message] = []
        self._chat_cache: Dict[str, Chat] = {}
        # Automatically create store during initialization
        self._store: 'LogStore' = self._create_store()

    @property
    @abstractmethod 
    def name(self) -> str:
        """Agent name identifier."""
        pass
        
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Agent display name."""
        pass

    @abstractmethod
    def _create_store(self) -> 'LogStore':
        """Create appropriate log store for this agent."""
        pass

    @abstractmethod
    def parse_content(self, content: str, log_uri: str) -> 'Chat':
        """Parse raw log content into Chat object."""
        pass





    # Log Store Management
    @property
    def store(self) -> 'LogStore':
        """Get log store for this agent."""
        return self._store

    # Log Parsing Methods
    def list_logs(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available logs for the agent.

        Returns:
            List of (log_uri, metadata) tuples
        """
        return self.store.list()

    def parse(self, log_uri: str = None) -> Chat:
        """Extract chat from specific or live session log.

        Args:
            log_uri: Log URI (None for live session log)

        Returns:
            Parsed Chat object
        """
        if log_uri:
            content = self.store.get(log_uri)
            return self.parse_content(content, log_uri)
        else:
            live_uri = self.store.live()
            if live_uri is None:
                return None
            content = self.store.get(live_uri)
            return self.parse_content(content, live_uri)

    def compose(self, *args) -> str:
        """Create Tigs text output from selected content.

        Args:
            *args: Messages or chats to compose (uses selected items if none provided)

        Returns:
            Tigs text output
        """
        # Determine what to compose
        if args:
            items = list(args)
            if all(isinstance(item, Message) for item in items):
                chat = Chat(messages=items)
            elif all(isinstance(item, Chat) for item in items):
                chat = self._merge_chats(items)
            else:
                raise ValueError("Mixed types in composition")
        else:
            # Use selected messages
            if not self.selected_messages:
                raise ValueError("No messages selected for composition")
            chat = Chat(messages=self.selected_messages)

        return chat.export()

    def decompose(self, tigs_yaml: str) -> Chat:
        """Convert Tigs YAML format back to Chat object.

        Args:
            tigs_yaml: Tigs-formatted YAML string

        Returns:
            Chat object with messages parsed from YAML

        Raises:
            ValueError: If YAML is invalid or not in Tigs format
        """
        try:
            data = yaml.safe_load(tigs_yaml)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

        # Validate Tigs format
        if not isinstance(data, dict):
            raise ValueError("YAML must contain a dictionary")

        if data.get('schema') != 'tigs.chat/v1':
            raise ValueError(f"Expected schema 'tigs.chat/v1', got '{data.get('schema')}'")

        if 'messages' not in data:
            raise ValueError("YAML must contain 'messages' field")

        if not isinstance(data['messages'], list):
            raise ValueError("'messages' field must be a list")

        # Parse messages
        messages = []
        from .core.models import Role

        for i, msg_data in enumerate(data['messages']):
            if not isinstance(msg_data, dict):
                raise ValueError(f"Message {i} must be a dictionary")

            if 'role' not in msg_data:
                raise ValueError(f"Message {i} must have 'role' field")

            if 'content' not in msg_data:
                raise ValueError(f"Message {i} must have 'content' field")

            # Parse role
            role_str = msg_data['role'].lower()
            try:
                role = Role(role_str)
            except ValueError:
                raise ValueError(f"Message {i} has invalid role '{role_str}'")

            # Parse content (handle both string and multiline formats)
            content = msg_data['content']
            if isinstance(content, str):
                content = content.strip()
            else:
                content = str(content).strip()

            # Parse timestamp if present
            timestamp = None
            if 'timestamp' in msg_data:
                timestamp_str = msg_data['timestamp']
                if timestamp_str:
                    try:
                        # Handle ISO format timestamps
                        if timestamp_str.endswith('Z'):
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        elif '+' in timestamp_str or '-' in timestamp_str[-6:]:
                            timestamp = datetime.fromisoformat(timestamp_str)
                        else:
                            timestamp = datetime.fromisoformat(timestamp_str)
                    except ValueError:
                        # If timestamp parsing fails, just skip it
                        pass

            # Parse log_uri if present
            log_uri = msg_data.get('log_uri', '')

            # Create message
            message = Message(
                role=role,
                content=content,
                provider=self.name,
                log_uri=log_uri,
                timestamp=timestamp,
                raw_data=msg_data
            )
            messages.append(message)

        return Chat(messages=messages)

    def select(self, log_uri: str, indices: List[int] = None) -> None:
        """Select messages for composition.

        Args:
            log_uri: Log URI
            indices: Message indices to select (None for all messages)
        """
        # Get or cache the chat
        if log_uri not in self._chat_cache:
            self._chat_cache[log_uri] = self.parse(log_uri)

        chat = self._chat_cache[log_uri]

        if indices is None:
            # Select all messages from the chat
            self.selected_messages.extend(chat.messages)
        else:
            # Select specific messages
            for i in indices:
                if 0 <= i < len(chat.messages):
                    self.selected_messages.append(chat.messages[i])

    def unselect(self, log_uri: str, indices: List[int] = None) -> None:
        """Remove messages from selection.

        Args:
            log_uri: Log URI
            indices: Message indices to unselect (None for all messages from this log)
        """
        # Get or cache the chat
        if log_uri not in self._chat_cache:
            self._chat_cache[log_uri] = self.parse(log_uri)

        chat = self._chat_cache[log_uri]

        if indices is None:
            # Remove all messages from this chat
            self.selected_messages = [msg for msg in self.selected_messages
                                    if msg not in chat.messages]
        else:
            # Remove specific messages
            messages_to_remove = []
            for i in indices:
                if 0 <= i < len(chat.messages):
                    messages_to_remove.append(chat.messages[i])

            self.selected_messages = [msg for msg in self.selected_messages
                                    if msg not in messages_to_remove]

    def clear_selection(self) -> None:
        """Clear current selection."""
        self.selected_messages.clear()
        self._chat_cache.clear()

    def _merge_chats(self, chats: List[Chat]) -> Chat:
        """Merge multiple chats into one.

        Args:
            chats: List of chats to merge

        Returns:
            Merged chat
        """
        if not chats:
            return Chat()

        if len(chats) == 1:
            return chats[0]

        # Use the merge method from the first chat
        result = chats[0]
        for chat in chats[1:]:
            result = result.merge(chat)

        return result

    def get_errors(self) -> Optional[str]:
        """Get error report if any errors occurred.

        Returns:
            Error report string or None
        """
        if self.error_collector.has_errors():
            return self.error_collector.generate_report()
        return None

    # Agent Information
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about current agent."""
        return {
            "name": self.name,
            "display_name": self.display_name,
        }


    def __repr__(self) -> str:
        """String representation of agent."""
        return f"{self.__class__.__name__}(name='{self.name}')"


# Factory function for creating agent instances
def create(agent_type: str = "claude"):
    """Create an agent for the specified type.
    
    Args:
        agent_type: Agent type ("claude", "gemini", "qwen")
        
    Returns:
        Appropriate agent instance
        
    Raises:
        ValueError: If agent_type is not supported
    """
    if agent_type in ["claude", "claude-code"]:
        from .agents.claude_code import ClaudeCligent
        return ClaudeCligent()
    elif agent_type in ["gemini", "gemini-cli"]:
        from .agents.gemini_cli import GeminiCligent
        return GeminiCligent()
    elif agent_type in ["qwen", "qwen-code"]:
        from .agents.qwen_code import QwenCligent
        return QwenCligent()
    else:
        supported_types = ["claude", "claude-code", "gemini", "gemini-cli", "qwen", "qwen-code"]
        raise ValueError(f"Unsupported agent type: {agent_type}. Supported: {supported_types}")