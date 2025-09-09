from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from pathlib import Path
from .models import Chat, Message
from .errors import ErrorCollector

if TYPE_CHECKING:
    from .models import LogStore

@dataclass
class AgentConfig:
    """Agent configuration and metadata."""
    name: str
    display_name: str
    log_extensions: List[str]  # [".jsonl", ".log"]
    default_log_dir: Optional[Path] = None
    requires_session_id: bool = True
    metadata: Dict[str, Any] = None

class AgentBackend(ABC):
    """Abstract base class for all agent implementations."""

    def __init__(self, location: Optional[str] = None):
        """Initialize agent backend.
        
        Args:
            location: Optional workspace location for logs
        """
        self.location = location
        self.error_collector = ErrorCollector()
        self.selected_messages: List[Message] = []
        self._chat_cache: Dict[str, Chat] = {}
        # Automatically create store during initialization
        self._store: 'LogStore' = self._create_store(location)

    @property
    @abstractmethod
    def config(self) -> AgentConfig:
        """Agent configuration and metadata."""
        pass

    @abstractmethod
    def _create_store(self, location: Optional[str] = None) -> 'LogStore':
        """Create appropriate log store for this agent."""
        pass

    @abstractmethod
    def parse_content(self, content: str, log_uri: str) -> 'Chat':
        """Parse raw log content into Chat object."""
        pass



    def validate_log(self, log_path: Path) -> bool:
        """Validate log file format (optional override)."""
        return log_path.exists() and log_path.suffix in self.config.log_extensions


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
        """Extract chat from specific or live log.

        Args:
            log_uri: Log URI (None for live log)

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
        config = self.config
        return {
            "name": config.name,
            "display_name": config.display_name,
            "log_extensions": config.log_extensions,
            "requires_session_id": config.requires_session_id,
            "metadata": config.metadata or {}
        }


    def __repr__(self) -> str:
        """String representation of agent."""
        return f"{self.__class__.__name__}(name='{self.config.name}', location='{self.location}')"


# Factory functions for creating agent instances
def claude(location: Optional[str] = None):
    """Create a Claude Code agent.
    
    Args:
        location: Optional workspace location for logs
        
    Returns:
        ClaudeCodeAgent instance
    """
    from ..agents.claude.claude_code import ClaudeCodeAgent
    return ClaudeCodeAgent(location=location)


def gemini(location: Optional[str] = None):
    """Create a Gemini CLI agent.
    
    Args:
        location: Optional workspace location for logs
        
    Returns:
        GeminiCliAgent instance
    """
    from ..agents.gemini.gemini_cli import GeminiCliAgent
    return GeminiCliAgent(location=location)


def qwen(location: Optional[str] = None):
    """Create a Qwen Code agent.
    
    Args:
        location: Optional workspace location for logs
        
    Returns:
        QwenCodeAgent instance
    """
    from ..agents.qwen.qwen_code import QwenCodeAgent
    return QwenCodeAgent(location=location)


def cligent(agent_type: str = "claude", location: Optional[str] = None):
    """Create an agent for the specified type.
    
    Args:
        agent_type: Agent type ("claude", "gemini", "qwen")
        location: Optional workspace location for logs
        
    Returns:
        Appropriate agent instance
        
    Raises:
        ValueError: If agent_type is not supported
    """
    agents = {
        "claude": claude,
        "claude-code": claude,
        "gemini": gemini,
        "gemini-cli": gemini,
        "qwen": qwen,
        "qwen-code": qwen,
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unsupported agent type: {agent_type}. Supported: {list(agents.keys())}")
    
    return agents[agent_type](location=location)