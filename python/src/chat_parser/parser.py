"""Main parser interface for the chat parser library."""

from typing import List, Optional, Union, Dict, Tuple, Any

from .models import Chat, Message
from .store import LogStore
from .errors import ErrorCollector
from pathlib import Path
from .agent import AgentBackend

class ChatParser:
    """Main interface for parsing and composing agent chats."""

    def __init__(self, agent_name: str = None, location: str = None):
        """Initialize parser for a specific agent.

        Args:
           agent_name: Name of the agent (None for auto-detection)
            location: Optional custom location for logs
        """
        self.location = location
        self.error_collector = ErrorCollector()
        self.selected_messages: List[Message] = []
        self._chat_cache: Dict[str, Chat] = {}

        if not agent_name:
            agent_name = "claude-code"  # Default

        self.agent_name = agent_name
        self.agent = self._create_agent(agent_name)
        self.store = self.agent.create_store(location)

    def _create_agent(self, agent_name: str) -> AgentBackend:
        """Create appropriate agent backend."""
        from .registry import registry

        agent_class = registry.get_agent(agent_name)
        if not agent_class:
            available = [config.name for config in registry.list_agents()]
            raise ValueError(f"Unsupported agent: {agent_name}. Available: {available}")

        return agent_class()

    def _parse_content(self, content: str, log_uri: str) -> Chat:
        """Parse raw log content using agent backend."""
        return self.agent.parse_content(content, log_uri, self.store)

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
            return self._parse_content(content, log_uri)
        else:
            live_uri = self.store.live()
            if live_uri is None:
                return None
            content = self.store.get(live_uri)
            return self._parse_content(content, live_uri)

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


