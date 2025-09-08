from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, AsyncIterator, TYPE_CHECKING
from pathlib import Path
import asyncio
from .models import Chat, Message
from ..execution.task_models import TaskResult, TaskUpdate, TaskConfig
from .errors import ErrorCollector, ChatParserError

if TYPE_CHECKING:
    from .models import LogStore
    from ..execution.executor import MockExecutor

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
        # Executor will be initialized by subclasses
        self._executor: Optional['MockExecutor'] = None

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
    def parse_content(self, content: str, log_uri: str, store: 'LogStore') -> 'Chat':
        """Parse raw log content into Chat object."""
        pass


    def validate_log(self, log_path: Path) -> bool:
        """Validate log file format (optional override)."""
        return log_path.exists() and log_path.suffix in self.config.log_extensions

    # Task execution methods with default implementation
    async def execute_task(self, task: str, config: TaskConfig = None) -> TaskResult:
        """Execute a task and return the result.
        
        Args:
            task: The task description/instruction
            config: Task execution configuration
            
        Returns:
            TaskResult with execution outcome
        """
        if config is None:
            config = TaskConfig()
        return await self._executor.execute_task(task, config)

    async def execute_task_stream(self, task: str, config: TaskConfig = None) -> AsyncIterator[TaskUpdate]:
        """Execute a task with streaming updates.
        
        Args:
            task: The task description/instruction  
            config: Task execution configuration
            
        Yields:
            TaskUpdate objects with execution progress
        """
        if config is None:
            config = TaskConfig(stream=True)
        async for update in self._executor.execute_task_stream(task, config):
            yield update


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
            return self.parse_content(content, log_uri, self.store)
        else:
            live_uri = self.store.live()
            if live_uri is None:
                return None
            content = self.store.get(live_uri)
            return self.parse_content(content, live_uri, self.store)

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

    # High-level Task Execution Methods
    async def execute(self, task: str, **kwargs) -> TaskResult:
        """Execute a task using this agent.
        
        Args:
            task: Task description or instruction
            **kwargs: Additional configuration options
            
        Returns:
            TaskResult with execution outcome
            
        Raises:
            ChatParserError: If execution fails
        """
        # Build task configuration
        config = TaskConfig(**kwargs)
        
        try:
            return await self.execute_task(task, config)
        except Exception as e:
            raise ChatParserError(f"Task execution failed: {e}") from e

    async def execute_stream(self, task: str, **kwargs) -> AsyncIterator[TaskUpdate]:
        """Execute a task with streaming updates.
        
        Args:
            task: Task description or instruction
            **kwargs: Additional configuration options
            
        Yields:
            TaskUpdate objects with execution progress
            
        Raises:
            ChatParserError: If execution fails
        """
        # Build task configuration
        config = TaskConfig(stream=True, **kwargs)
        
        try:
            async for update in self.execute_task_stream(task, config):
                yield update
        except Exception as e:
            raise ChatParserError(f"Streaming execution failed: {e}") from e


    def __repr__(self) -> str:
        """String representation of agent."""
        return f"{self.__class__.__name__}(name='{self.config.name}', location='{self.location}')"