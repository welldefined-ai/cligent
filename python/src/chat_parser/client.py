"""Unified client interface for AI agent orchestration."""

import asyncio
import uuid
from typing import AsyncIterator, Optional, Dict, Any, List
from pathlib import Path

from .registry import registry
from .agent import AgentBackend
from .parser import ChatParser
from .models import Chat
from .task_models import TaskResult, TaskUpdate, TaskConfig, TaskStatus
from .errors import ChatParserError


class CligentClient:
    """Unified client for AI agent task execution and log parsing."""

    def __init__(self, agent_name: str = "claude-code", location: Optional[str] = None):
        """Initialize client with specified agent.
        
        Args:
            agent_name: Name or alias of the agent to use
            location: Optional workspace location
        """
        self.agent_name = agent_name
        self.location = location
        self._agent: Optional[AgentBackend] = None
        self._parser: Optional[ChatParser] = None
        
        # Initialize agent
        self._init_agent()

    def _init_agent(self) -> None:
        """Initialize the agent backend."""
        agent_class = registry.get_agent(self.agent_name)
        if not agent_class:
            available = [config.name for config in registry.list_agents()]
            raise ValueError(f"Unknown agent: {self.agent_name}. Available: {available}")
        
        self._agent = agent_class()
        
        # Initialize parser for log operations
        self._parser = ChatParser(self.agent_name, self.location)

    @property
    def agent(self) -> AgentBackend:
        """Get the current agent backend."""
        return self._agent

    def switch_agent(self, agent_name: str) -> None:
        """Switch to a different agent.
        
        Args:
            agent_name: Name or alias of the new agent
        """
        self.agent_name = agent_name
        self._init_agent()

    def list_available_agents(self) -> List[str]:
        """List all available agents."""
        return [config.name for config in registry.list_agents()]

    def supports_execution(self) -> bool:
        """Check if current agent supports task execution."""
        return self._agent.supports_task_execution()

    # Task Execution Methods
    async def execute(self, task: str, **kwargs) -> TaskResult:
        """Execute a task using the current agent.
        
        Args:
            task: Task description or instruction
            **kwargs: Additional configuration options
            
        Returns:
            TaskResult with execution outcome
            
        Raises:
            ValueError: If agent doesn't support task execution
            ChatParserError: If execution fails
        """
        if not self.supports_execution():
            raise ValueError(f"Agent {self.agent_name} does not support task execution")

        # Build task configuration
        config = TaskConfig(**kwargs)
        
        try:
            return await self._agent.execute_task(task, config)
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
            ValueError: If agent doesn't support task execution
            ChatParserError: If execution fails
        """
        if not self.supports_execution():
            raise ValueError(f"Agent {self.agent_name} does not support task execution")

        # Build task configuration
        config = TaskConfig(stream=True, **kwargs)
        
        try:
            async for update in self._agent.execute_task_stream(task, config):
                yield update
        except Exception as e:
            raise ChatParserError(f"Streaming execution failed: {e}") from e

    # Log Parsing Methods (existing functionality)
    def list_logs(self) -> List[tuple]:
        """List available logs for current agent."""
        return self._parser.list_logs()

    def parse_logs(self, log_uri: str = None) -> Chat:
        """Parse logs into Chat object.
        
        Args:
            log_uri: Specific log URI (None for latest)
            
        Returns:
            Parsed Chat object
        """
        return self._parser.parse(log_uri)

    def compose_yaml(self, *args) -> str:
        """Compose selected messages into YAML format.
        
        Args:
            *args: Messages or chats to compose
            
        Returns:
            YAML formatted string
        """
        return self._parser.compose(*args)

    def select_messages(self, log_uri: str, indices: List[int] = None) -> None:
        """Select messages for composition.
        
        Args:
            log_uri: Log URI
            indices: Message indices (None for all)
        """
        self._parser.select(log_uri, indices)

    def unselect_messages(self, log_uri: str, indices: List[int] = None) -> None:
        """Unselect messages.
        
        Args:
            log_uri: Log URI
            indices: Message indices (None for all from this log)
        """
        self._parser.unselect(log_uri, indices)

    def clear_selection(self) -> None:
        """Clear current message selection."""
        self._parser.clear_selection()

    def get_errors(self) -> Optional[str]:
        """Get any parsing errors."""
        return self._parser.get_errors()

    # Convenience methods
    async def task_and_parse(self, task: str, **kwargs) -> tuple[TaskResult, Chat]:
        """Execute task and immediately parse the logs.
        
        Args:
            task: Task to execute
            **kwargs: Task configuration
            
        Returns:
            Tuple of (TaskResult, Chat with parsed logs)
        """
        # Execute task
        result = await self.execute(task, **kwargs)
        
        # Small delay to ensure logs are written
        await asyncio.sleep(0.5)
        
        # Parse the latest logs
        chat = self.parse_logs()
        
        return result, chat

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about current agent."""
        config = self._agent.config
        return {
            "name": config.name,
            "display_name": config.display_name,
            "supports_execution": config.supports_execution,
            "log_extensions": config.log_extensions,
            "requires_session_id": config.requires_session_id,
            "metadata": config.metadata or {}
        }

    def __repr__(self) -> str:
        """String representation of client."""
        return f"CligentClient(agent='{self.agent_name}', location='{self.location}')"


# Factory functions for convenience
def cligent(agent_name: str = "claude-code", location: Optional[str] = None) -> CligentClient:
    """Create a CligentClient instance.
    
    Args:
        agent_name: Agent name or alias (default: "claude-code")
        location: Optional workspace location
        
    Returns:
        CligentClient instance
    """
    return CligentClient(agent_name, location)


def claude(location: Optional[str] = None) -> CligentClient:
    """Create a Claude Code client."""
    return CligentClient("claude-code", location)


def gemini(location: Optional[str] = None) -> CligentClient:
    """Create a Gemini CLI client."""
    return CligentClient("gemini-cli", location)


def qwen(location: Optional[str] = None) -> CligentClient:
    """Create a Qwen Code client."""
    return CligentClient("qwen-code", location)