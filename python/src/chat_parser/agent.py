from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, AsyncIterator
from pathlib import Path
from .store import LogStore
from .models import Chat
from .task_models import TaskResult, TaskUpdate, TaskConfig

@dataclass
class AgentConfig:
    """Agent configuration and metadata."""
    name: str
    display_name: str
    log_extensions: List[str]  # [".jsonl", ".log"]
    default_log_dir: Optional[Path] = None
    requires_session_id: bool = True
    supports_execution: bool = True  # Whether agent supports task execution
    metadata: Dict[str, Any] = None

class AgentBackend(ABC):
    """Abstract base class for all agent implementations."""

    @property
    @abstractmethod
    def config(self) -> AgentConfig:
        """Agent configuration and metadata."""
        pass

    @abstractmethod
    def create_store(self, location: Optional[str] = None) -> 'LogStore':
        """Create appropriate log store for this agent."""
        pass

    @abstractmethod
    def parse_content(self, content: str, log_uri: str, store: 'LogStore') -> 'Chat':
        """Parse raw log content into Chat object."""
        pass

    @abstractmethod
    def detect_agent(self, log_path: Path) -> bool:
        """Detect if a log file belongs to this agent."""
        pass

    def validate_log(self, log_path: Path) -> bool:
        """Validate log file format (optional override)."""
        return log_path.exists() and log_path.suffix in self.config.log_extensions

    # New task execution methods
    @abstractmethod
    async def execute_task(self, task: str, config: TaskConfig = None) -> TaskResult:
        """Execute a task and return the result.
        
        Args:
            task: The task description/instruction
            config: Task execution configuration
            
        Returns:
            TaskResult with execution outcome
        """
        pass

    @abstractmethod
    async def execute_task_stream(self, task: str, config: TaskConfig = None) -> AsyncIterator[TaskUpdate]:
        """Execute a task with streaming updates.
        
        Args:
            task: The task description/instruction  
            config: Task execution configuration
            
        Yields:
            TaskUpdate objects with execution progress
        """
        pass

    def supports_task_execution(self) -> bool:
        """Check if this agent supports task execution."""
        return self.config.supports_execution