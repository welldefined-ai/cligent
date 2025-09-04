"""Task execution models for unified agent orchestration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, AsyncIterator
from enum import Enum


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UpdateType(Enum):
    """Types of task updates during streaming execution."""
    STATUS = "status"
    PROGRESS = "progress"
    OUTPUT = "output"
    ERROR = "error"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


@dataclass
class TaskResult:
    """Result of task execution."""
    
    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "logs": self.logs
        }


@dataclass
class TaskUpdate:
    """Streaming update during task execution."""
    
    task_id: str
    update_type: UpdateType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "type": self.update_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TaskConfig:
    """Configuration for task execution."""
    
    timeout: Optional[int] = None  # seconds
    stream: bool = False
    save_logs: bool = True
    workspace: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timeout": self.timeout,
            "stream": self.stream,
            "save_logs": self.save_logs,
            "workspace": self.workspace,
            "environment": self.environment,
            "tools": self.tools
        }