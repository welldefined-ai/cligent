"""Core data models for the chat parser."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class Role(Enum):
    """Message participant roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single communication unit."""
    
    role: Role
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        raise NotImplementedError
    
    def __str__(self) -> str:
        """String representation for display."""
        raise NotImplementedError


@dataclass
class Chat:
    """A collection of messages."""
    
    messages: List[Message] = field(default_factory=list)
    
    def add(self, message: Message) -> None:
        """Include a message in the chat."""
        raise NotImplementedError
    
    def remove(self, message: Message) -> None:
        """Exclude a message from the chat."""
        raise NotImplementedError
    
    def parse(self, log_content: str) -> None:
        """Extract chat from a log."""
        raise NotImplementedError
    
    def merge(self, other: 'Chat') -> 'Chat':
        """Combine with another chat."""
        raise NotImplementedError
    
    def export(self) -> str:
        """Output as Tigs text."""
        raise NotImplementedError


@dataclass
class ErrorReport:
    """Information on a parsing failure."""
    
    error: str
    log: str
    location: Optional[str] = None
    recoverable: bool = True
    
    def __str__(self) -> str:
        """Format error for display."""
        raise NotImplementedError