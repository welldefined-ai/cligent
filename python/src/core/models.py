"""Core data models for the chat parser."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


def _strip_ansi_codes(text: str) -> str:
    """Strip ANSI escape codes from text to ensure YAML compatibility.
    
    Args:
        text: Text that may contain ANSI escape codes
        
    Returns:
        Text with ANSI escape codes removed
    """
    # ANSI escape sequence pattern: ESC[ followed by parameter bytes and final byte
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    return ansi_pattern.sub('', text)


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
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """String representation for display."""
        timestamp_str = f" [{self.timestamp.strftime('%H:%M:%S')}]" if self.timestamp else ""
        return f"{self.role.value.upper()}{timestamp_str}: {self.content}"


@dataclass
class Chat:
    """A collection of messages."""
    
    messages: List[Message] = field(default_factory=list)
    
    def add(self, message: Message) -> None:
        """Include a message in the chat."""
        self.messages.append(message)
    
    def remove(self, message: Message) -> None:
        """Exclude a message from the chat."""
        if message in self.messages:
            self.messages.remove(message)
    
    def merge(self, other: 'Chat') -> 'Chat':
        """Combine with another chat."""
        merged_messages = self.messages + other.messages
        # Sort by timestamp if available
        merged_messages.sort(key=lambda m: m.timestamp or datetime.min)
        return Chat(messages=merged_messages)
    
    def export(self) -> str:
        """Output as Tigs YAML format with human-readable content blocks."""
        from datetime import datetime
        
        # Build the YAML manually for better control over formatting
        lines = []
        lines.append("schema: tigs.chat/v1")
        lines.append("messages:")
        
        for message in self.messages:
            lines.append(f"- role: {message.role.value}")
            
            # Always use literal block style for content
            lines.append("  content: |")
            # Strip ANSI codes to ensure YAML compatibility
            clean_content = _strip_ansi_codes(message.content)
            # Split content by any line ending style (cross-platform)
            content_lines = clean_content.splitlines()
            if content_lines:
                for content_line in content_lines:
                    lines.append(f"    {content_line}")
            else:
                # Handle empty content
                lines.append(f"    {clean_content}")
            
            # Add timestamp if available
            if message.timestamp:
                lines.append(f"  timestamp: '{message.timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')}'")
            
            # Add model from metadata if available
            if "model" in message.metadata:
                lines.append(f"  model: {message.metadata['model']}")
        
        return '\n'.join(lines)


@dataclass
class ErrorReport:
    """Information on a parsing failure."""
    
    error: str
    log: str
    location: Optional[str] = None
    recoverable: bool = True
    
    def __str__(self) -> str:
        """Format error for display."""
        location_str = f" at {self.location}" if self.location else ""
        recovery_str = " (recoverable)" if self.recoverable else " (fatal)"
        return f"Error{location_str}: {self.error}{recovery_str}\nLog snippet: {self.log}"


class LogStore(ABC):
    """Manager for an agent's session logs."""
    
    def __init__(self, agent: str, location: str = None):
        """Initialize log store for a specific agent.
        
        Args:
            agent: Name of the agent (e.g., "claude-code")
            location: Location of the agent's logs (optional, implementation-specific)
        """
        self.agent = agent
        self.location = location
    
    @abstractmethod
    def list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available session logs for the agent.
        
        Returns:
            List of (log_uri, metadata) tuples with implementation-specific metadata.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get(self, log_uri: str) -> str:
        """Retrieve raw content of a specific log.
        
        Args:
            log_uri: Log URI
            
        Returns:
            Raw log content as string
        """
        raise NotImplementedError
    
    @abstractmethod
    def live(self) -> Optional[str]:
        """Get URI of currently active log.
        
        Returns:
            Log URI for active log or None if no active log
        """
        raise NotImplementedError