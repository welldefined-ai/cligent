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
    
    def parse(self, log_content: str) -> None:
        """Extract chat from a log."""
        # This is a placeholder - actual parsing would be agent-specific
        # In practice, this would delegate to agent-specific parsers
        lines = log_content.strip().split('\n')
        for line in lines:
            if line.strip():
                # Simple text-based parsing for demo
                role = Role.USER if line.startswith('User:') else Role.ASSISTANT
                content = line.split(':', 1)[1].strip() if ':' in line else line
                message = Message(role=role, content=content)
                self.messages.append(message)
    
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
            # Split content by any line ending style (cross-platform)
            content_lines = message.content.splitlines()
            if content_lines:
                for content_line in content_lines:
                    lines.append(f"    {content_line}")
            else:
                # Handle empty content
                lines.append(f"    {message.content}")
            
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