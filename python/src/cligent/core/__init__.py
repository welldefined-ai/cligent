"""Core models and base classes for Cligent."""

from .models import Chat, Message, Role, ErrorReport, LogStore
from .errors import (
    ChatParserError,
    ParseError, 
    LogAccessError,
    LogCorruptionError,
    InvalidFormatError,
)
from .agent import AgentBackend, AgentConfig, claude, gemini, qwen, cligent

__all__ = [
    # Core models
    "Chat",
    "Message", 
    "Role",
    "ErrorReport",
    "LogStore",
    
    # Agent framework
    "AgentBackend",
    "AgentConfig",
    
    # Factory functions
    "claude",
    "gemini", 
    "qwen",
    "cligent",
    
    # Errors
    "ChatParserError",
    "ParseError",
    "LogAccessError", 
    "LogCorruptionError",
    "InvalidFormatError",
]