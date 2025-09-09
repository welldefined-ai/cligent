"""Cligent - Unified SDK for CLI agent orchestration."""

__version__ = "0.2.0"

from .core import Chat, Message, Role, ErrorReport
from .core import LogStore
from .core import (
    ChatParserError,
    ParseError, 
    LogAccessError,
    LogCorruptionError,
    InvalidFormatError,
)
from .agents import ClaudeCodeAgent, GeminiCliAgent, QwenCodeAgent
from .core import AgentBackend, AgentConfig, claude, gemini, qwen, cligent

# Backwards compatibility alias
ChatParser = cligent


__all__ = [
    # Core models
    "Chat",
    "Message", 
    "Role",
    "ErrorReport",
    
    # Stores (for advanced usage)
    "LogStore",
    
    # Agent implementations
    "ClaudeCodeAgent",
    "GeminiCliAgent", 
    "QwenCodeAgent",
    
    # Factory functions
    "claude",
    "gemini",
    "qwen", 
    "cligent",
    "ChatParser",  # Backwards compatibility
    
    # Agent framework
    "AgentBackend",
    "AgentConfig",
    
    # Errors
    "ChatParserError",
    "ParseError",
    "LogAccessError", 
    "LogCorruptionError",
    "InvalidFormatError",
]

