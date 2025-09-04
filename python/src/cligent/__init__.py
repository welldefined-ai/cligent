"""Cligent - Unified SDK for CLI agent orchestration."""

from .core import Chat, Message, Role, ErrorReport
from .parsers import ChatParser, LogStore
from .core import (
    ChatParserError,
    ParseError, 
    LogAccessError,
    LogCorruptionError,
    InvalidFormatError,
)
from .registry import registry
from .agents import ClaudeCodeAgent, GeminiCliAgent, QwenCodeAgent

# Unified client interface
from .client import CligentClient, cligent, claude, gemini, qwen
from .execution import TaskResult, TaskUpdate, TaskConfig, TaskStatus, UpdateType
from .core import AgentBackend, AgentConfig


__all__ = [
    # Core models
    "Chat",
    "Message", 
    "Role",
    "ErrorReport",
    
    # Main interface
    "ChatParser",
    
    # Stores
    "LogStore",
    
    # Unified client interface
    "CligentClient",
    "cligent",
    "claude", 
    "gemini",
    "qwen",
    
    # Task execution
    "TaskResult",
    "TaskUpdate", 
    "TaskConfig",
    "TaskStatus",
    "UpdateType",
    
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


# Auto-register all agents
registry.register(ClaudeCodeAgent, aliases=["claude"])
registry.register(GeminiCliAgent, aliases=["gemini", "gemini-cli"])
registry.register(QwenCodeAgent, aliases=["qwen", "qwen-code"])