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
from .registry import registry
from .agents import ClaudeCodeAgent, GeminiCliAgent, QwenCodeAgent

# Unified agent interface - AgentBackend is the main entry point
from .client import cligent, claude, gemini, qwen, list_available_agents
from .execution import TaskResult, TaskUpdate, TaskConfig, TaskStatus, UpdateType
from .core import AgentBackend, AgentConfig


__all__ = [
    # Core models
    "Chat",
    "Message", 
    "Role",
    "ErrorReport",
    
    # Stores (for advanced usage)
    "LogStore",
    
    # Main interface - AgentBackend only
    "cligent",
    "claude", 
    "gemini",
    "qwen",
    "list_available_agents",
    
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