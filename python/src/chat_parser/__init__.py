"""Chat Parser subpackage for parsing AI agent conversation logs."""

from .models import Chat, Message, Role, ErrorReport
from .store import LogStore
from .parser import ChatParser
from .errors import (
    ChatParserError,
    ParseError, 
    LogAccessError,
    LogCorruptionError,
    InvalidFormatError,
)
from .registry import registry
from .claude.claude_code import ClaudeCodeAgent
from .gemini.gemini_cli import GeminiCliAgent


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
    
    
    # Errors
    "ChatParserError",
    "ParseError",
    "LogAccessError", 
    "LogCorruptionError",
    "InvalidFormatError",
]


# 自动注册所有agent
registry.register(ClaudeCodeAgent, aliases=["claude"])
registry.register(GeminiCliAgent, aliases=["gemini", "gemini-cli"])