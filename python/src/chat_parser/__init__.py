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