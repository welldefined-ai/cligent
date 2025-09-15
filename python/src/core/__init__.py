"""Core models and base classes for Cligent."""

from .models import Chat, Message, Role, ErrorReport, LogStore
from .base import BaseRecord, BaseLogFile, BaseStore, ProviderConfig
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
    "LogStore",

    # Base classes
    "BaseRecord",
    "BaseLogFile",
    "BaseStore",
    "ProviderConfig",

    # Errors
    "ChatParserError",
    "ParseError",
    "LogAccessError",
    "LogCorruptionError",
    "InvalidFormatError",
]