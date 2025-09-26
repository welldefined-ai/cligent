"""Gemini CLI specific implementation for parsing session logs."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ...core.models import Message, Chat, ErrorReport, Role, LogStore, Record, LogFile, ProviderConfig
from ...cligent import Cligent


# Gemini-specific configuration
GEMINI_CONFIG = ProviderConfig(
    name="gemini-cli",
    display_name="Gemini CLI",
    home_dir=".gemini",
    role_mappings={
        'user': Role.USER,
        'assistant': Role.ASSISTANT,
        'model': Role.ASSISTANT,
        'system': Role.SYSTEM,
        'human': Role.USER,
        'ai': Role.ASSISTANT,
    },
    log_patterns=["*.json"]
)


@dataclass
class GeminiRecord(Record):
    """A single message record in a Gemini CLI JSON log file."""

    role: str = ""
    timestamp: Optional[str] = None
    content: str = ""
    session_id: Optional[str] = None

    @classmethod
    def load(cls, json_string: str) -> 'GeminiRecord':
        """Parse a JSON string into a GeminiRecord."""
        return super().load(json_string, GEMINI_CONFIG)

    def _post_load(self, data: Dict[str, Any]) -> None:
        """Extract Gemini-specific fields."""
        # Handle Google conversation format (checkpoint files)
        if 'role' in data and 'parts' in data:
            # Google format: {"role": "user", "parts": [{"text": "..."}]}
            self.role = data.get('role', '')
            self.content = self._extract_parts_content(data.get('parts', []))
            self.timestamp = ''  # No timestamp in Google format
            self.session_id = ''
        else:
            # Legacy format for main logs.json - type field becomes role
            self.role = data.get('type', data.get('role', data.get('sender', 'unknown')))
            self.content = data.get('content', data.get('text', data.get('message', '')))
            self.timestamp = data.get('timestamp', data.get('time', data.get('created_at', '')))
            self.session_id = data.get('session_id', data.get('sessionId', data.get('conversation_id', '')))

    def get_role(self) -> str:
        return self.role

    def get_content(self) -> str:
        return self.content

    def get_timestamp(self) -> Optional[str]:
        return self.timestamp

    def _extract_parts_content(self, parts: List[Dict[str, Any]]) -> str:
        """Extract text content from Google conversation parts array."""
        content_parts = []
        for part in parts:
            if isinstance(part, dict):
                # Extract text from parts
                if 'text' in part:
                    text = part['text'].strip()
                    if text:
                        content_parts.append(text)
                # Skip function calls and other non-text parts for now

        return '\n'.join(content_parts)

    def extract_message(self, log_uri: str = "") -> Optional[Message]:
        """Get a Message from this record, if applicable."""
        # Use base class for message extraction
        message = super().extract_message(log_uri)
        if message:
            # Add Gemini-specific session_id
            message.session_id = self.session_id
        return message



@dataclass
class GeminiLogFile(LogFile):
    """A complete JSONL log file representing a Gemini CLI chat."""

    def __init__(self, file_path: Path):
        super().__init__(file_path, GEMINI_CONFIG)

    def _create_record(self, json_string: str) -> Record:
        """Create a Gemini Record instance."""
        return GeminiRecord.load(json_string)


class GeminiLogStore(LogStore):
    """Gemini CLI log store implementation."""

    def __init__(self):
        """Initialize with base path for Gemini CLI logs.

        Note: Gemini CLI stores logs globally in ~/.gemini/, not per-project.
        """
        super().__init__(GEMINI_CONFIG)


    def _resolve_log_path(self, log_uri: str) -> Path:
        """Resolve log URI to file path for Gemini's structure."""
        # Handle new <uuid>/<file_name> format
        if "/" in log_uri and not log_uri.startswith("/"):
            # Format: <uuid>/<file_name>
            parts = log_uri.split("/", 1)
            if len(parts) == 2:
                session_id, file_name = parts
                return self._logs_dir / session_id / file_name
            else:
                # Fallback to old format
                return Path(log_uri)
        elif "\\" in log_uri or log_uri.startswith("/"):
            # Full path format
            return Path(log_uri)
        else:
            # Legacy: just session ID, assume logs.json
            return self._logs_dir / log_uri / "logs.json"



class GeminiCligent(Cligent):
    """Gemini CLI agent implementation."""

    def __init__(self):
        """Initialize Gemini CLI agent."""
        super().__init__()

    @property
    def name(self) -> str:
        return "gemini-cli"
        
    @property
    def display_name(self) -> str:
        return "Gemini CLI"

    def _create_store(self) -> LogStore:
        return GeminiLogStore()

    def parse_content(self, content: str, log_uri: str) -> Chat:
        # Handle new <uuid>/<file_name> format
        if "/" in log_uri and not log_uri.startswith("/"):
            # Format: <uuid>/<file_name>
            parts = log_uri.split("/", 1)
            if len(parts) == 2:
                session_id, file_name = parts
                file_path = self.store._logs_dir / session_id / file_name
            else:
                # Fallback to old format
                file_path = Path(log_uri)
        elif "\\" in log_uri or log_uri.startswith("/"):
            # Full path format
            file_path = Path(log_uri)
        else:
            # Legacy: just session ID, assume logs.json
            file_path = self.store._logs_dir / log_uri / "logs.json"

        log_file = GeminiLogFile(file_path=file_path)
        log_file.load()
        return log_file.to_chat(log_uri=log_uri)



