"""Qwen Code specific implementation for parsing JSONL session logs."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ...core.models import Message, Chat, ErrorReport, Role, LogStore, BaseRecord, BaseLogFile, ProviderConfig
from ...cligent import Cligent


# Qwen-specific configuration
QWEN_CONFIG = ProviderConfig(
    name="qwen-code",
    display_name="Qwen Code",
    home_dir=".qwen",
    role_mappings={
        'user': Role.USER,
        'assistant': Role.ASSISTANT,
        'model': Role.ASSISTANT,
        'qwen': Role.ASSISTANT,
        'system': Role.SYSTEM,
        'human': Role.USER,
        'ai': Role.ASSISTANT,
    },
    log_patterns=["*.json", "*.jsonl"]
)


@dataclass
class QwenRecord(BaseRecord):
    """A single JSON record from a Qwen log file."""

    role: str = ""
    timestamp: Optional[str] = None
    content: str = ""
    session_id: Optional[str] = None
    checkpoint_tag: Optional[str] = None
    model: Optional[str] = None

    @classmethod
    def load(cls, json_string: str) -> 'QwenRecord':
        """Parse a JSON string into a QwenRecord."""
        return super().load(json_string, QWEN_CONFIG)

    def _post_load(self, data: Dict[str, Any]) -> None:
        """Extract Qwen-specific fields."""
        # Handle Google conversation format (checkpoint files)
        if 'role' in data and 'parts' in data:
            # Google format: {"role": "user", "parts": [{"text": "..."}]}
            self.role = data.get('role', '')
            self.content = self._extract_parts_content(data.get('parts', []))
            self.timestamp = ''  # No timestamp in Google format
            self.session_id = ''
            self.checkpoint_tag = ''
            self.model = ''
        else:
            # Legacy JSONL format - prioritize type/messageType fields for role
            self.role = data.get('type', data.get('messageType', data.get('role', data.get('sender', data.get('from', 'unknown')))))
            self.content = data.get('content', data.get('text', data.get('message', '')))
            self.timestamp = data.get('timestamp', data.get('time', data.get('created_at', '')))
            self.session_id = data.get('session_id', data.get('sessionId', data.get('conversationId', '')))
            self.checkpoint_tag = data.get('checkpoint_tag', data.get('checkpointTag', data.get('tag', '')))
            self.model = data.get('model', data.get('modelName', ''))

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

        return ''.join(content_parts)  # Join without separator for Qwen (parts are often single chars)

    def extract_message(self) -> Optional[Message]:
        """Get a Message from this record, if applicable."""
        # Use base class for message extraction
        message = super().extract_message()
        if message and message.metadata:
            # Add Qwen-specific metadata
            message.metadata.update({
                'session_id': self.session_id,
                'checkpoint_tag': self.checkpoint_tag,
                'model': self.model
            })
        return message



@dataclass
class QwenLogFile(BaseLogFile):
    """A complete JSONL log file representing a Qwen Code chat."""

    checkpoint_tags: List[str] = field(default_factory=list)

    def __init__(self, file_path: Path):
        super().__init__(file_path, QWEN_CONFIG)
        self.checkpoint_tags = []

    def _create_record(self, json_string: str) -> BaseRecord:
        """Create a Qwen Record instance."""
        return QwenRecord.load(json_string)

    def _extract_session_metadata(self, record: BaseRecord) -> None:
        """Extract Qwen-specific session metadata."""
        super()._extract_session_metadata(record)

        # Extract Qwen-specific metadata
        if isinstance(record, QwenRecord):
            if record.checkpoint_tag and record.checkpoint_tag not in self.checkpoint_tags:
                self.checkpoint_tags.append(record.checkpoint_tag)



class QwenLogStore(LogStore):
    """Qwen Code log store implementation."""

    def __init__(self):
        """Initialize with base path for Qwen Code logs.

        Note: Qwen Code stores logs globally in ~/.qwen/, not per-project.
        """
        super().__init__(QWEN_CONFIG)


    def _resolve_log_path(self, log_uri: str) -> Path:
        """Resolve log URI to file path for Qwen's structure."""
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
            # Legacy: just session ID, try JSON first, then JSONL
            json_path = self._logs_dir / log_uri / "logs.json"
            jsonl_path = self._logs_dir / f"{log_uri}.jsonl"

            if json_path.exists():
                return json_path
            elif jsonl_path.exists():
                return jsonl_path
            else:
                return json_path  # Will fail with proper error message



class QwenCligent(Cligent):
    """Qwen Code agent implementation."""

    def __init__(self):
        """Initialize Qwen Code agent."""
        super().__init__()

    @property
    def name(self) -> str:
        return "qwen-code"
        
    @property
    def display_name(self) -> str:
        return "Qwen Code"

    def _create_store(self) -> LogStore:
        return QwenLogStore()

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
            # Legacy: just session ID, try JSON first, then JSONL
            json_path = self.store._logs_dir / log_uri / "logs.json"
            jsonl_path = self.store._logs_dir / f"{log_uri}.jsonl"
            
            if json_path.exists():
                file_path = json_path
            elif jsonl_path.exists():
                file_path = jsonl_path
            else:
                file_path = json_path  # Will fail with proper error message

        log_file = QwenLogFile(file_path=file_path)
        log_file.load()
        return log_file.to_chat()



