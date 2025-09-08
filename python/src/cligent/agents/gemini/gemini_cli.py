"""Gemini CLI specific implementation for parsing JSONL logs."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ...core.models import Message, Chat, ErrorReport, Role
from ...core.models import LogStore
from ...core.agent import AgentBackend, AgentConfig
from ...execution.task_models import TaskResult, TaskUpdate, TaskConfig
from ...execution.executor import GeminiExecutor, MockExecutor
from typing import AsyncIterator


@dataclass
class GeminiRecord:
    """A single JSON line in a Gemini CLI JSONL log file."""

    type: str  # user, assistant, system, tool_use, tool_result
    timestamp: Optional[str] = None
    content: str = ""
    role: Optional[str] = None
    session_id: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, json_string: str) -> 'GeminiRecord':
        """Parse a JSON string into a GeminiRecord."""
        try:
            data = json.loads(json_string)
            
            # Flexible parsing for different Gemini CLI log formats
            record_type = data.get('type', 'unknown')
            content = data.get('content', data.get('text', data.get('message', '')))
            role = data.get('role', data.get('sender', ''))
            timestamp = data.get('timestamp', data.get('time', data.get('created_at', '')))
            session_id = data.get('session_id', data.get('sessionId', data.get('conversation_id', '')))
            
            return cls(
                type=record_type,
                content=content,
                role=role,
                timestamp=timestamp,
                session_id=session_id,
                raw_data=data
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid JSON record: {e}")

    def extract_message(self) -> Optional[Message]:
        """Get a Message from this record, if applicable."""
        if not self.is_message():
            return None

        # Map Gemini CLI roles to our Role enum
        role_mapping = {
            'user': Role.USER,
            'assistant': Role.ASSISTANT,
            'model': Role.ASSISTANT,  # Gemini often uses 'model' instead of 'assistant'
            'system': Role.SYSTEM,
            'human': Role.USER,  # Alternative user role
            'ai': Role.ASSISTANT,  # Alternative assistant role
        }

        role = role_mapping.get(self.role.lower() if self.role else '', Role.ASSISTANT)

        # Handle different content formats
        content = self.content
        if isinstance(content, list):
            # Extract text from structured content
            content_parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get('text', item.get('content', ''))
                elif isinstance(item, str):
                    text = item
                else:
                    text = str(item)
                
                if text.strip():
                    content_parts.append(text.strip())
            
            content = '\n'.join(content_parts)
        elif isinstance(content, dict):
            # Extract text from dict format
            content = content.get('text', content.get('content', str(content)))
        
        content = str(content).strip()
        
        # Skip empty messages
        if not content:
            return None

        # Parse timestamp
        timestamp = None
        if self.timestamp:
            try:
                # Handle various timestamp formats
                if self.timestamp.endswith('Z'):
                    timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
                elif '+' in self.timestamp or '-' in self.timestamp[-6:]:
                    timestamp = datetime.fromisoformat(self.timestamp)
                else:
                    # Try parsing as Unix timestamp
                    timestamp = datetime.fromtimestamp(float(self.timestamp))
            except (ValueError, AttributeError):
                pass

        metadata = {
            'type': self.type,
            'session_id': self.session_id,
            'raw_data': self.raw_data
        }

        return Message(
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata
        )

    def is_message(self) -> bool:
        """Check if this record represents a message."""
        message_types = {
            'user', 'assistant', 'system', 'human', 'ai', 'model', 'message'
        }
        return (
            self.type.lower() in message_types or
            (self.role and self.role.lower() in {'user', 'assistant', 'system', 'human', 'ai', 'model'})
        )


@dataclass
class GeminiSession:
    """A complete JSONL log file representing a Gemini CLI chat."""

    file_path: Path
    session_id: Optional[str] = None
    records: List[GeminiRecord] = field(default_factory=list)

    def load(self) -> None:
        """Read and parse all Records from the file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.file_path}")

        self.records.clear()
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = GeminiRecord.load(line)
                        self.records.append(record)

                        # Extract session metadata
                        if not self.session_id and record.session_id:
                            self.session_id = record.session_id

                    except ValueError as e:
                        # Log parsing error but continue
                        print(f"Warning: Skipped invalid record at line {line_num}: {e}")
                        continue

        except (IOError, OSError) as e:
            raise FileNotFoundError(f"Cannot read log file: {e}")

    def to_chat(self) -> Chat:
        """Convert session records to a Chat object."""
        messages = []

        for record in self.records:
            message = record.extract_message()
            if message:
                messages.append(message)

        return Chat(messages=messages)


class GeminiStore(LogStore):
    """Gemini CLI log store implementation."""

    def __init__(self):
        """Initialize with base path for Gemini CLI logs.
        
        Note: Gemini CLI stores logs globally in ~/.gemini/, not per-project.
        """
        # Gemini CLI typically stores logs in ~/.gemini/tmp or ~/.gemini/logs
        gemini_base = Path.home() / ".gemini"
        self._logs_dir = gemini_base / "tmp"
        
        # Fallback to logs directory if tmp doesn't exist
        if not self._logs_dir.exists():
            self._logs_dir = gemini_base / "logs"
            
        # If neither exists, use sessions subdirectory
        if not self._logs_dir.exists():
            self._logs_dir = gemini_base / "logs" / "sessions"

        # Use current directory for LogStore base class compatibility
        super().__init__("gemini-cli", str(Path.cwd()))
        self.session_pattern = "*.jsonl"  # Pattern for session file names

    def list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available logs."""
        logs = []

        try:
            if not self._logs_dir.exists():
                return logs

            # Scan for JSONL files
            for log_file in self._logs_dir.glob(self.session_pattern):
                if log_file.is_file():
                    stat = log_file.stat()
                    session_id = log_file.stem
                    metadata = {
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "accessible": log_file.is_file() and os.access(log_file, os.R_OK)
                    }
                    logs.append((session_id, metadata))

        except (OSError, PermissionError):
            # Return empty list if we can't access the directory
            pass

        return logs

    def get(self, log_uri: str) -> str:
        """Retrieve raw content of a specific log.

        Args:
            log_uri: Either a session ID or full path to log file
        """
        # Handle both session IDs and full paths
        if "/" in log_uri or "\\" in log_uri:
            log_path = Path(log_uri)
        else:
            log_path = self._logs_dir / f"{log_uri}.jsonl"

        try:
            if not log_path.exists():
                raise FileNotFoundError(f"Log file not found: {log_uri}")

            with open(log_path, 'r', encoding='utf-8') as f:
                return f.read()

        except (OSError, PermissionError, UnicodeDecodeError) as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise IOError(f"Cannot read log file {log_uri}: {e}")

    def live(self) -> Optional[str]:
        """Get URI of currently active log (most recent)."""
        logs = self.list()
        if not logs:
            return None

        # Sort by modification time (most recent first)
        sorted_logs = sorted(logs, key=lambda x: x[1].get('modified', ''), reverse=True)
        return sorted_logs[0][0] if sorted_logs else None


class GeminiCliAgent(AgentBackend):
    """Gemini CLI agent implementation."""

    def __init__(self, location: Optional[str] = None, api_key: Optional[str] = None, use_mock: bool = False):
        """Initialize Gemini CLI agent.
        
        Args:
            location: Optional workspace location for logs
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
            use_mock: If True, use mock executor instead of real Gemini API
        """
        super().__init__(location)
        
        # Initialize executor - real Gemini API or mock
        if use_mock:
            self._executor = MockExecutor("gemini-cli")
        else:
            self._executor = GeminiExecutor(api_key)

    @property
    def config(self) -> AgentConfig:
        return AgentConfig(
            name="gemini-cli",
            display_name="Gemini CLI",
            log_extensions=[".jsonl", ".json"],
            requires_session_id=True,
            metadata={
                "log_format": "jsonl",
                "base_dir": "~/.gemini/",
                "supports_tools": True
            }
        )

    def _create_store(self, location: Optional[str] = None) -> LogStore:
        return GeminiStore()

    def parse_content(self, content: str, log_uri: str, store: LogStore) -> Chat:
        # Handle both session IDs and full paths
        if "/" in log_uri or "\\" in log_uri:
            file_path = Path(log_uri)
        else:
            file_path = store._logs_dir / f"{log_uri}.jsonl"

        session = GeminiSession(file_path=file_path)
        session.load()
        return session.to_chat()



