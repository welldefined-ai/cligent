"""Core data models for the chat parser."""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum


def _strip_ansi_codes(text: str) -> str:
    """Strip ANSI escape codes from text to ensure YAML compatibility.
    
    Args:
        text: Text that may contain ANSI escape codes
        
    Returns:
        Text with ANSI escape codes removed
    """
    # ANSI escape sequence pattern: ESC[ followed by parameter bytes and final byte
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    return ansi_pattern.sub('', text)


class Role(Enum):
    """Message participant roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ProviderConfig:
    """Configuration for provider-specific behavior."""

    name: str
    display_name: str
    home_dir: str  # e.g., ".claude", ".gemini", ".qwen"
    role_mappings: Dict[str, Role] = field(default_factory=dict)
    message_roles: Set[str] = field(default_factory=set)
    skip_roles: Set[str] = field(default_factory=set)
    log_patterns: List[str] = field(default_factory=list)  # e.g., ["*.jsonl", "*.json"]

    def __post_init__(self):
        """Set default values if not provided."""
        if not self.role_mappings:
            self.role_mappings = {
                'user': Role.USER,
                'assistant': Role.ASSISTANT,
                'model': Role.ASSISTANT,
                'system': Role.SYSTEM,
                'human': Role.USER,
                'ai': Role.ASSISTANT,
            }

        if not self.message_roles:
            self.message_roles = {
                'user', 'assistant', 'system', 'human', 'ai', 'model', 'message', 'unknown'
            }

        if not self.skip_roles:
            self.skip_roles = {'tool_use', 'tool_result', 'checkpoint'}

        if not self.log_patterns:
            self.log_patterns = ["*.json", "*.jsonl"]


@dataclass
class Message:
    """A single communication unit."""

    role: Role
    content: str
    provider: str
    log_uri: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "provider": self.provider,
            "raw_data": self.raw_data,
            "session_id": self.session_id,
            "log_uri": self.log_uri
        }
    
    def __str__(self) -> str:
        """String representation for display."""
        timestamp_str = f" [{self.timestamp.strftime('%H:%M:%S')}]" if self.timestamp else ""
        return f"{self.role.value.upper()}{timestamp_str}: {self.content}"


@dataclass
class Chat:
    """A collection of messages."""
    
    messages: List[Message] = field(default_factory=list)
    
    def add(self, message: Message) -> None:
        """Include a message in the chat."""
        self.messages.append(message)
    
    def remove(self, message: Message) -> None:
        """Exclude a message from the chat."""
        if message in self.messages:
            self.messages.remove(message)
    
    def merge(self, other: 'Chat') -> 'Chat':
        """Combine with another chat."""
        merged_messages = self.messages + other.messages
        # Sort by timestamp if available
        merged_messages.sort(key=lambda m: m.timestamp or datetime.min)
        return Chat(messages=merged_messages)
    
    def export(self) -> str:
        """Output as Tigs YAML format with human-readable content blocks."""
        from datetime import datetime
        
        # Build the YAML manually for better control over formatting
        lines = []
        lines.append("schema: tigs.chat/v1")
        lines.append("messages:")
        
        for message in self.messages:
            lines.append(f"- role: {message.role.value}")
            
            # Always use literal block style for content
            lines.append("  content: |")
            # Strip ANSI codes to ensure YAML compatibility
            clean_content = _strip_ansi_codes(message.content)
            # Split content by any line ending style (cross-platform)
            content_lines = clean_content.splitlines()
            if content_lines:
                for content_line in content_lines:
                    lines.append(f"    {content_line}")
            else:
                # Handle empty content
                lines.append(f"    {clean_content}")
            
            # Add timestamp if available
            if message.timestamp:
                lines.append(f"  timestamp: '{message.timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')}'")

            # Add log_uri
            lines.append(f"  log_uri: '{message.log_uri}'")


        return '\n'.join(lines)


@dataclass
class ErrorReport:
    """Information on a parsing failure."""
    
    error: str
    log: str
    location: Optional[str] = None
    recoverable: bool = True
    
    def __str__(self) -> str:
        """Format error for display."""
        location_str = f" at {self.location}" if self.location else ""
        recovery_str = " (recoverable)" if self.recoverable else " (fatal)"
        return f"Error{location_str}: {self.error}{recovery_str}\nLog snippet: {self.log}"


@dataclass
class Record(ABC):
    """Base class for provider-specific records."""

    raw_data: Dict[str, Any] = field(default_factory=dict)
    config: ProviderConfig = field(default_factory=lambda: ProviderConfig("base", "Base", ".base"))

    @classmethod
    def load(cls, json_string: str, config: ProviderConfig) -> 'Record':
        """Parse a JSON string into a Record."""
        try:
            data = json.loads(json_string)
            record = cls(raw_data=data, config=config)
            record._post_load(data)
            return record
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid JSON record: {e}")

    @abstractmethod
    def _post_load(self, data: Dict[str, Any]) -> None:
        """Process loaded data - implemented by subclasses."""
        pass

    @abstractmethod
    def get_role(self) -> str:
        """Get the role from record data."""
        pass

    @abstractmethod
    def get_content(self) -> str:
        """Get the content from record data."""
        pass

    @abstractmethod
    def get_timestamp(self) -> Optional[str]:
        """Get the timestamp from record data."""
        pass

    def extract_message(self, log_uri: str = "") -> Optional[Message]:
        """Get a Message from this record, if applicable."""
        if not self.is_message():
            return None

        role = self.config.role_mappings.get(
            self.get_role().lower() if self.get_role() else '',
            Role.ASSISTANT
        )

        content = self._process_content(self.get_content())
        if not content:
            return None

        timestamp = self._parse_timestamp(self.get_timestamp())

        return Message(
            role=role,
            content=content,
            provider=self.config.name,
            log_uri=log_uri,
            timestamp=timestamp,
            raw_data=self.raw_data
        )

    def is_message(self) -> bool:
        """Check if this record represents a message."""
        role = self.get_role()
        content = self.get_content()

        has_content = bool(str(content).strip()) if content else False
        valid_role = (
            role and
            role.lower() in self.config.message_roles and
            role.lower() not in self.config.skip_roles
        )

        return valid_role and has_content

    def _process_content(self, content: Any) -> str:
        """Process content from various formats to text."""
        if isinstance(content, list):
            content_parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get('text', item.get('content', ''))
                elif isinstance(item, str):
                    text = item
                else:
                    text = str(item)

                if text and text.strip():
                    content_parts.append(text.strip())

            return '\n'.join(content_parts)
        elif isinstance(content, dict):
            return content.get('text', content.get('content', str(content)))

        return str(content).strip() if content else ""

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if not timestamp_str:
            return None

        try:
            # Handle various timestamp formats
            if timestamp_str.endswith('Z'):
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            elif '+' in timestamp_str or '-' in timestamp_str[-6:]:
                return datetime.fromisoformat(timestamp_str)
            else:
                # Try parsing as Unix timestamp
                return datetime.fromtimestamp(float(timestamp_str))
        except (ValueError, AttributeError):
            return None



@dataclass
class LogFile:
    """Base class for log files."""

    file_path: Path
    config: ProviderConfig
    records: List[Record] = field(default_factory=list)

    def load(self) -> None:
        """Read and parse all Records from the file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.file_path}")

        self.records.clear()
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return

                try:
                    # Try to parse as JSON array first
                    data = json.loads(content)
                    if isinstance(data, list):
                        self._load_json_array(data)
                    else:
                        self._load_single_json(content)
                except json.JSONDecodeError:
                    # Fall back to JSONL format
                    self._load_jsonl(content)

        except (IOError, OSError) as e:
            raise FileNotFoundError(f"Cannot read log file: {e}")

    def _load_json_array(self, data: List[Dict[str, Any]]) -> None:
        """Load from JSON array format."""
        for record_data in data:
            try:
                record = self._create_record(json.dumps(record_data))
                self.records.append(record)
            except ValueError as e:
                print(f"Warning: Skipped invalid record: {e}")

    def _load_single_json(self, content: str) -> None:
        """Load single JSON object."""
        try:
            record = self._create_record(content)
            self.records.append(record)
        except ValueError as e:
            print(f"Warning: Skipped invalid record: {e}")

    def _load_jsonl(self, content: str) -> None:
        """Load JSONL format."""
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = self._create_record(line)
                self.records.append(record)
            except ValueError as e:
                print(f"Warning: Skipped invalid record at line {line_num}: {e}")

    @abstractmethod
    def _create_record(self, json_string: str) -> Record:
        """Create a record instance - implemented by subclasses."""
        pass


    def to_chat(self, log_uri: str = "") -> Chat:
        """Convert session records to a Chat object."""
        messages = []
        for record in self.records:
            message = record.extract_message(log_uri=log_uri)
            if message:
                messages.append(message)
        return Chat(messages=messages)


class LogStore:
    """Base class for provider log stores."""

    def __init__(self, config: ProviderConfig):
        """Initialize with provider config."""
        self.config = config
        self.agent = config.name
        self.location = str(Path.cwd())
        self._logs_dir = self._find_logs_directory()

    def _find_logs_directory(self) -> Path:
        """Find the logs directory for this provider."""
        base_dir = Path.home() / self.config.home_dir

        # Try common directory structures
        possible_dirs = [
            base_dir / "tmp",
            base_dir / "logs",
            base_dir / "sessions",
            base_dir / "conversations",
            base_dir
        ]

        for dir_path in possible_dirs:
            if dir_path.exists():
                return dir_path

        # Default to logs directory
        return base_dir / "logs"

    def list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available session logs."""
        logs = []

        try:
            if not self._logs_dir.exists():
                return logs

            # Scan for session directories
            for session_dir in self._logs_dir.iterdir():
                if session_dir.is_dir():
                    logs.extend(self._scan_session_directory(session_dir))

            # Also scan for direct files
            for pattern in self.config.log_patterns:
                for log_file in self._logs_dir.glob(pattern):
                    if log_file.is_file():
                        logs.append(self._create_log_entry(log_file, log_file.stem))

        except (OSError, PermissionError):
            pass

        return logs

    def _scan_session_directory(self, session_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """Scan a session directory for log files."""
        logs = []
        session_id = session_dir.name

        for pattern in self.config.log_patterns:
            for log_file in session_dir.glob(pattern):
                if log_file.is_file():
                    uri = f"{session_id}/{log_file.name}"
                    metadata = self._create_file_metadata(log_file)
                    metadata.update({
                        "file_name": log_file.name,
                        "session_id": session_id
                    })
                    logs.append((uri, metadata))

        return logs

    def _create_log_entry(self, log_file: Path, session_id: str) -> Tuple[str, Dict[str, Any]]:
        """Create a log entry tuple."""
        metadata = self._create_file_metadata(log_file)
        metadata.update({
            "file_name": log_file.name,
            "session_id": session_id
        })
        return (session_id, metadata)

    def _create_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Create metadata dict for a file."""
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessible": file_path.is_file() and os.access(file_path, os.R_OK)
        }

    def get(self, log_uri: str) -> str:
        """Retrieve raw content of a specific log."""
        log_path = self._resolve_log_path(log_uri)

        try:
            if not log_path.exists():
                raise FileNotFoundError(f"Session log file not found: {log_uri}")

            with open(log_path, 'r', encoding='utf-8') as f:
                return f.read()

        except (OSError, PermissionError, UnicodeDecodeError) as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise IOError(f"Cannot read session log file {log_uri}: {e}")

    def _resolve_log_path(self, log_uri: str) -> Path:
        """Resolve log URI to file path."""
        # Handle <session_id>/<file_name> format
        if "/" in log_uri and not log_uri.startswith("/"):
            parts = log_uri.split("/", 1)
            if len(parts) == 2:
                session_id, file_name = parts
                return self._logs_dir / session_id / file_name

        # Handle full path
        if "\\" in log_uri or log_uri.startswith("/"):
            return Path(log_uri)

        # Handle session ID only - try common file names
        for pattern in self.config.log_patterns:
            # Remove wildcard and try as filename
            filename = pattern.replace("*", "logs")
            path = self._logs_dir / log_uri / filename
            if path.exists():
                return path

        # Fallback
        return self._logs_dir / log_uri / "logs.json"

    def live(self) -> Optional[str]:
        """Get URI of currently active log (most recent)."""
        logs = self.list()
        if not logs:
            return None

        sorted_logs = sorted(logs, key=lambda x: x[1].get('modified', ''), reverse=True)
        return sorted_logs[0][0] if sorted_logs else None