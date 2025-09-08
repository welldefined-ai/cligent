"""Qwen Code specific implementation for parsing JSONL logs."""

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
from ...execution.executor import QwenExecutor, MockExecutor
from typing import AsyncIterator


@dataclass
class QwenRecord:
    """A single JSON line in a Qwen Code JSONL log file."""

    type: str  # user, assistant, system, checkpoint, tool_use, tool_result
    timestamp: Optional[str] = None
    content: str = ""
    role: Optional[str] = None
    session_id: Optional[str] = None
    checkpoint_tag: Optional[str] = None
    model: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, json_string: str) -> 'QwenRecord':
        """Parse a JSON string into a QwenRecord."""
        try:
            data = json.loads(json_string)
            
            # Flexible parsing for Qwen Code log formats (based on Gemini CLI fork)
            record_type = data.get('type', data.get('messageType', 'unknown'))
            content = data.get('content', data.get('text', data.get('message', '')))
            role = data.get('role', data.get('sender', data.get('from', '')))
            timestamp = data.get('timestamp', data.get('time', data.get('created_at', '')))
            session_id = data.get('session_id', data.get('sessionId', data.get('conversationId', '')))
            checkpoint_tag = data.get('checkpoint_tag', data.get('checkpointTag', data.get('tag', '')))
            model = data.get('model', data.get('modelName', ''))
            
            return cls(
                type=record_type,
                content=content,
                role=role,
                timestamp=timestamp,
                session_id=session_id,
                checkpoint_tag=checkpoint_tag,
                model=model,
                raw_data=data
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid JSON record: {e}")

    def extract_message(self) -> Optional[Message]:
        """Get a Message from this record, if applicable."""
        if not self.is_message():
            return None

        # Map Qwen Code roles to our Role enum
        role_mapping = {
            'user': Role.USER,
            'assistant': Role.ASSISTANT,
            'model': Role.ASSISTANT,  # Qwen often uses 'model' for assistant
            'qwen': Role.ASSISTANT,   # Qwen-specific role
            'system': Role.SYSTEM,
            'human': Role.USER,       # Alternative user role
            'ai': Role.ASSISTANT,     # Alternative assistant role
        }

        role = role_mapping.get(self.role.lower() if self.role else '', Role.ASSISTANT)

        # Handle different content formats
        content = self.content
        if isinstance(content, list):
            # Extract text from structured content
            content_parts = []
            for item in content:
                if isinstance(item, dict):
                    # Handle various content block types
                    text = item.get('text', item.get('content', ''))
                    if not text and 'tool_use' in item:
                        # Skip tool use blocks for text output
                        continue
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
                    try:
                        timestamp = datetime.fromtimestamp(float(self.timestamp))
                    except ValueError:
                        pass
            except (ValueError, AttributeError):
                pass

        metadata = {
            'type': self.type,
            'session_id': self.session_id,
            'checkpoint_tag': self.checkpoint_tag,
            'model': self.model,
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
            'user', 'assistant', 'system', 'human', 'ai', 'model', 'qwen', 'message'
        }
        return (
            self.type.lower() in message_types or
            (self.role and self.role.lower() in {'user', 'assistant', 'system', 'human', 'ai', 'model', 'qwen'}) or
            # Skip checkpoint and tool records
            (self.type.lower() not in {'checkpoint', 'tool_use', 'tool_result'} and self.content.strip())
        )


@dataclass
class QwenSession:
    """A complete JSONL log file representing a Qwen Code chat."""

    file_path: Path
    session_id: Optional[str] = None
    checkpoint_tags: List[str] = field(default_factory=list)
    records: List[QwenRecord] = field(default_factory=list)

    def load(self) -> None:
        """Read and parse all Records from the file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.file_path}")

        self.records.clear()
        self.checkpoint_tags.clear()
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = QwenRecord.load(line)
                        self.records.append(record)

                        # Extract session metadata
                        if not self.session_id and record.session_id:
                            self.session_id = record.session_id
                        
                        # Collect checkpoint tags
                        if record.checkpoint_tag and record.checkpoint_tag not in self.checkpoint_tags:
                            self.checkpoint_tags.append(record.checkpoint_tag)

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


class QwenStore(LogStore):
    """Qwen Code log store implementation."""

    def __init__(self):
        """Initialize with base path for Qwen Code logs.
        
        Note: Qwen Code stores logs globally in ~/.qwen/, not per-project.
        """
        # Qwen Code stores logs in ~/.qwen/ directory
        qwen_base = Path.home() / ".qwen"
        
        # Look for logs in various possible locations
        possible_dirs = [
            qwen_base / "logs",
            qwen_base / "sessions", 
            qwen_base / "conversations",
            qwen_base / "tmp",
            qwen_base,  # Root qwen directory
        ]
        
        self._logs_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists() and any(dir_path.glob("*.jsonl")):
                self._logs_dir = dir_path
                break
        
        # Default to logs directory if none found
        if self._logs_dir is None:
            self._logs_dir = qwen_base / "logs"

        # Use current directory for LogStore base class compatibility
        super().__init__("qwen-code", str(Path.cwd()))
        self.session_pattern = "*.jsonl"

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


class QwenCodeAgent(AgentBackend):
    """Qwen Code agent implementation."""

    def __init__(self, location: Optional[str] = None, api_key: Optional[str] = None, use_mock: bool = False):
        """Initialize Qwen Code agent.
        
        Args:
            location: Optional workspace location for logs
            api_key: Qwen API key (uses DASHSCOPE_API_KEY env var if not provided)  
            use_mock: If True, use mock executor instead of real Qwen API
        """
        super().__init__(location)
        
        # Initialize executor - real Qwen API or mock
        if use_mock:
            self._executor = MockExecutor("qwen-code")
        else:
            self._executor = QwenExecutor(api_key)

    @property
    def config(self) -> AgentConfig:
        return AgentConfig(
            name="qwen-code",
            display_name="Qwen Code",
            log_extensions=[".jsonl", ".json"],
            requires_session_id=True,
            metadata={
                "log_format": "jsonl",
                "base_dir": "~/.qwen/",
                "supports_checkpoints": True,
                "supports_tools": True,
                "based_on": "gemini-cli"
            }
        )

    def _create_store(self, location: Optional[str] = None) -> LogStore:
        return QwenStore()

    def parse_content(self, content: str, log_uri: str, store: LogStore) -> Chat:
        # Handle both session IDs and full paths
        if "/" in log_uri or "\\" in log_uri:
            file_path = Path(log_uri)
        else:
            file_path = store._logs_dir / f"{log_uri}.jsonl"

        session = QwenSession(file_path=file_path)
        session.load()
        return session.to_chat()



