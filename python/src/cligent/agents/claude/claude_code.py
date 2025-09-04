"""Claude Code specific implementation for parsing JSONL logs."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ...core.models import Message, Chat, ErrorReport, Role
from ...parsers.store import LogStore

from ...core.agent import AgentBackend, AgentConfig
from ...execution.task_models import TaskResult, TaskUpdate, TaskConfig
from ...execution.executor import MockExecutor
from typing import AsyncIterator


@dataclass
class Record:
    """A single JSON line in a JSONL log file."""

    type: str  # user, assistant, tool_use, tool_result, summary
    uuid: str
    parent_uuid: Optional[str] = None
    timestamp: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, json_string: str) -> 'Record':
        """Parse a JSON string into a Record."""
        try:
            data = json.loads(json_string)
            return cls(
                type=data.get('type', 'unknown'),
                uuid=data.get('uuid', ''),
                parent_uuid=data.get('parent_uuid'),
                timestamp=data.get('timestamp'),
                raw_data=data
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid JSON record: {e}")

    def extract_message(self) -> Optional[Message]:
        """Get a Message from this record, if applicable."""
        if not self.is_message():
            return None

        # Map Claude types to our Role enum
        role_mapping = {
            'user': Role.USER,
            'assistant': Role.ASSISTANT,
            'system': Role.SYSTEM
        }

        role = role_mapping.get(self.type, Role.ASSISTANT)

        # Claude logs have message content nested under 'message' key
        message_data = self.raw_data.get('message', {})
        content = message_data.get('content', '')

        # Handle content that might be a list (Claude API format)
        if isinstance(content, list) and content:
            # Extract only text blocks - ignore everything else
            content_parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text = block.get('text', '').strip()
                    if text:  # Only include non-empty text
                        content_parts.append(text)

            # If no text content found, skip this message
            if not content_parts:
                return None

            content = '\n'.join(content_parts)
        elif isinstance(content, str):
            content = content.strip()
            # If it's just whitespace or empty, skip
            if not content:
                return None
        else:
            # Skip non-string, non-list content
            return None

        # Parse timestamp if available
        timestamp = None
        if self.timestamp:
            try:
                timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass

        metadata = {
            'uuid': self.uuid,
            'parent_uuid': self.parent_uuid,
            'raw_type': self.type
        }

        return Message(
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata
        )

    def is_message(self) -> bool:
        """Check if this record represents a message."""
        return self.type in ('user', 'assistant', 'system')


@dataclass
class Session:
    """A complete JSONL log file representing a chat."""

    file_path: Path
    session_id: Optional[str] = None
    records: List[Record] = field(default_factory=list)
    summary: Optional[str] = None

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
                        record = Record.load(line)
                        self.records.append(record)

                        # Extract session metadata
                        if not self.session_id and 'sessionId' in record.raw_data:
                            self.session_id = record.raw_data.get('sessionId')
                        elif record.type == 'summary':
                            self.summary = record.raw_data.get('summary', '')

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


class ClaudeStore(LogStore):
    """Claude Code log store implementation."""

    def __init__(self, location: Path = None):
        """Initialize with base path for Claude logs.

        Args:
            location: Working directory to find Claude logs for
                     (default: current working directory)
        """
        # Determine the working directory
        if location is None:
            working_dir = Path.cwd()
        else:
            working_dir = Path(location)

        # Convert working directory to Claude project folder name
        # Claude uses path with / replaced by -
        project_folder_name = str(working_dir.absolute()).replace("/", "-")

        # Find the Claude logs directory for this project
        claude_base = Path.home() / ".claude" / "projects"
        self._project_dir = claude_base / project_folder_name

        super().__init__("claude-code", str(working_dir))
        self.project_folder_name = project_folder_name
        self.session_pattern = "*.jsonl"  # Pattern for session file names

    def list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available logs for the current project."""
        logs = []

        try:
            if not self._project_dir.exists():
                return logs

            # Scan for JSONL files in this project's directory only
            for log_file in self._project_dir.glob(self.session_pattern):
                if log_file.is_file():
                    stat = log_file.stat()
                    # Use session ID (filename without path) as URI
                    session_id = log_file.stem  # filename without .jsonl extension
                    metadata = {
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "project": self.project_folder_name,
                        "accessible": log_file.is_file() and os.access(log_file, os.R_OK)
                    }
                    # Return session ID as URI, not full path
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
        # Handle both session IDs and full paths for compatibility
        if "/" in log_uri or "\\" in log_uri:
            # Full path provided (backwards compatibility)
            log_path = Path(log_uri)
        else:
            # Session ID provided - construct full path
            log_path = self._project_dir / f"{log_uri}.jsonl"

        try:
            if not log_path.exists():
                raise FileNotFoundError(f"Log file not found: {log_uri}")

            with open(log_path, 'r', encoding='utf-8') as f:
                return f.read()

        except (OSError, PermissionError, UnicodeDecodeError) as e:
            if isinstance(e, FileNotFoundError):
                raise  # Re-raise FileNotFoundError as-is
            raise IOError(f"Cannot read log file {log_uri}: {e}")

    def live(self) -> Optional[str]:
        """Get URI of currently active log (most recent)."""
        # Find the most recently modified log file
        logs = self.list()
        if not logs:
            return None

        # Sort by modification time (most recent first)
        sorted_logs = sorted(logs, key=lambda x: x[1].get('modified', ''), reverse=True)
        # Return session ID of most recent log
        return sorted_logs[0][0] if sorted_logs else None

class ClaudeCodeAgent(AgentBackend):
    """Claude Code agent implementation."""

    def __init__(self):
        # Initialize executor for task execution
        self._executor = MockExecutor("claude-code")  # Using mock for now

    @property
    def config(self) -> AgentConfig:
        return AgentConfig(
            name="claude-code",
            display_name="Claude Code",
            log_extensions=[".jsonl"],
            requires_session_id=True,
            supports_execution=True,  # Enable task execution
            metadata={
                "log_format": "jsonl",
                "project_based": True,
                "base_dir": "~/.claude/projects/"
            }
        )

    def create_store(self, location: Optional[str] = None) -> LogStore:
        return ClaudeStore(location=location)

    def parse_content(self, content: str, log_uri: str, store: LogStore) -> Chat:

        # 使用现有的Session逻辑
        if "/" in log_uri or "\\" in log_uri:
            file_path = Path(log_uri)
        else:
            file_path = store._project_dir / f"{log_uri}.jsonl"

        session = Session(file_path=file_path)
        session.load()
        return session.to_chat()

    def detect_agent(self, log_path: Path) -> bool:
        """Detect Claude Code logs by checking JSON structure."""
        if log_path.suffix != ".jsonl":
            return False

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                # Check first few lines
                for i, line in enumerate(f):
                    if i >= 3:  # Only check first 3 lines
                        break
                    if line.strip():
                        data = json.loads(line.strip())
                        # Claude Code specific fields
                        if 'type' in data and 'uuid' in data and data['type'] in ['user', 'assistant', 'tool_use']:
                           return True
        except:
            pass

        return False

    # Task execution methods
    async def execute_task(self, task: str, config: TaskConfig = None) -> TaskResult:
        """Execute a task using Claude Code."""
        if config is None:
            config = TaskConfig()
        return await self._executor.execute_task(task, config)

    async def execute_task_stream(self, task: str, config: TaskConfig = None) -> AsyncIterator[TaskUpdate]:
        """Execute task with streaming updates."""
        if config is None:
            config = TaskConfig(stream=True)
        async for update in self._executor.execute_task_stream(task, config):
            yield update