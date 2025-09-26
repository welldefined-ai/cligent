"""Claude Code specific implementation for parsing JSONL session logs."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ...core.models import Message, Chat, ErrorReport, Role, LogStore, Record, LogFile, ProviderConfig

from ...cligent import Cligent


# Claude-specific configuration
CLAUDE_CONFIG = ProviderConfig(
    name="claude-code",
    display_name="Claude Code",
    home_dir=".claude",
    role_mappings={
        'user': Role.USER,
        'assistant': Role.ASSISTANT,
        'system': Role.SYSTEM
    },
    log_patterns=["*.jsonl"]
)


@dataclass
class ClaudeRecord(Record):
    """A single JSON line in a JSONL log file."""

    type: str = ""
    timestamp: Optional[str] = None

    @classmethod
    def load(cls, json_string: str) -> 'ClaudeRecord':
        """Parse a JSON string into a ClaudeRecord."""
        return super().load(json_string, CLAUDE_CONFIG)

    def _post_load(self, data: Dict[str, Any]) -> None:
        """Extract Claude-specific fields."""
        self.type = data.get('type', 'unknown')
        self.timestamp = data.get('timestamp')

    def get_role(self) -> str:
        return self.type

    def get_content(self) -> str:
        message_data = self.raw_data.get('message', {})
        return message_data.get('content', '')

    def get_timestamp(self) -> Optional[str]:
        return self.timestamp

    def extract_message(self, log_uri: str = "") -> Optional[Message]:
        """Get a Message from this record, if applicable."""
        # First check for special tool uses (ExitPlanMode, plan responses)
        # These take priority over regular message extraction
        if self.is_exit_plan_mode():
            return self._extract_plan_message(log_uri)

        # Use base class for regular messages
        return super().extract_message(log_uri)

    def _process_content(self, content: Any) -> str:
        """Process content from various formats to text."""
        # Handle Claude API format with text blocks
        if isinstance(content, list) and content:
            # Extract only text blocks - ignore everything else
            content_parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text = block.get('text', '').strip()
                    if text:  # Only include non-empty text
                        content_parts.append(text)

            # If no text content found, return empty
            if not content_parts:
                return ""

            return '\n'.join(content_parts)

        return super()._process_content(content)

    def is_message(self) -> bool:
        """Check if this record represents a message."""
        # Use base class logic but also handle Claude-specific cases
        if self.is_exit_plan_mode():
            return True
        return super().is_message()

    def is_exit_plan_mode(self) -> bool:
        """Check if this record is an ExitPlanMode tool use."""
        if self.type != 'assistant':
            return False
        
        message_data = self.raw_data.get('message', {})
        content = message_data.get('content', [])
        
        if not isinstance(content, list):
            return False
            
        # Look for ExitPlanMode tool use
        for block in content:
            if (isinstance(block, dict) and 
                block.get('type') == 'tool_use' and 
                block.get('name') == 'ExitPlanMode'):
                return True
        return False

    def _extract_plan_message(self, log_uri: str = "") -> Optional[Message]:
        """Extract plan content from ExitPlanMode tool use."""
        message_data = self.raw_data.get('message', {})
        content = message_data.get('content', [])

        if not isinstance(content, list):
            return None

        # Find the ExitPlanMode tool use
        plan_content = None
        for block in content:
            if (isinstance(block, dict) and
                block.get('type') == 'tool_use' and
                block.get('name') == 'ExitPlanMode'):
                plan_input = block.get('input', {})
                plan_content = plan_input.get('plan', '')
                break

        if not plan_content:
            return None

        # Format the plan as a readable message
        formatted_content = f"📋 **Plan Proposal**\n\n{plan_content}\n\n---\n*This plan was presented for user approval in Claude Code's planning mode.*"

        # Parse timestamp if available
        timestamp = None
        if self.timestamp:
            try:
                timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass

        return Message(
            role=Role.ASSISTANT,
            content=formatted_content,
            provider=self.config.name,
            log_uri=log_uri,
            timestamp=timestamp,
            raw_data=self.raw_data
        )



@dataclass
class ClaudeLogFile(LogFile):
    """A complete JSONL log file representing a chat."""

    def __init__(self, file_path: Path):
        super().__init__(file_path, CLAUDE_CONFIG)

    def _create_record(self, json_string: str) -> Record:
        """Create a Claude Record instance."""
        return ClaudeRecord.load(json_string)



class ClaudeLogStore(LogStore):
    """Claude Code log store implementation."""

    def __init__(self):
        """Initialize with base path for Claude session logs.

        Uses current working directory to find Claude session logs.
        """
        super().__init__(CLAUDE_CONFIG)

        # Claude has a special project-based directory structure
        working_dir = Path.cwd()
        project_folder_name = str(working_dir.absolute()).replace("/", "-")
        claude_base = Path.home() / ".claude" / "projects"
        self._project_dir = claude_base / project_folder_name

        # Override the default logs directory for Claude's structure
        self._logs_dir = self._project_dir
        self.project_folder_name = project_folder_name

    def list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available session logs for the current project."""
        logs = []

        try:
            if not self._project_dir.exists():
                return logs

            # Scan for JSONL files in this project's directory only
            for log_file in self._project_dir.glob("*.jsonl"):
                if log_file.is_file():
                    metadata = self._create_file_metadata(log_file)
                    metadata["project"] = self.project_folder_name
                    # Use session ID (filename without path) as URI
                    session_id = log_file.stem
                    logs.append((session_id, metadata))

        except (OSError, PermissionError):
            pass

        return logs

    def _resolve_log_path(self, log_uri: str) -> Path:
        """Resolve log URI to file path for Claude's structure."""
        # Handle both session IDs and full paths for compatibility
        if "/" in log_uri or "\\" in log_uri:
            # Full path provided (backwards compatibility)
            return Path(log_uri)
        else:
            # Session ID provided - construct full path
            return self._project_dir / f"{log_uri}.jsonl"


class ClaudeCligent(Cligent):
    """Claude Code agent implementation."""

    def __init__(self):
        """Initialize Claude Code agent."""
        super().__init__()

    @property
    def name(self) -> str:
        return "claude-code"
        
    @property
    def display_name(self) -> str:
        return "Claude Code"

    def _create_store(self) -> LogStore:
        return ClaudeLogStore()

    def parse_content(self, content: str, log_uri: str) -> Chat:
        # 使用现有的LogFile逻辑
        if "/" in log_uri or "\\" in log_uri:
            file_path = Path(log_uri)
        else:
            file_path = self.store._project_dir / f"{log_uri}.jsonl"

        log_file = ClaudeLogFile(file_path=file_path)
        log_file.load()
        return log_file.to_chat(log_uri=log_uri)



