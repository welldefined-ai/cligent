"""Claude Code specific implementation for parsing JSONL session logs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ...core.models import Message, Chat, Role, LogStore, Record, LogFile, ProviderConfig

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
    def load(cls, json_string: str, config: ProviderConfig = CLAUDE_CONFIG) -> 'ClaudeRecord':
        """Parse a JSON string into a ClaudeRecord."""
        return super().load(json_string, config)  # type: ignore[return-value]

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
        self._projects_root = claude_base
        self._project_dir = claude_base / project_folder_name

        # Override the default logs directory for Claude's structure
        self._logs_dir = self._project_dir
        self.project_folder_name = project_folder_name

    def list(self, recursive: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available session logs.

        When recursive is True (default), include logs from any Claude
        project directories whose names start with this project's
        prefix. URIs in recursive mode are relative paths under the
        projects root (e.g., "Users-me-proj-python/log.jsonl"). In
        non-recursive mode, URIs are session IDs (filename stems).
        """
        logs = []

        try:
            if recursive:
                # Aggregate across all project dirs that start with this prefix
                root = getattr(self, "_projects_root", None)
                if root and root.exists():
                    base = self.project_folder_name
                    for proj_dir in root.iterdir():
                        if not proj_dir.is_dir():
                            continue
                        name = str(proj_dir.name)
                        if not name.startswith(base):
                            continue
                        # Derive URI prefix from suffix after base ('-sub' -> 'sub')
                        suffix = name[len(base):]
                        if suffix.startswith('-'):
                            suffix = suffix[1:]
                        else:
                            suffix = ''
                        uri_prefix = suffix.replace('-', '/') if suffix else ''
                        for log_file in proj_dir.glob("*.jsonl"):
                            if not log_file.is_file():
                                continue
                            metadata = self._create_file_metadata(log_file)
                            metadata["project"] = proj_dir.name
                            uri = f"{uri_prefix}/{log_file.name}" if uri_prefix else log_file.name
                            logs.append((uri, metadata))
                else:
                    # Fallback to non-recursive if root missing
                    recursive = False

            if not recursive:
                if not self._project_dir.exists():
                    return logs
                for log_file in self._project_dir.glob("*.jsonl"):
                    if log_file.is_file():
                        metadata = self._create_file_metadata(log_file)
                        metadata["project"] = self.project_folder_name
                        logs.append((log_file.name, metadata))

        except (OSError, PermissionError):
            pass

        return logs

    def _resolve_log_path(self, log_uri: str) -> Path:
        """Resolve log URI to file path for Claude's structure."""
        # Handle URIs: either file name (no path sep) or relative
        # paths under projects root based on the base project prefix.
        if "/" in log_uri or "\\" in log_uri:
            uri_path = Path(log_uri)
            if uri_path.is_absolute():
                raise FileNotFoundError(f"Session log file not found: {log_uri}")

            parts = uri_path.parts
            if not parts:
                raise FileNotFoundError(f"Invalid session log URI: {log_uri}")
            if any(part in {"", ".", ".."} for part in parts):
                raise FileNotFoundError(f"Invalid session log URI: {log_uri}")

            filename = parts[-1]
            suffix_parts = list(parts[:-1])

            if not suffix_parts:
                project_dir = self._project_dir
            else:
                suffix = "-".join(suffix_parts)
                project_dir = self._projects_root / f"{self.project_folder_name}-{suffix}"

            return project_dir / filename

        # File name provided - resolve in current project dir
        return self._project_dir / log_uri


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

    def _parse_from_store(self, log_uri: str) -> Chat:
        file_path = self.store._resolve_log_path(log_uri)

        log_file = ClaudeLogFile(file_path=file_path)
        log_file.load()
        return log_file.to_chat(log_uri=log_uri)
