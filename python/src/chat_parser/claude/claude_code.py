"""Claude Code specific implementation for parsing JSONL logs."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..models import Message, Chat, ErrorReport
from ..store import LogStore


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
        raise NotImplementedError
    
    def extract_message(self) -> Optional[Message]:
        """Get a Message from this record, if applicable."""
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def to_chat(self) -> Chat:
        """Convert session records to a Chat object."""
        raise NotImplementedError


class ClaudeStore(LogStore):
    """Claude Code log store implementation."""
    
    def __init__(self, location: Path = None, project_name: str = None):
        """Initialize with base path for Claude logs.
        
        Args:
            location: Root path containing agent logs 
                     (default: ~/.claude/projects/)
            project_name: Optional project name to filter logs
        """
        if location is None:
            location = Path.home() / ".claude" / "projects"
        
        super().__init__("claude-code", location)
        self.project_name = project_name
        self.project_pattern = "*"  # Pattern for project directory names
        self.session_pattern = "*.jsonl"  # Pattern for session file names
    
    def list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available logs for the agent."""
        raise NotImplementedError
    
    def get(self, log_uri: str) -> str:
        """Retrieve raw content of a specific log."""
        raise NotImplementedError
    
    def live(self) -> Optional[str]:
        """Get URI of currently active log."""
        raise NotImplementedError
    
    def _scan_logs(self, project_name: str = None) -> List[Path]:
        """Find all log files for Claude Code.
        
        Args:
            project_name: Optional specific project to scan
            
        Returns:
            List of paths to JSONL log files
        """
        raise NotImplementedError
    
    def _load_session(self, path: Path) -> Session:
        """Load a Session from a specific path.
        
        Args:
            path: Path to JSONL file
            
        Returns:
            Loaded Session object
        """
        raise NotImplementedError