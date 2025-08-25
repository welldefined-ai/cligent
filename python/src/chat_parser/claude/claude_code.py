"""Claude Code specific implementation for parsing JSONL logs."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..models import Message, Chat, ErrorReport, Role
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
    
    def __init__(self, location: Path = None, project_name: str = None):
        """Initialize with base path for Claude logs.
        
        Args:
            location: Root path containing agent logs 
                     (default: ~/.claude/projects/)
            project_name: Optional project name to filter logs
        """
        if location is None:
            location = str(Path.home() / ".claude" / "projects")
        
        # Store as Path for internal use
        self._location_path = Path(location)
        
        super().__init__("claude-code", location)
        self.project_name = project_name
        self.project_pattern = "*"  # Pattern for project directory names
        self.session_pattern = "*.jsonl"  # Pattern for session file names
    
    def list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available logs for the agent."""
        logs = []
        
        try:
            if not self._location_path.exists():
                return logs
            
            # Scan for JSONL files
            pattern = self.project_name if self.project_name else "*"
            for project_dir in self._location_path.glob(pattern):
                if not project_dir.is_dir():
                    continue
                    
                for log_file in project_dir.glob(self.session_pattern):
                    if log_file.is_file():
                        stat = log_file.stat()
                        metadata = {
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "project": project_dir.name,
                            "accessible": log_file.is_file() and os.access(log_file, os.R_OK)
                        }
                        logs.append((str(log_file), metadata))
                        
        except (OSError, PermissionError):
            # Return empty list if we can't access the directory
            pass
            
        return logs
    
    def get(self, log_uri: str) -> str:
        """Retrieve raw content of a specific log."""
        log_path = Path(log_uri)
        
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
        """Get URI of currently active log."""
        # Find the most recently modified log file
        logs = self.list()
        if not logs:
            return None
            
        # Sort by modification time (most recent first)
        sorted_logs = sorted(logs, key=lambda x: x[1].get('modified', ''), reverse=True)
        return sorted_logs[0][0] if sorted_logs else None
