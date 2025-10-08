"""Codex CLI specific implementation for parsing JSONL session logs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.models import Chat, LogFile, LogStore, ProviderConfig, Record, Role
from ...cligent import Cligent


CODEX_CONFIG = ProviderConfig(
    name="codex-cli",
    display_name="Codex CLI",
    home_dir=".codex",
    role_mappings={
        "user": Role.USER,
        "assistant": Role.ASSISTANT,
        "system": Role.SYSTEM,
    },
    log_patterns=["*.jsonl"],
)


@dataclass
class CodexRecord(Record):
    """A single JSON line in a Codex CLI JSONL log file."""

    entry_type: str = ""
    payload_type: Optional[str] = None
    role: str = ""
    timestamp: Optional[str] = None
    _content: Any = field(default=None, repr=False)

    @classmethod
    def load(cls, json_string: str, config: ProviderConfig = CODEX_CONFIG) -> "CodexRecord":
        """Parse a JSON string into a CodexRecord."""
        return super().load(json_string, config)  # type: ignore[return-value]

    def _post_load(self, data: Dict[str, Any]) -> None:
        """Extract Codex-specific fields."""
        self.entry_type = data.get("type", "")
        self.timestamp = data.get("timestamp")

        payload_data = data.get("payload")
        payload = payload_data if isinstance(payload_data, dict) else {}
        self.payload_type = payload.get("type")
        self.role = payload.get("role", "")
        self._content = payload.get("content")

    def get_role(self) -> str:
        return self.role or ""

    def get_content(self) -> Any:
        return self._content

    def get_timestamp(self) -> Optional[str]:
        return self.timestamp

    def is_message(self) -> bool:
        if self.entry_type != "response_item":
            return False
        if self.payload_type != "message":
            return False
        return super().is_message()


@dataclass
class CodexLogFile(LogFile):
    """A complete JSONL log file representing a Codex CLI chat."""

    def __init__(self, file_path: Path):
        super().__init__(file_path, CODEX_CONFIG)

    def _create_record(self, json_string: str) -> Record:
        return CodexRecord.load(json_string)


class CodexLogStore(LogStore):
    """Codex CLI log store implementation."""

    def __init__(self):
        super().__init__(CODEX_CONFIG)
        sessions_dir = Path.home() / ".codex" / "sessions"
        self._logs_dir = sessions_dir

    def list(self, recursive: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
        logs: List[Tuple[str, Dict[str, Any]]] = []
        if not self._logs_dir.exists():
            return logs

        for file_path in sorted(self._logs_dir.rglob("*.jsonl")):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(self._logs_dir)
            metadata = self._create_file_metadata(file_path)
            metadata.update(
                {
                    "file_name": file_path.name,
                    "session_path": str(relative_path.parent),
                }
            )
            logs.append((str(relative_path), metadata))

        return logs

    def _resolve_log_path(self, log_uri: str) -> Path:
        candidate = Path(log_uri)
        if candidate.is_absolute():
            return candidate

        resolved = self._logs_dir / candidate
        if resolved.exists():
            return resolved

        if candidate.suffix != ".jsonl":
            matches = list(self._logs_dir.rglob(f"{candidate.name}.jsonl"))
            if matches:
                return matches[0]

        return resolved


class CodexCligent(Cligent):
    """Codex CLI agent implementation."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "codex-cli"

    @property
    def display_name(self) -> str:
        return "Codex CLI"

    def _create_store(self) -> LogStore:
        return CodexLogStore()

    def _parse_from_store(self, log_uri: str) -> Chat:
        path_hint = Path(log_uri)
        if path_hint.is_absolute():
            file_path = path_hint
        else:
            candidate = self.store._logs_dir / path_hint
            if candidate.exists():
                file_path = candidate
            else:
                matches = list(self.store._logs_dir.rglob(path_hint.name))
                if matches:
                    file_path = matches[0]
                else:
                    file_path = candidate

        log_file = CodexLogFile(file_path=file_path)
        log_file.load()
        return log_file.to_chat(log_uri=log_uri)
