# Cligent (Python)

Unified SDK for parsing CLI agent chats in Python. Ships a lightweight
trial/debugging CLI (`cligent`) and a small API for listing, parsing,
selecting, and composing messages across supported providers.
| Agent | Description | Log Location                             |
|-------|-------------|------------------------------------------|
| **claude** | Claude Code sessions | `~/.claude/projects/` (project-specific) |
| **gemini** | Gemini CLI conversations | `~/.gemini/logs/`      |
| **qwen** | Qwen Coder sessions | `~/.qwen/sessions/`           |
| **codex** | Codex CLI sessions | `~/.codex/sessions/`          |

## Install

```bash
pip install cligent
```

## Python API

```python
from cligent import create

agent = create("claude")
logs = agent.list_logs()
chat = agent.parse()  # most recent
agent.select(logs[0][0], [0, 2])
yaml_text = agent.compose()
loaded = agent.decompose(yaml_text)
```

## Development

```bash
pip install -e .[dev]
pytest -q
```

### CLI

After installation, the `cligent` command is available:

```bash
# List recent logs for a provider
cligent --agent codex list

# Parse a specific log URI
cligent --agent codex parse 2025/10/01/rollout-...jsonl

# Show the last N messages (e.g., 5) from the live (most recent) log
cligent --agent codex live -n 5

# Help
cligent --help
```

Providers: `claude`, `gemini`, `qwen`, `codex`.

## Package Structure

- src/ – library sources
- tests/ – test suite
- pyproject.toml – packaging config
