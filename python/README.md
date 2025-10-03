# Cligent (Python)

Unified SDK for parsing CLI agent chats in Python. Ships a lightweight
trial/debugging CLI (`cligent`) and a small API for listing, parsing,
selecting, and composing messages across supported providers.

## Install

```bash
pip install cligent
```

## CLI

After installation, the `cligent` command is available:

```bash
# List recent logs for a provider
cligent --agent codex list

# Parse a specific log URI
cligent --agent codex parse 2025/10/01/rollout-...jsonl

# Parse the most recent (live) log
cligent --agent codex live

# Interactive REPL
cligent interactive
```

Providers: `claude`, `gemini`, `qwen`, `codex`.

## Python API

```python
from cligent import create

agent = create("codex")
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

## Package Structure

- src/ – library sources
- tests/ – test suite
- pyproject.toml – packaging config
