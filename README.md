# Cligent

Unified SDK for CLI agents (Claude Code, Gemini CLI, Qwen Coder). Parse, select, and compose messages from AI agent session logs with support for YAML export/import.

## Features

- **Multi-Agent Support**: Works with Claude Code, Gemini CLI, and Qwen Coder
- **Session Log Parsing**: Extract conversations from JSONL log files
- **Message Selection**: Select specific messages from conversations
- **YAML Export/Import**: Convert messages to/from Tigs YAML format
- **Interactive CLI**: Built-in command-line interface for testing
- **Project-Aware**: Automatically finds logs for current project directory

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/cligent.git
cd cligent

# Install dependencies
pip install -r requirements.txt
```

## Language Implementations

- Python: see python/README.md

### Python API

```python
from src import create

# Create an agent (claude, gemini, or qwen)
agent = create("claude")

# List available logs for current project
logs = agent.list_logs()
print(f"Found {len(logs)} conversation logs")

# Parse the most recent conversation
chat = agent.parse()  # No URI = live/most recent log
if chat:
    print(f"Latest chat has {len(chat.messages)} messages")

# Select specific messages and export to YAML
agent.select(logs[0][0], [0, 2])  # Select 1st and 3rd messages
yaml_output = agent.compose()
print(yaml_output)

# Load messages from YAML file
with open("conversation.yaml", "r") as f:
    yaml_content = f.read()
loaded_chat = agent.decompose(yaml_content)
print(f"Loaded {len(loaded_chat.messages)} messages from YAML")
```

### Interactive CLI

```bash
python example_usage.py
```

The CLI provides commands for:
- **list**: Show available conversation logs
- **parse**: Parse a specific conversation
- **live**: Parse the most recent conversation
- **select**: Select messages for composition
- **compose**: Export selected messages to YAML
- **decompose**: Load messages from YAML file
- **clear**: Clear message selection

## Supported Agents

| Agent | Description | Log Location |
|-------|-------------|--------------|
| **claude** | Claude Code sessions | `~/.claude/tmp/` (project-specific) |
| **gemini** | Gemini CLI conversations | `~/.gemini/logs/` |
| **qwen** | Qwen Coder sessions | `~/.qwen/sessions/` |

## API Reference

### Core Classes

- **`Cligent`**: Abstract base class for all agents
- **`Chat`**: Collection of messages representing a conversation
- **`Message`**: Individual message with role, content, and metadata
- **`Role`**: Enum for message roles (USER, ASSISTANT, SYSTEM)

### Key Methods

- **`create(agent_type)`**: Factory function to create agent instances
- **`parse(log_uri=None)`**: Parse conversation from log (None = most recent)
- **`list_logs()`**: List available conversation logs with metadata
- **`select(log_uri, indices=None)`**: Select messages for composition
- **`compose(*args)`**: Export messages to Tigs YAML format
- **`decompose(yaml_string)`**: Import messages from Tigs YAML format
- **`clear_selection()`**: Clear currently selected messages

### YAML Format

Uses the Tigs chat format (`tigs.chat/v1`):

```yaml
schema: tigs.chat/v1
messages:
- role: user
  content: |
    Hello, can you help me with Python?
  timestamp: '2024-01-01T12:00:00Z'
- role: assistant
  content: |
    Of course! I'd be happy to help with Python.
```

## Error Handling

```python
from src import create

try:
    agent = create("claude")
    chat = agent.parse("invalid-session-id")
except FileNotFoundError:
    print("Log file not found")
except ValueError as e:
    print(f"Invalid data format: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the GNU Affero General Public License v3.0.