# Cligent - Chat Parser Library

A Python library for parsing AI agent conversation logs, with specialized support for Claude AI conversations.

## Installation

```bash
pip install cligent
```

## Features

- Parse AI conversation logs from JSONL format
- Support for multi-turn conversations with user, assistant, and system messages
- Handle tool usage and function calls in conversations
- Filter and process Claude-specific message formats
- Export conversations to YAML format
- Robust error handling for malformed logs

## Quick Start

```python
from cligent import ChatParser

# Initialize parser with a project directory
parser = ChatParser("/path/to/project")

# Parse all conversation logs
chats = parser.parse_logs()

# Access individual messages
for chat in chats:
    print(f"Chat ID: {chat.id}")
    print(f"Messages: {len(chat.messages)}")
    
    for message in chat.messages:
        print(f"  {message.role}: {message.content[:100]}...")
```

## Advanced Usage

### Working with specific stores

```python
from cligent import ChatParser, LogStore

# Parse logs from specific Claude Code conversations
parser = ChatParser("/path/to/project")

# Get available stores
stores = parser.list_stores()

# Parse logs from a specific store
store = LogStore("project_name", "/path/to/logs")
chats = parser.parse_logs(stores=[store])
```

### Exporting to YAML

```python
from cligent import ChatParser

parser = ChatParser("/path/to/project")
chats = parser.parse_logs()

# Export to YAML format
yaml_output = parser.export_yaml(chats)
print(yaml_output)
```

### Error Handling

```python
from cligent import ChatParser, ChatParserError, LogCorruptionError

try:
    parser = ChatParser("/path/to/project")
    chats = parser.parse_logs()
except LogCorruptionError as e:
    print(f"Log file corrupted: {e}")
except ChatParserError as e:
    print(f"Parser error: {e}")
```

## API Reference

### Core Classes

- `ChatParser`: Main parser class for processing conversation logs
- `Chat`: Represents a complete conversation
- `Message`: Individual message within a conversation
- `LogStore`: Represents a storage location for conversation logs

### Exceptions

- `ChatParserError`: Base exception for all parser errors
- `ParseError`: Error during parsing operations
- `LogAccessError`: Error accessing log files
- `LogCorruptionError`: Corrupted or malformed log file
- `InvalidFormatError`: Invalid message format

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.