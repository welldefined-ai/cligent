# Cligent - Chat Parser Library

A Python library for parsing AI agent conversation logs, with specialized support for Claude Code conversations.

## Installation

```bash
pip install cligent
```

## Features

- Parse Claude Code conversation logs from JSONL format
- Support for multi-turn conversations with user, assistant, and system messages
- Handle tool usage and function calls in conversations
- Filter and process Claude-specific message formats
- Export conversations to YAML format
- Robust error handling for malformed logs

## Quick Start

### Basic Usage - Parse Claude Code Logs

If you're working in a git repository where you've used Claude Code, you can easily access all your conversation logs:

```python
from cligent import ChatParser

# Initialize parser for Claude Code (uses current directory by default)
parser = ChatParser("claude-code")

# List all available conversation logs
logs = parser.list_logs()
for log_uri, metadata in logs:
    print(f"Log: {log_uri}")
    print(f"  Project: {metadata['project']}")
    print(f"  Size: {metadata['size']} bytes")

# Parse the most recent conversation
latest_chat = parser.parse()  # Gets live/most recent log
if latest_chat:
    print(f"Latest chat has {len(latest_chat.messages)} messages")
    
    for message in latest_chat.messages:
        print(f"  {message.role.value}: {message.content[:100]}...")
```

### Parse a Specific Conversation

```python
from cligent import ChatParser

parser = ChatParser("claude-code")

# List logs and pick one
logs = parser.list_logs()
if logs:
    log_uri, metadata = logs[0]  # First available log
    
    # Parse specific conversation
    chat = parser.parse(log_uri)
    print(f"Chat has {len(chat.messages)} messages")
    
    # Access individual messages
    for message in chat.messages:
        print(f"{message.role.value}: {message.content}")
```

### Working with Custom Log Locations

```python
from cligent import ChatParser

# Specify custom location for logs
parser = ChatParser("claude-code", location="/path/to/custom/logs")

# Or parse logs from a different project directory
parser = ChatParser("claude-code", location="/path/to/other/project")
logs = parser.list_logs()
```

## Advanced Usage

### Selecting and Composing Messages

```python
from cligent import ChatParser

parser = ChatParser("claude-code")
logs = parser.list_logs()

if logs:
    log_uri, _ = logs[0]
    
    # Select specific messages (by index)
    parser.select(log_uri, [0, 2, 4])  # Select 1st, 3rd, and 5th messages
    
    # Compose selected messages into YAML format
    yaml_output = parser.compose()
    print(yaml_output)
    
    # Clear selection before making a new one
    parser.clear_selection()
    
    # Select all messages from a conversation
    parser.select(log_uri)  # Selects all messages
    yaml_output = parser.compose()
    print(yaml_output)
    
    # Note: select() calls are additive - they add to existing selection.
    # Use clear_selection() between selects to avoid duplicates.
    
    # Clear selection when done
    parser.clear_selection()
```

### Exporting to YAML (Tigs Format)

```python
from cligent import ChatParser

parser = ChatParser("claude-code")
chat = parser.parse()  # Get latest chat

if chat:
    # Export individual chat to YAML
    yaml_output = chat.export()
    print(yaml_output)
    
    # Save to file
    with open("conversation.yaml", "w") as f:
        f.write(yaml_output)
```

### Working with Multiple Conversations

```python
from cligent import ChatParser

parser = ChatParser("claude-code")

# Select messages from multiple conversations
logs = parser.list_logs()
if len(logs) >= 2:
    # Select first message from first log
    parser.select(logs[0][0], [0])
    
    # Select last message from second log  
    chat2 = parser.parse(logs[1][0])
    parser.select(logs[1][0], [len(chat2.messages) - 1])
    
    # Compose combined output
    combined_yaml = parser.compose()
    print(combined_yaml)
```

### Error Handling

```python
from cligent import ChatParser, ChatParserError, LogCorruptionError

try:
    parser = ChatParser("claude-code")
    chat = parser.parse("/path/to/specific/log.jsonl")
except FileNotFoundError:
    print("Log file not found")
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