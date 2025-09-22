# Cligent - Chat Parser Library

A Python library for parsing AI agent session logs, with specialized support for Claude Code conversations.

## Installation

```bash
pip install cligent
```

## Features

- Parse Claude Code session logs from JSONL format
- Support for multi-turn conversations with user, assistant, and system messages
- Handle tool usage and function calls in conversations
- Filter and process Claude-specific message formats
- Export conversations to YAML format
- Convert YAML back to Chat objects (decompose functionality)
- Robust error handling for malformed session logs

## Quick Start

### Basic Usage - Parse Claude Code Logs

If you're working in a directory where you've used Claude Code, you can easily access all your session logs for that specific project:

```python
from cligent import ChatParser

# Initialize parser for Claude Code (uses current working directory)
parser = ChatParser("claude-code")

# List all session logs for the current project only
logs = parser.list()
for log_uri, metadata in logs:
    print(f"Session ID: {log_uri}")  # Returns session ID, not full path
    print(f"  Size: {metadata['size']} bytes")
    print(f"  Modified: {metadata['modified']}")

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

# List session logs and pick one (log_uri is now a session ID)
logs = parser.list()
if logs:
    session_id, metadata = logs[0]  # First available log
    
    # Parse specific conversation using session ID
    chat = parser.parse(session_id)
    print(f"Chat has {len(chat.messages)} messages")
    
    # Access individual messages
    for message in chat.messages:
        print(f"{message.role.value}: {message.content}")
```

### Working with Different Project Directories

```python
from cligent import ChatParser

# Parse session logs from a different project directory
parser = ChatParser("claude-code", location="/home/user/projects/my-app")
logs = parser.list()  # Returns session logs for my-app project
```

## Advanced Usage

### Selecting and Composing Messages

```python
from cligent import ChatParser

parser = ChatParser("claude-code")
logs = parser.list()

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

### Exporting to YAML (Tigs Format) and Re-importing

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

    # Re-import YAML back to Chat object
    reconstructed_chat = parser.decompose(yaml_output)
    print(f"Reconstructed chat has {len(reconstructed_chat.messages)} messages")
```

### Working with Multiple Conversations

```python
from cligent import ChatParser

parser = ChatParser("claude-code")

# Select messages from multiple conversations
logs = parser.list()
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
    chat = parser.parse("some-session-id")  # Parse using session ID
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
- `LogStore`: Represents a storage location for session logs

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