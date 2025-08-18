# Chat Parser Implementation Terms

## Claude Code

### `Record`
A single JSON line in a JSONL log file.

Attributes:
- *type*: record category (user, assistant, tool_use, tool_result, summary).
- *uuid*: unique identifier.
- *parent_uuid*: reference to parent record.
- *timestamp*: ISO 8601 time.
- *raw_data*: original JSON object.

Actions:
- **load**: parse a JSON string.
- **extract_message**: get a Message.

### `Session`
A complete JSONL log file representing a chat.

Attributes:
- *file_path*: location of the JSONL file.
- *session_id*: unique session identifier.
- *records*: ordered list of Records.
- *summary*: optional chat summary.

Actions:
- **load**: read and parse all Records.

## `Error Report`
Information on a parsing failure.

Attributes:
- *error*: description of the failure.
- *log*: log snippet that caused the error.
