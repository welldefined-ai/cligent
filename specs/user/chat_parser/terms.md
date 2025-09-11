# Chat Parser Terms for Users

## `Message`
A single communication unit.

Attributes:
- *role*: Participant (system, user, assistant).
- *content*: Message text.
- *timestamp*: When occurred.
- *metadata*: Additional context (e.g., AI model used).

## `Chat`
A collection of messages.

Attributes:
- *messages*: Ordered list of messages.

Actions:
- **parse**: Extract chat from a log.
- **add**: Include a message.
- **remove**: Exclude a message.
- **export**: Output as Tigs text.

## `Log Store`
Manager for an agent's session logs.

Attributes:
- *agent*: Name of the agent (e.g., "claude-code").
- *location*: Location of the agent's logs (e.g., a directory path).

Actions:
- **list**: Show available session logs for the agent.
- **get**: Retrieve specific log by URI.
- **live**: Get URI of currently active log.

## `Chat Parser`
Main interface for parsing and composing agent chats.

Attributes:
- *agent*: Name of the agent being parsed.
- *store*: Session Log Store for the agent.

Actions:
- **list**: Show available session logs for the agent.
- **parse**: Extract chat from specific or live session log URI.
- **select**: Choose messages from a session log URI for composition.
- **unselect**: Remove messages from selection.
- **compose**: Create Tigs text from selected content.
