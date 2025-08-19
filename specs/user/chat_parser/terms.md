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
Manager for an agent's logs.

Attributes:
- *agent*: Name of the agent (e.g., "claude-code").
- *location*: Location of the agent's logs (e.g., a directory path).

Actions:
- **list**: Show available logs for the agent.
- **get**: Retrieve specific log.
- **live**: Retrieve currently active log.

## `Chat Parser`
Main interface for parsing and composing agent chats.

Attributes:
- *agent*: Name of the agent being parsed.
- *store*: Log Store for the agent.

Actions:
- **list_logs**: Show available logs for the agent.
- **parse**: Extract chat from specific or live log.
- **select_messages**: Choose specific messages for composition.
- **select_logs**: Choose multiple logs for composition.
- **compose**: Create Tigs text from selected content.
