# Chat Parser Requirements

## Concepts

**[CHP-CON-010]** The Chat Parser _shall_ treat each basic collection of an agent's messages (e.g., a file) as a **log**.

**[CHP-CON-020]** The Chat Parser _shall_ convert messages from logs to a human-readable **chat** that conforms to [Tigs Chat Format](https://raw.githubusercontent.com/sublang-ai/tigs/4a10cedad4a048b29cbdf285ee4c9d2b260811ad/specs/dev/chat_format.md).

## Features

### Parse

**[CHP-PARS-010]** _When_ requested for a specific agent’s logs, the Chat Parser _shall_ provide the organization of the logs (e.g., directories and files).

**[CHP-PARS-020]** _When_ requested for a specific log, the Chat Parser _shall_ extract a chat from the log.

**[CHP-PARS-030]** _When_ a live chat is requested for a specific agent, the Chat Parser _shall_ extract a chat from the latest working log.

### Compose

**[CHP-COMP-010]** The Chat Parser _shall_ allow users to select arbitrary messages from extracted chats for composition.

**[CHP-COMP-020]** The Chat Parser _shall_ allow users to select arbitrary logs for composition.

**[CHP-COMP-030]** _When_ logs and/or messages are selected, the Chat Parser _shall_ compose them into a chat text.

## Exception

**[CHP-EXC-010]** _If_ an invalid or corrupted log is encountered, _then_ the Chat Parser _shall_ skip that log and generate an error report identifying the problematic messages.

**[CHP-EXC-020]** _If_ a log cannot be accessed due to permissions, _then_ the Chat Parser _shall_ skip that log and generate an error report.

## Interface

**[CHP-INT-010]** The Chat Parser _shall_ expose expose programmatic APIs in supported programming languages.

## Security

**[CHP-SEC-010]** The Chat Parser _shall_ operate with read-only access to logs.
