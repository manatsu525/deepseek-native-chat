---
name: conversation-management
description: Create, inspect, rename, and remove conversations through first-class tools.
---

# Conversation management

Use `conversation_list` to discover IDs, `conversation_read` to inspect history,
`conversation_create` for a new thread, and `conversation_rename` for titles.
Before deletion, confirm the target ID and use `conversation_delete`; never
delete a different user's data. Treat the current conversation as the place for
the user's request, not as an instruction to alter unrelated history.
