---
name: frontend-page-management
description: Find, edit, validate, and maintain real frontend pages on the host.
---

# Frontend page management

For a page that is a deliverable of the current conversation, start with the
conversation workspace tools (`list_files`, `read_file`, `write_file`, and
`check_web_syntax`) so it appears in the UI's workspace panel. For a page that
must be installed or changed on the real host, start with `frontend_list_pages`
or a focused host listing, then use `frontend_read_page`,
`frontend_write_page`, or `host_apply_patch`. Run `frontend_validate_page` and
the project's own lint, typecheck, build, or test command when available. For
any page change, verify responsive and accessible states rather than only
checking that the file saves.
