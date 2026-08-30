---
name: frontend-page-management
description: Find, edit, validate, and maintain real frontend pages on the host.
---

# Frontend page management

Start with `frontend_list_pages` or a focused host listing. Read the target
page, preserve unrelated layout and behavior, then use `frontend_write_page` or
`host_apply_patch`. Run `frontend_validate_page` and the project's own lint,
typecheck, build, or test command when available. For a page change, verify
responsive and accessible states rather than only checking that the file saves.
