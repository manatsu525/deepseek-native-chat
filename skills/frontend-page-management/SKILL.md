---
name: frontend-page-management
description: Find, edit, validate, and maintain real frontend pages on the host.
---

# Frontend page management

In Agent mode, use the host-level `frontend_*` and `host_*` tools. Relative
paths use the shared Agent workspace `/home/share`, which is separate from the
ordinary conversation workspace. Start with `list_files` or a focused
host listing, then use `read_file`, `write_file`, or
`edit_file`. Use absolute paths when changing the real application or
another host resource. Run `check_web_syntax` and the project's own lint,
typecheck, build, or test command when available. For any page change, verify
responsive and accessible states rather than only checking that the file saves.
