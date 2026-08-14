"""Shared reasoning-effort levels accepted by the chat API and adapters."""

from __future__ import annotations

from typing import Final


LEVELS: Final[tuple[str, ...]] = ("low", "medium", "high", "xhigh", "max")
DEFAULT: Final[str] = "high"


def normalize(value: str) -> str:
    """Return a supported effort, falling back only for internal adapter use."""
    return value if value in LEVELS else DEFAULT
