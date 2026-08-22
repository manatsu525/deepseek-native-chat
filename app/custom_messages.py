from __future__ import annotations

from typing import Any

from .mimo_local import stream_response as _stream_response


async def stream_response(**kwargs: Any) -> dict[str, Any]:
    """Run the shared Custom agent loop over the Anthropic Messages protocol."""
    return await _stream_response(**kwargs, api_protocol="messages")
