"""Local input-token estimates, calibrated by provider usage in the caller."""
from __future__ import annotations

import json
import math
from typing import Any
from .text_tokens import count_tokens

DEFAULT_TOKEN_BUDGET = 250_000
MIN_TOKEN_BUDGET = 8192
MAX_TOKEN_BUDGET = 4_194_304


def normalize_token_budget(value: Any) -> int:
    try:
        return max(MIN_TOKEN_BUDGET, min(MAX_TOKEN_BUDGET, int(value)))
    except (TypeError, ValueError):
        return DEFAULT_TOKEN_BUDGET


def legacy_token_budget(chars: Any) -> int:
    if int(chars or 240_000) == 240_000:
        return DEFAULT_TOKEN_BUDGET
    return normalize_token_budget(int(chars) // 4)


def estimate_tokens(value: Any) -> int:
    """Estimate text/schema/media costs without a model-specific tokenizer.

    Native assistant mirrors are counted once. Opaque state and image costs
    are approximate; real input usage supersedes the estimate in the caller.
    """
    extras = 0
    def project(item: Any) -> Any:
        nonlocal extras
        if isinstance(item, list):
            return [project(x) for x in item]
        if not isinstance(item, dict):
            return item
        if item.get("responses_output_items"):
            return project(item["responses_output_items"])
        if item.get("type") in {"image", "image_url", "input_image"}:
            extras += 765
            return {"type": "image"}
        result = {}
        for key, part in item.items():
            if key == "encrypted_content" and isinstance(part, str):
                extras += math.ceil(len(part) / 8)
                result[key] = "[opaque reasoning]"
            else:
                result[key] = project(part)
        return result
    text = json.dumps(project(value), ensure_ascii=False, separators=(",", ":"))
    return extras + count_tokens(text)
