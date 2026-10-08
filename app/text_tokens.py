"""Shared local content limits, in o200k_base tokens (not provider billing tokens)."""
from functools import lru_cache
import tiktoken


@lru_cache(maxsize=1)
def encoding():
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(encoding().encode_ordinary(text))


def truncate_tokens(text: str, limit: int, *, tail: bool = False) -> str:
    ids = encoding().encode_ordinary(text)
    if len(ids) <= limit:
        return text
    if limit <= 0:
        return ""
    ids = ids[-limit:] if tail else ids[:limit]
    result = encoding().decode(ids, errors="ignore")
    # A boundary may bisect a UTF-8 character. Never add replacement characters.
    while count_tokens(result) > limit:
        ids = ids[1:] if tail else ids[:-1]
        result = encoding().decode(ids, errors="ignore")
    return result


def upstream_char_ceiling(tokens: int) -> int:
    """Overfetch at char-only endpoints; apply the actual token limit locally."""
    return tokens * 256 + 1024
