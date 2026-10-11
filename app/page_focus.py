"""Fit a long fetched page into the token allowance around the reader's objective.

Readers used to keep only the first N tokens, so a rule book or wiki section
past that point could never be read, however exact the objective. Pages within
the allowance are returned unchanged. Longer pages keep their beginning (title
and context) plus the passages that best match the objective and the search
queries, in document order, with explicit gaps.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

from .text_tokens import count_tokens, truncate_tokens

CHUNK_TOKENS = 400
HEAD_TOKENS = 1000
_LATIN = re.compile(r"[a-z][a-z'’-]+|\d+(?:\.\d+)*", re.IGNORECASE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]+")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "how", "find", "about",
    "full", "page", "details", "detail", "info", "information", "rules", "rule", "list",
    "into", "are", "its", "their", "including", "include", "of", "to", "in", "on", "or",
}


def _terms(texts: Iterable[str]) -> list[str]:
    found: list[str] = []
    for text in texts:
        for word in _LATIN.findall(text or ""):
            word = word.casefold().strip("'’-")
            if word.isdigit() or (len(word) > 2 and word not in _STOPWORDS):
                found.append(word)
        for run in _CJK.findall(text or ""):
            found.extend(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return list(dict.fromkeys(found))


def _matches(term: str, folded: str) -> bool:
    if term.isdigit() or term.isascii():
        return re.search(rf"(?<![0-9a-z]){re.escape(term)}(?![0-9a-z])", folded) is not None
    return term in folded


def _chunks(content: str) -> list[str]:
    """Split on paragraph boundaries into pieces of roughly CHUNK_TOKENS."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in re.split(r"(\n\s*\n)", content):
        tokens = count_tokens(paragraph)
        if tokens > CHUNK_TOKENS:
            if current:
                pieces.append("".join(current))
                current, size = [], 0
            rest = paragraph
            while rest:
                head = truncate_tokens(rest, CHUNK_TOKENS)
                if not head:
                    break
                pieces.append(head)
                rest = rest[len(head):]
            continue
        if size + tokens > CHUNK_TOKENS and current:
            pieces.append("".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += tokens
    if current:
        pieces.append("".join(current))
    return pieces


def focus_page(content: str, limit: int, objective: str = "", queries: Iterable[str] = ()) -> str:
    total = count_tokens(content)
    if total <= limit:
        return content
    terms = _terms([objective, *queries])
    chunks = _chunks(content)
    folded = [chunk.casefold() for chunk in chunks]
    sizes = [count_tokens(chunk) for chunk in chunks]
    # Rarer terms (a haunt number, a proper name) outweigh words found everywhere.
    weights = {}
    for term in terms:
        df = sum(1 for text in folded if _matches(term, text))
        if df:
            weights[term] = math.log(1 + len(chunks) / df)
    scores = [sum(w for term, w in weights.items() if _matches(term, text)) for text in folded]

    budget = limit - 120  # room for the gap markers and the note
    chosen: set[int] = set()
    used = 0
    for index, size in enumerate(sizes):
        if used + size > min(HEAD_TOKENS, budget):
            break
        chosen.add(index)
        used += size
    ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
    if not ranked:
        marker = "\n\n[网页内容已截断，仅保留前面部分]"
        return truncate_tokens(content, limit - count_tokens(marker)).rstrip() + marker
    # A heading match is usually followed by the text it introduces, so widen
    # around the best matches first, then further out while budget remains.
    for offsets in ((0, 1, -1), (2, -2, 3, -3), (4, 5, 6, -4)):
        for index in ranked:
            for offset in offsets:
                neighbour = index + offset
                if 0 <= neighbour < len(chunks) and neighbour not in chosen and used + sizes[neighbour] <= budget:
                    chosen.add(neighbour)
                    used += sizes[neighbour]

    parts: list[str] = []
    previous = -1
    for index in sorted(chosen):
        if index != previous + 1:
            parts.append("\n\n[……中间省略……]\n\n")
        parts.append(chunks[index])
        previous = index
    if previous != len(chunks) - 1:
        parts.append("\n\n[……后文省略……]")
    note = (
        f"\n\n[网页共约 {total} token，超过单次读取上限 {limit}；"
        "已保留开头和与 objective 最相关的片段，按原文顺序排列。]"
    )
    return truncate_tokens("".join(parts).strip(), limit - count_tokens(note)) + note
