"""Pure-Python helpers for citations (no heavy imports, so they are unit-tested in CI)."""
from __future__ import annotations

import re

_CITE_BLOCK = re.compile(r"\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Reasoning models (e.g. Qwen3 via Ollama) may emit <think>…</think>; drop it."""
    return _THINK.sub("", text).strip()


def parse_citations(text: str) -> list[int]:
    """'… 99.5% [S1][S3, S4]' → [1, 3, 4] (ordered, unique)."""
    seen: list[int] = []
    for block in _CITE_BLOCK.findall(text):
        for num in re.findall(r"\d+", block):
            n = int(num)
            if n not in seen:
                seen.append(n)
    return seen


def split_sentences(text: str) -> list[str]:
    lines = [ln.strip(" -*•\t") for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    for ln in lines:
        out.extend(s.strip() for s in _SENT_SPLIT.split(ln) if s.strip())
    return out


def claims_with_citations(text: str) -> list[tuple[str, list[int]]]:
    """Each sentence with the source numbers it cites (empty list = uncited)."""
    return [(s, parse_citations(s)) for s in split_sentences(text)]


def uncited_ratio(text: str, min_words: int = 6) -> float:
    """Share of substantive sentences that carry no citation (lower is better)."""
    claims = [(s, c) for s, c in claims_with_citations(text) if len(s.split()) >= min_words]
    if not claims:
        return 0.0
    return sum(1 for _, c in claims if not c) / len(claims)


def validate(cited: list[int], n_sources: int) -> tuple[list[int], list[int]]:
    """Split cited numbers into (valid, invalid) given how many sources were shown."""
    valid = [i for i in cited if 1 <= i <= n_sources]
    invalid = [i for i in cited if not 1 <= i <= n_sources]
    return valid, invalid
