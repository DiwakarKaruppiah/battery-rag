"""The answer pipeline: hybrid retrieve → rerank → guardrail → generate → verify citations."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import store
from .citations import parse_citations, strip_thinking, uncited_ratio, validate
from .config import settings

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You are a research assistant for {settings.domain} literature.
Rules:
1. Use ONLY the numbered sources provided. No outside knowledge.
2. End every factual sentence with its citation(s), e.g. [S2] or [S1][S3].
3. Copy numbers and units exactly as the sources give them (mAh g-1, %, mV, mA cm-2, cycles).
4. If sources disagree, say so and cite each side.
5. If the sources do not contain the answer, reply with exactly: {ABSTAIN_TOKEN}
Be concise and technical; the reader is an electrochemist."""


@dataclass
class RagResult:
    question: str
    answer: str
    sources: list[dict]
    cited: list[int] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""
    uncited_ratio: float = 0.0
    invalid_citations: list[int] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [self.answer, "", "**Sources**"]
        shown = self.sources if self.abstained else [self.sources[i - 1] for i in self.cited]
        for i, s in enumerate(shown, 1):
            n = self.cited[i - 1] if not self.abstained else i
            lines.append(f"- [S{n}] {source_label(s)}")
        return "\n".join(lines)


def source_label(s: dict) -> str:
    who = f"{s.get('authors') or s.get('paper_id')} ({s.get('year') or 'n.d.'})"
    kind = {"table": "table", "figure": "figure"}.get(s.get("type"), "text")
    doi = f" doi:{s['doi']}" if s.get("doi") else ""
    return f"{who}, \"{s.get('title', '')[:80]}\", p.{s.get('page')}, {kind}{doi}"


def format_sources(sources: list[dict]) -> str:
    blocks = []
    for i, s in enumerate(sources, 1):
        body = s["text"]
        if s.get("description"):
            body += f"\n[Figure description] {s['description']}"
        blocks.append(f"[S{i}] {source_label(s)}\n{body}")
    return "\n\n---\n\n".join(blocks)


def _llm(messages: list[dict], model: str | None = None) -> str:
    import litellm

    resp = litellm.completion(model=model or settings.llm_model, messages=messages, temperature=0)
    return strip_thinking(resp.choices[0].message.content or "")


def answer(question: str, *, mode: str = "hybrid", use_rerank: bool = True,
           min_score: float | None = None, top_k: int | None = None) -> RagResult:
    threshold = settings.min_rerank_score if min_score is None else min_score
    n_ctx = top_k or settings.top_k_context
    t = {}

    t0 = time.perf_counter()
    candidates = store.search(question, k=settings.top_k_retrieve, mode=mode)
    t["retrieve_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if use_rerank:
        top = store.rerank(question, candidates, n_ctx)
        best = top[0]["rerank_score"] if top else 0.0
    else:
        top, best = candidates[:n_ctx], 1.0  # no gate without a calibrated score
    t["rerank_s"] = time.perf_counter() - t0

    # Guardrail 1 — retrieval gate: nothing relevant enough in the library
    if not top or best < threshold:
        t["total_s"] = sum(t.values())
        return RagResult(question, _abstain_text(top), top[:3], abstained=True,
                         reason=f"retrieval_gate (best={best:.2f} < {threshold})", timings=t)

    t0 = time.perf_counter()
    text = _llm([{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": f"Sources:\n\n{format_sources(top)}\n\nQuestion: {question}"}])
    t["generate_s"] = time.perf_counter() - t0
    t["total_s"] = sum(t.values())

    # Guardrail 2 — the model itself says the evidence is missing
    if ABSTAIN_TOKEN in text:
        return RagResult(question, _abstain_text(top), top[:3], abstained=True,
                         reason="llm_declined", timings=t)

    # Guardrail 3 — an answer with no valid citation is not allowed out
    valid, invalid = validate(parse_citations(text), len(top))
    if not valid:
        return RagResult(question, _abstain_text(top), top[:3], abstained=True,
                         reason="no_valid_citations", timings=t)

    return RagResult(question, text, top, cited=valid, invalid_citations=invalid,
                     uncited_ratio=uncited_ratio(text), timings=t)


def _abstain_text(closest: list[dict]) -> str:
    msg = "I could not find enough evidence in your library to answer this."
    if closest:
        papers = "; ".join(dict.fromkeys(f"{c.get('paper_id')} p.{c.get('page')}" for c in closest[:3]))
        msg += f" Closest passages: {papers}."
    return msg
