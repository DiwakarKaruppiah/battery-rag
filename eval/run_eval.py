"""Evaluate the RAG system on your golden set.

  python eval/run_eval.py                 # main config + Ragas
  python eval/run_eval.py --ablation      # compare dense / hybrid × rerank on/off (no Ragas)
  python eval/run_eval.py --skip-ragas    # only our own metrics (fast, no judge cost for Ragas)

Writes eval/results/report.md (paste the table into your README) and results.json.
Metrics:
  retrieval  hit@k (a relevant paper is among the sources), Ragas context precision / recall
  answer     Ragas faithfulness + response relevancy
  citations  validity (cited ids exist) and support (LLM judge: does the cited source support the sentence?)
  guardrail  correct abstention on unanswerable Qs, false abstention on answerable Qs
  latency    p50 / p95 per stage
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mgrag.citations import claims_with_citations  # noqa: E402
from mgrag.config import settings  # noqa: E402
from mgrag.rag import _llm, answer  # noqa: E402

OUT = ROOT / "eval" / "results"
CONFIGS = {
    "dense": {"mode": "dense", "use_rerank": False},
    "dense+rerank": {"mode": "dense", "use_rerank": True},
    "hybrid": {"mode": "hybrid", "use_rerank": False},
    "hybrid+rerank": {"mode": "hybrid", "use_rerank": True},
}


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("//")]


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def citation_support(r) -> float | None:
    """Fraction of (sentence, cited source) pairs that a judge LLM says are supported."""
    pairs = [(s, n) for s, ns in claims_with_citations(r.answer) for n in ns if 1 <= n <= len(r.sources)]
    if not pairs:
        return None
    ok = 0
    for sentence, n in pairs:
        src = r.sources[n - 1]["text"][:3000]
        verdict = _llm([{"role": "user", "content":
                         f"SOURCE:\n{src}\n\nCLAIM:\n{sentence}\n\n"
                         "Is every factual part of the CLAIM supported by the SOURCE? Answer YES or NO only."}],
                       model=settings.judge_model)
        ok += verdict.strip().upper().startswith("YES")
    return ok / len(pairs)


def run_config(golden: list[dict], cfg: dict, judge_citations: bool) -> tuple[list[dict], dict]:
    rows = []
    for g in golden:
        t0 = time.perf_counter()
        r = answer(g["question"], **cfg)
        wall = time.perf_counter() - t0
        rel = set(g.get("relevant_papers", []))
        row = {
            "id": g["id"], "question": g["question"], "answerable": g.get("answerable", True),
            "abstained": r.abstained, "reason": r.reason, "answer": r.answer,
            "hit": bool(rel & {s["paper_id"] for s in r.sources}) if rel else None,
            "n_cited": len(r.cited), "invalid_citations": len(r.invalid_citations),
            "uncited_ratio": r.uncited_ratio, "latency_s": wall, **r.timings,
            "contexts": [s["text"] for s in r.sources], "reference": g.get("reference", ""),
        }
        if judge_citations and not r.abstained:
            row["citation_support"] = citation_support(r)
        rows.append(row)
        print(f"  {g['id']}: {'ABSTAIN ' + r.reason if r.abstained else 'answered'} ({wall:.1f}s)")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    hits = [r["hit"] for r in ans if r["hit"] is not None]
    cites = [r for r in rows if not r["abstained"]]
    support = [r["citation_support"] for r in cites if r.get("citation_support") is not None]
    lat = [r["latency_s"] for r in rows]
    summary = {
        "n": len(rows),
        "hit@k": sum(hits) / len(hits) if hits else None,
        "correct_abstention": sum(r["abstained"] for r in unans) / len(unans) if unans else None,
        "false_abstention": sum(r["abstained"] for r in ans) / len(ans) if ans else None,
        "citation_validity": (1 - sum(r["invalid_citations"] for r in cites) /
                              max(1, sum(r["n_cited"] + r["invalid_citations"] for r in cites))) if cites else None,
        "citation_support": statistics.mean(support) if support else None,
        "uncited_sentences": statistics.mean(r["uncited_ratio"] for r in cites) if cites else None,
        "latency_p50_s": pct(lat, 0.5), "latency_p95_s": pct(lat, 0.95),
    }
    return rows, summary


def run_ragas(rows: list[dict]) -> dict:
    """Ragas 0.4.x. API names move between versions — pin ragas==0.4.3 (see pyproject)."""
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    try:
        from langchain_litellm import ChatLiteLLM
    except ImportError:
        from langchain_community.chat_models import ChatLiteLLM
    from langchain_huggingface import HuggingFaceEmbeddings

    data = [{"user_input": r["question"], "retrieved_contexts": r["contexts"],
             "response": r["answer"], "reference": r["reference"]}
            for r in rows if r["answerable"] and not r["abstained"] and r["reference"]
            and not r["reference"].startswith("TODO")]
    if not data:
        print("  (Ragas skipped: no answered questions with a written reference)")
        return {}
    llm = LangchainLLMWrapper(ChatLiteLLM(model=settings.judge_model, temperature=0))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.dense_model))
    res = evaluate(EvaluationDataset.from_list(data),
                   metrics=[Faithfulness(), ResponseRelevancy(),
                            LLMContextPrecisionWithReference(), LLMContextRecall()],
                   llm=llm, embeddings=emb)
    df = res.to_pandas()
    num = df.select_dtypes("number")
    return {c: float(num[c].mean()) for c in num.columns}


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden_set.jsonl"))
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--skip-ragas", action="store_true")
    ap.add_argument("--no-citation-judge", action="store_true")
    args = ap.parse_args()

    golden = load_golden(Path(args.golden))
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(CONFIGS) if args.ablation else ["hybrid+rerank"]
    report, all_rows = {}, {}
    for name in names:
        print(f"\n== {name} ==")
        rows, summary = run_config(golden, CONFIGS[name],
                                   judge_citations=not args.no_citation_judge and not args.ablation)
        if not args.skip_ragas and not args.ablation:
            try:
                summary.update(run_ragas(rows))
            except Exception as e:  # never lose the run because the judge failed
                print(f"  Ragas failed: {e}")
        report[name], all_rows[name] = summary, rows

    keys = sorted({k for s in report.values() for k in s})
    lines = [f"# Evaluation — {time.strftime('%Y-%m-%d')}",
             (f"LLM `{settings.llm_model}` · judge `{settings.judge_model}` · "
              f"reranker `{settings.reranker_model}` · {len(golden)} questions"), "",
             "| metric | " + " | ".join(report) + " |", "|---|" + "---|" * len(report)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(fmt(report[n].get(k)) for n in report) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps({"summary": report, "rows": all_rows}, indent=2, default=str))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
