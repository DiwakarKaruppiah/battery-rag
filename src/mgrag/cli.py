"""Command line:  python -m mgrag.cli <command>

  list     --zotero "Mg battery" | --folder PATH     show which PDFs would be ingested
  parse    --zotero "Mg battery" | --folder PATH     PDF → chunks (text/tables/figures)
  index                                              chunks → Qdrant hybrid index
  ask      "question"                                answer with citations
  search   "query" [--mode hybrid|dense|sparse]      inspect raw retrieval
  describe-figures                                   vision-LLM figure descriptions (then re-run index)
  extract                                            build data/properties.csv (normalized units)
  sql      "numeric question"                        rank / filter measurements with SQL
  spreads  [--property coulombic_efficiency]         where papers report notably different values
  gaps                                               coverage map + suggested research gaps
"""
from __future__ import annotations

import argparse
import json

from .config import settings


def _refs(args):
    from .sources import from_folder, from_zotero
    if args.zotero:
        return from_zotero(args.zotero)
    if args.folder:
        return from_folder(args.folder)
    raise SystemExit("give --zotero COLLECTION or --folder PATH")


def main() -> None:
    p = argparse.ArgumentParser(prog="mgrag")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "parse"):
        sp = sub.add_parser(name)
        sp.add_argument("--zotero")
        sp.add_argument("--folder")
        sp.add_argument("--limit", type=int, default=0)
        if name == "parse":
            sp.add_argument("--force", action="store_true")
            sp.add_argument("--ocr", action="store_true")
    sub.add_parser("index")
    sp = sub.add_parser("ask")
    sp.add_argument("question")
    sp.add_argument("--json", action="store_true")
    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--rerank", action="store_true")
    sp = sub.add_parser("describe-figures")
    sp.add_argument("--force", action="store_true")
    sub.add_parser("extract")
    sp = sub.add_parser("sql")
    sp.add_argument("question")
    sp = sub.add_parser("spreads")
    sp.add_argument("--property", default="")
    sp.add_argument("--explain", action="store_true")
    sub.add_parser("gaps")
    args = p.parse_args()

    if args.cmd in ("list", "parse"):
        refs = _refs(args)
        if args.limit:
            refs = refs[: args.limit]
        if args.cmd == "list":
            for r in refs:
                print(f"{r.paper_id:28s} {r.year:5s} {r.title[:80]}")
            print(f"\n{len(refs)} papers")
        else:
            from .parse import parse_all
            parse_all(refs, force=args.force, ocr=args.ocr)

    elif args.cmd == "index":
        from .parse import load_all_chunks
        from .store import build_index
        chunks = load_all_chunks()
        print(f"indexing {len(chunks)} chunks into '{settings.collection}'")
        build_index(chunks)

    elif args.cmd == "ask":
        from .rag import answer
        r = answer(args.question)
        if args.json:
            print(json.dumps({k: v for k, v in r.__dict__.items() if k != "sources"}, indent=2))
        else:
            print(r.to_markdown())
            if r.abstained:
                print(f"\n(abstained: {r.reason})")
            print("\n" + "  ".join(f"{k}={v:.2f}s" for k, v in r.timings.items()))

    elif args.cmd == "search":
        from .store import rerank, search
        hits = search(args.query, k=30 if args.rerank else args.k, mode=args.mode)
        if args.rerank:
            hits = rerank(args.query, hits, args.k)
        for h in hits:
            score = h.get("rerank_score", h["retrieval_score"])
            print(f"{score:.3f}  [{h['type']:6s}] {h['paper_id']} p.{h.get('page')}  {h['text'][:110]!r}")

    elif args.cmd == "describe-figures":
        from .figures import describe_all
        describe_all(force=args.force)

    elif args.cmd == "extract":
        from .extract import extract_all
        extract_all()

    elif args.cmd == "sql":
        from .sqltool import ask_sql
        sql, df = ask_sql(args.question)
        print(sql, "\n")
        print(df.to_string(index=False) if not df.empty else "(no rows)")

    elif args.cmd == "spreads":
        from .evidence import explain_spread, find_spreads, load_properties
        spreads = find_spreads(load_properties())
        if args.property:
            spreads = [s for s in spreads if s["property"] == args.property]
        for s in spreads[:20]:
            print(f"\n{s['material']} | {s['component']} | {s['property']}: "
                  f"{s['min']:.3g}-{s['max']:.3g} {s['unit']} ({s['n_papers']} papers)")
            for r in s["rows"]:
                cond = str(r.get("conditions"))[:50]
                print(f"   {r['value']} {r['unit']:10s} {cond:50s} {r['paper_id']} p.{r.get('page')}")
            if args.explain:
                print("   →", explain_spread(s).replace("\n", "\n     "))
        if not spreads:
            print("no notable spreads (run extract first, or the library agrees!)")

    elif args.cmd == "gaps":
        from .evidence import coverage_matrix, load_properties, suggest_gaps
        df = load_properties()
        print(coverage_matrix(df).to_string(), "\n")
        print(suggest_gaps(df))


if __name__ == "__main__":
    main()
