"""Expose MgScholar as an MCP server, so any AI agent (Claude Desktop, Claude Code,
Cursor, ...) can use your Mg-battery library as a set of tools.

Run:  python -m mgrag.mcp_server
Claude Desktop config (claude_desktop_config.json):
  {"mcpServers": {"mg-battery-rag": {"command": "python", "args": ["-m", "mgrag.mcp_server"],
                                     "cwd": "/path/to/mg-battery-rag"}}}
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import store
from .evidence import find_spreads, load_properties
from .rag import answer, source_label
from .sqltool import ask_sql

mcp = FastMCP("mg-battery-rag")


@mcp.tool()
def ask_mg_literature(question: str) -> str:
    """Answer a question about magnesium-battery research from the user's Zotero library.
    Returns a cited answer, or says the library lacks the evidence."""
    return answer(question).to_markdown()


@mcp.tool()
def search_mg_literature(query: str, k: int = 8, only: str = "") -> list[dict]:
    """Search passages, tables or figures (only = 'text' | 'table' | 'figure' | '')."""
    hits = store.search(query, k=k, types=[only] if only else None)
    return [{"source": source_label(h), "type": h["type"], "text": h["text"][:800]} for h in hits]


@mcp.tool()
def query_measurements(question: str) -> dict:
    """Numeric questions (rank / filter / average capacities, efficiencies, overpotentials...)
    answered by SQL over measurements extracted from the papers. Rows include paper and page."""
    sql, df = ask_sql(question)
    return {"sql": sql, "rows": df.head(50).to_dict("records")}


@mcp.tool()
def where_papers_disagree(property_name: str = "", limit: int = 10) -> list[dict]:
    """Materials whose reported values differ notably between papers (optionally one property,
    e.g. 'coulombic_efficiency'). Each item lists the values, conditions and pages."""
    spreads = find_spreads(load_properties())
    if property_name:
        spreads = [s for s in spreads if s["property"] == property_name]
    keep = ("value", "unit", "conditions", "paper_id", "page")
    return [{**{k: s[k] for k in ("material", "component", "property", "unit", "min", "max", "n_papers")},
             "reports": [{k: r.get(k) for k in keep} for r in s["rows"]]} for s in spreads[:limit]]


if __name__ == "__main__":
    mcp.run()
