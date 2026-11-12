"""Numeric questions go to the property database, not to text search.

'Top 5 cathodes by specific capacity after 100 cycles' is a ranking over numbers.
Text retrieval answers it badly; SQL over the extracted measurements answers it
exactly, and every returned row still carries paper + page. The LLM writes a
read-only DuckDB query; a validator blocks anything that is not a single SELECT.
"""
from __future__ import annotations

import re

import pandas as pd

from .citations import strip_thinking
from .config import settings
from .normalize import UNITS

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|pragma|install|load|"
    r"call|set|read_csv\w*|read_parquet|read_json\w*|read_text|glob|httpfs)\b", re.IGNORECASE)

_RANKING = re.compile(
    r"\b(top|highest|lowest|best|worst|rank\w*|compare|average|mean|median|how many|list all|"
    r"maximum|minimum|above|below|greater|less|more than)\b|[<>≥≤]", re.IGNORECASE)
_PROPERTY = re.compile(
    r"capacity|coulombic|\bCE\b|efficien|overpotential|cycle|conductivit|voltage|stability|energy density",
    re.IGNORECASE)

SCHEMA = """measurements(
  material TEXT, material_key TEXT, component TEXT, property TEXT,
  value DOUBLE, unit TEXT, value_norm DOUBLE, unit_norm TEXT, conditions TEXT,
  paper_id TEXT, title TEXT, year TEXT, doi TEXT, page INTEGER, chunk_id TEXT)"""


def route(question: str) -> str:
    """'sql' for ranking / filtering / aggregation over a property, else 'rag'."""
    return "sql" if _RANKING.search(question) and _PROPERTY.search(question) else "rag"


def is_safe(sql: str) -> bool:
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        return False
    if not re.match(r"(?is)^(select|with)\b", s):
        return False
    return not _FORBIDDEN.search(s)


def _extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


def write_sql(question: str, df: pd.DataFrame) -> str:
    import litellm

    components = sorted(df["component"].dropna().unique().tolist()) if "component" in df else []
    units = {p: u for p, (u, _) in UNITS.items()}
    prompt = f"""Write ONE DuckDB SQL SELECT query for the question, over this table:
{SCHEMA}
Values: property in {sorted(units)}; component in {components}.
Compare and sort with value_norm (canonical units: {units}); match materials with ILIKE on material.
Always return material, component, property, value_norm, unit_norm, conditions, paper_id, page.
Use LIMIT 50 or less. Return only the SQL in a ```sql block.

Question: {question}"""
    resp = litellm.completion(model=settings.llm_model, temperature=0,
                              messages=[{"role": "user", "content": prompt}])
    return _extract_sql(strip_thinking(resp.choices[0].message.content or ""))


def ask_sql(question: str, df: pd.DataFrame | None = None) -> tuple[str, pd.DataFrame]:
    import duckdb

    from .evidence import load_properties

    df = load_properties() if df is None else df
    if df.empty:
        raise RuntimeError("No property database yet: run `python -m mgrag.cli extract`")
    sql = write_sql(question, df)
    if not is_safe(sql):
        raise ValueError(f"Refused unsafe or invalid SQL:\n{sql}")
    con = duckdb.connect(database=":memory:")
    con.register("measurements", df)
    return sql, con.execute(sql).df()
