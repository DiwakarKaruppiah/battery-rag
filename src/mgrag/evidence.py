"""Research features built on the property database (data/properties.csv):

1. Evidence spread — where papers report very different values for the same
   material + property. Often the reason is conditions (current density,
   temperature, cell type), sometimes it is a real disagreement. Either way a
   researcher wants to see it, with both sources side by side.
2. Coverage / gap map — which component × property combinations your library
   has measured a lot, and which almost never: candidate research gaps.
"""
from __future__ import annotations

import pandas as pd

from .citations import strip_thinking
from .config import settings

# Absolute spread (in canonical units) that counts as notable; otherwise relative spread is used
ABS_SPREAD = {"coulombic_efficiency": 1.0, "average_voltage": 0.2, "anodic_stability": 0.3}
REL_SPREAD = 0.5  # max/min − 1 ≥ 50 %


def load_properties() -> pd.DataFrame:
    p = settings.data_dir / "properties.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def find_spreads(df: pd.DataFrame, min_papers: int = 2) -> list[dict]:
    """Groups (material, component, property) reported by ≥ min_papers papers whose
    normalized values differ notably. Sorted by size of the spread."""
    need = {"material_key", "component", "property", "value_norm", "paper_id"}
    if df.empty or not need <= set(df.columns):
        return []
    d = df.dropna(subset=["value_norm"])
    out = []
    for (mat, comp, prop), g in d.groupby(["material_key", "component", "property"]):
        if g["paper_id"].nunique() < min_papers:
            continue
        lo, hi = float(g["value_norm"].min()), float(g["value_norm"].max())
        if prop in ABS_SPREAD:
            notable, size = hi - lo >= ABS_SPREAD[prop], (hi - lo) / ABS_SPREAD[prop]
        else:
            rel = (hi - lo) / abs(lo) if lo else float("inf")
            notable, size = rel >= REL_SPREAD, rel / REL_SPREAD
        if notable:
            out.append({
                "material": g["material"].iloc[0], "material_key": mat, "component": comp,
                "property": prop, "unit": g["unit_norm"].iloc[0], "min": lo, "max": hi,
                "n_papers": int(g["paper_id"].nunique()), "size": float(size),
                "rows": g.sort_values("value_norm").to_dict("records"),
            })
    return sorted(out, key=lambda s: s["size"], reverse=True)


def coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct papers per component × property (0 = never measured in your library)."""
    if df.empty:
        return pd.DataFrame()
    m = df.pivot_table(index="component", columns="property", values="paper_id",
                       aggfunc=pd.Series.nunique, fill_value=0)
    return m.astype(int)


def explain_spread(spread: dict) -> str:
    """Ask the LLM why the values differ, using ONLY the source passages. Cited."""
    import litellm

    from .parse import load_all_chunks

    by_id = {c["id"]: c for c in load_all_chunks()}
    rows = spread["rows"][:6]
    passages = "\n\n".join(
        f"[P{i}] {r['paper_id']} p.{r.get('page')}: reports {r['value']} {r['unit']} "
        f"({r.get('conditions') or 'conditions not stated'})\n{by_id.get(r['chunk_id'], {}).get('text', '')[:1500]}"
        for i, r in enumerate(rows, 1))
    prompt = (
        f"Papers report {spread['property'].replace('_', ' ')} of {spread['material']} ({spread['component']}) "
        f"from {spread['min']:.3g} to {spread['max']:.3g} {spread['unit']}.\n"
        "Using ONLY the passages, explain the most likely reasons (current density, temperature, "
        "electrolyte, loading, cell configuration, cycle number, measured vs theoretical). "
        "Cite passages as [P#]. Say clearly if it looks like a genuine disagreement. Max 150 words.\n\n"
        + passages)
    resp = litellm.completion(model=settings.llm_model, temperature=0,
                              messages=[{"role": "user", "content": prompt}])
    return strip_thinking(resp.choices[0].message.content or "")


def suggest_gaps(df: pd.DataFrame, n: int = 3) -> str:
    """Propose research gaps grounded in the coverage matrix and the measured values."""
    import litellm

    matrix = coverage_matrix(df)
    if matrix.empty:
        return "Build the property database first: python -m mgrag.cli extract"
    sample = df.dropna(subset=["value_norm"]).sort_values("year", ascending=False).head(40)
    lines = "\n".join(f"- {r.material} | {r.component} | {r.property} = {r.value_norm:.3g} {r.unit_norm} "
                      f"| {r.conditions} | {r.paper_id} p.{r.page}" for r in sample.itertuples())
    prompt = (
        "You help a magnesium-battery researcher find gaps in THEIR literature collection.\n"
        f"Papers per component × property (0 = never reported):\n{matrix.to_string()}\n\n"
        f"Recent measurements:\n{lines}\n\n"
        f"Propose {n} specific, testable research gaps. For each: the gap in one sentence, the evidence "
        "from the table above that points to it (cite paper_id p.page), and one experiment to close it. "
        "These are hypotheses for the researcher to judge, not facts; do not invent measurements.")
    resp = litellm.completion(model=settings.llm_model, temperature=0.2,
                              messages=[{"role": "user", "content": prompt}])
    return strip_thinking(resp.choices[0].message.content or "")
