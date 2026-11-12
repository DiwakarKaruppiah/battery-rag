"""Novel feature: turn the literature into a citable property database.

For every paper, numeric passages and tables are sent to an LLM that returns
structured measurements (material, component, property, value, unit, conditions).
Each row keeps paper + page + chunk id, so every number in the database is traceable.
Output: data/properties.csv — explored in the Streamlit 'Property explorer' tab."""
from __future__ import annotations

import json
import re

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

from .citations import strip_thinking
from .config import settings
from .normalize import material_key, normalize_value
from .parse import load_all_chunks

PROPERTIES = ["specific_capacity", "areal_capacity", "coulombic_efficiency", "overpotential", "cycle_life",
              "ionic_conductivity", "anodic_stability", "average_voltage", "energy_density",
              "rate_capability", "other"]
COMPONENTS = ["anode", "cathode", "electrolyte", "separator", "interphase", "full_cell", "other"]

NUMERIC_HINT = re.compile(
    r"mAh\s*g|mA\s*h\s*g|coulombic|CE\b|overpotential|mV|cycles?\b|S\s*cm|mS|V vs|Wh\s*kg|%", re.IGNORECASE)


class Measurement(BaseModel):
    material: str
    component: str = "other"
    property: str = "other"
    value: float
    unit: str
    conditions: str = ""
    source: int  # the [S#] number of the passage it came from

    @field_validator("component")
    @classmethod
    def _comp(cls, v: str) -> str:
        v = v.lower().strip()
        return v if v in COMPONENTS else "other"

    @field_validator("property")
    @classmethod
    def _prop(cls, v: str) -> str:
        v = v.lower().strip().replace(" ", "_")
        return v if v in PROPERTIES else "other"


PROMPT = """Extract every quantitative electrochemical measurement from the passages below
(magnesium-battery paper). Return ONLY a JSON array; each element:
{{"material": str (chemical formula where possible, e.g. "Mo6S8", "Mg(TFSI)2-MgCl2/DME"),
  "component": one of {components}, "property": one of {properties},
  "value": number, "unit": str, "conditions": str (current density, temperature, cycle no., etc.),
  "source": the passage number N from [PN]}}
Only values stated explicitly. If none, return [].

{passages}"""


def _parse_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def extract_paper(chunks: list[dict], max_passages: int = 8) -> list[dict]:
    import litellm

    cand = [c for c in chunks if c["type"] == "table" or NUMERIC_HINT.search(c["text"])]
    rows: list[dict] = []
    for start in range(0, len(cand), max_passages):
        batch = cand[start:start + max_passages]
        passages = "\n\n".join(f"[P{i}] (p.{c.get('page')}) {c['text'][:2500]}" for i, c in enumerate(batch, 1))
        prompt = PROMPT.format(components=COMPONENTS, properties=PROPERTIES, passages=passages)
        resp = litellm.completion(model=settings.llm_model, temperature=0,
                                  messages=[{"role": "user", "content": prompt}])
        for raw in _parse_json_array(strip_thinking(resp.choices[0].message.content or "")):
            try:
                m = Measurement(**raw)
            except (ValidationError, TypeError):
                continue
            if not 1 <= m.source <= len(batch):
                continue
            src = batch[m.source - 1]
            rows.append({**m.model_dump(exclude={"source"}),
                         "paper_id": src["paper_id"], "title": src.get("title", ""),
                         "year": src.get("year", ""), "doi": src.get("doi", ""),
                         "page": src.get("page"), "chunk_id": src["id"]})
    return rows


def extract_all() -> pd.DataFrame:
    chunks = load_all_chunks()
    by_paper: dict[str, list[dict]] = {}
    for c in chunks:
        by_paper.setdefault(c["paper_id"], []).append(c)
    rows: list[dict] = []
    for n, (pid, cs) in enumerate(by_paper.items(), 1):
        got = extract_paper(cs)
        print(f"[{n}/{len(by_paper)}] {pid}: {len(got)} measurements")
        rows.extend(got)
    df = pd.DataFrame(rows)
    if not df.empty:
        # comparable numbers + loose material identity → enables ranking, spreads, gap map
        df["material_key"] = df["material"].map(material_key)
        norm = [normalize_value(p, v, u) for p, v, u in zip(df["property"], df["value"], df["unit"], strict=True)]
        df["value_norm"] = [n[0] for n in norm]
        df["unit_norm"] = [n[1] for n in norm]
        print(f"normalized {df['value_norm'].notna().sum()}/{len(df)} values to canonical units")
    out = settings.data_dir / "properties.csv"
    df.to_csv(out, index=False)
    print(f"saved {len(df)} rows → {out}")
    return df
