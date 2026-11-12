"""Multimodal step: ask a vision LLM to describe each figure (axes, trends, numbers),
so a question like 'which electrolyte showed the lowest overpotential in the
voltage profile?' can be answered from a plot, not only from text."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from .citations import strip_thinking
from .config import settings
from .parse import chunk_file

PROMPT = """This figure is from a magnesium-battery research paper. Caption: "{caption}"
Describe what is plotted for a scientist who cannot see it, in under 120 words:
plot type, axes with units, each series/material, the key trend, and any readable
numbers (capacity, voltage, overpotential, cycle number, efficiency). No speculation."""


def describe_figure(image_path: str, caption: str) -> str:
    import litellm

    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    resp = litellm.completion(
        model=settings.vision_model, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(caption=caption[:500])},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    return strip_thinking(resp.choices[0].message.content or "")


def describe_all(force: bool = False) -> None:
    for fp in sorted(settings.chunks_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in fp.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = False
        for r in rows:
            if r["type"] != "figure" or not r.get("image_path"):
                continue
            if r.get("description") and not force:
                continue
            try:
                r["description"] = describe_figure(r["image_path"], r["text"])
                changed = True
                print(f"  described {r['id']}")
            except Exception as e:
                print(f"  ! {r['id']}: {e}")
        if changed:
            paper_id = rows[0]["paper_id"]
            with chunk_file(paper_id).open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
