"""Find the PDFs to ingest: from a Zotero collection, or from a plain folder."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import settings


@dataclass
class PaperRef:
    paper_id: str      # short, human-readable id used in citations, e.g. "Zhang2023-ab12"
    pdf_path: str
    title: str
    authors: str
    year: str
    doi: str = ""
    zotero_key: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", text)[:20] or "paper"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:4]


def from_folder(folder: str | Path) -> list[PaperRef]:
    """Fallback: every PDF in a folder (metadata = file name)."""
    refs = []
    for pdf in sorted(Path(folder).expanduser().glob("**/*.pdf")):
        refs.append(PaperRef(
            paper_id=f"{_slug(pdf.stem)}-{_short_hash(str(pdf))}",
            pdf_path=str(pdf), title=pdf.stem, authors="", year="",
        ))
    return refs


def from_zotero(collection_name: str) -> list[PaperRef]:
    """Read a Zotero collection through the Zotero 7 local API (Zotero must be open)."""
    from pyzotero import zotero

    zot = zotero.Zotero(library_id="0", library_type="user", local=True)
    collections = zot.everything(zot.collections())
    match = [c for c in collections if c["data"]["name"].lower() == collection_name.lower()]
    if not match:
        names = ", ".join(sorted(c["data"]["name"] for c in collections))
        raise SystemExit(f"Collection '{collection_name}' not found. Available: {names}")

    refs: list[PaperRef] = []
    for item in zot.everything(zot.collection_items_top(match[0]["key"])):
        data = item["data"]
        if data.get("itemType") in {"attachment", "note", "annotation"}:
            continue
        pdf_path = _find_pdf(zot, item["key"])
        if pdf_path is None:
            print(f"  ! no PDF for: {data.get('title', '')[:70]}")
            continue
        creators = [c.get("lastName") or c.get("name", "") for c in data.get("creators", [])]
        year_match = re.search(r"\d{4}", data.get("date", ""))
        year = year_match.group(0) if year_match else ""
        first = _slug(creators[0]) if creators else "Anon"
        refs.append(PaperRef(
            paper_id=f"{first}{year}-{item['key'][:4].lower()}",
            pdf_path=str(pdf_path),
            title=data.get("title", ""),
            authors=", ".join(creators[:3]) + (" et al." if len(creators) > 3 else ""),
            year=year,
            doi=data.get("DOI", ""),
            zotero_key=item["key"],
        ))
    return refs


def _find_pdf(zot, parent_key: str) -> Path | None:
    for child in zot.children(parent_key):
        d = child["data"]
        if d.get("contentType") != "application/pdf":
            continue
        if d.get("linkMode") == "linked_file" and d.get("path"):
            p = Path(d["path"])
        else:
            p = settings.zotero_storage / child["key"] / d.get("filename", "")
        if p.is_file():
            return p
    return None
