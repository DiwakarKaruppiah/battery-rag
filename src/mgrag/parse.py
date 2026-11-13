"""PDF → structure-aware chunks (text, tables, figures) using Docling.

Each paper becomes data/chunks/<paper_id>.jsonl. Every chunk keeps its page number,
section heading and type, which is what makes page-level citations possible.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import settings
from .sources import PaperRef

SKIP_SECTIONS = ("references", "acknowledg", "author contributions", "conflict", "supporting information")
MAX_TABLE_CHARS = 6000


def make_converter(ocr: bool = False):
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = ocr                     # journal PDFs are born-digital; turn on for scans
    opts.do_table_structure = True        # TableFormer model → real rows/columns
    opts.generate_picture_images = True   # keep figure bitmaps so we can show / describe them
    opts.images_scale = 2.0
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
        pipeline_options=opts, backend=PyPdfiumDocumentBackend)})


def make_chunker():
    """Docling's HybridChunker: splits by document structure, then by token budget.
    The tokenizer matches the embedding model so chunks never overflow it."""
    from docling.chunking import HybridChunker

    try:  # docling-core >= 2.8 API
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        tok = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(settings.dense_model),
            max_tokens=settings.max_chunk_tokens,
        )
        return HybridChunker(tokenizer=tok, merge_peers=True)
    except ImportError:  # older API
        return HybridChunker(tokenizer=settings.dense_model,
                             max_tokens=settings.max_chunk_tokens, merge_peers=True)


def _label(item) -> str:
    return getattr(item.label, "value", str(item.label)).lower()


def _page(item) -> int | None:
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def parse_paper(ref: PaperRef, converter, chunker) -> list[dict]:
    doc = converter.convert(ref.pdf_path).document
    base = {k: v for k, v in ref.to_dict().items() if k != "pdf_path"}
    chunks: list[dict] = []

    # Human-readable copy for you to inspect what the parser saw
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    (settings.parsed_dir / f"{ref.paper_id}.md").write_text(doc.export_to_markdown(), encoding="utf-8")

    # 1) Text chunks (structure-aware)
    for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
        items = ch.meta.doc_items
        if items and all(_label(it) == "table" for it in items):
            continue  # tables are indexed separately below, whole
        headings = " > ".join(ch.meta.headings or [])
        if any(s in headings.lower() for s in SKIP_SECTIONS):
            continue
        pages = [p for p in (_page(it) for it in items) if p]
        chunks.append({**base, "id": f"{ref.paper_id}::t{i}", "type": "text",
                       "section": headings, "page": min(pages) if pages else None,
                       "text": chunker.contextualize(chunk=ch)})

    # 2) Tables as markdown, one chunk each, with caption
    for j, table in enumerate(doc.tables):
        md = table.export_to_markdown(doc=doc)
        if not md.strip():
            continue
        caption = table.caption_text(doc) or ""
        text = f"Table {j + 1}. {caption}\n\n{md}"[:MAX_TABLE_CHARS]
        chunks.append({**base, "id": f"{ref.paper_id}::tab{j + 1}", "type": "table",
                       "section": caption[:120], "page": _page(table), "text": text})

    # 3) Figures: caption now; a vision-LLM description can be added later (cli describe-figures)
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    for k, pic in enumerate(doc.pictures):
        caption = pic.caption_text(doc) or ""
        if len(caption) < 20:  # logos, TOC graphics, decorations
            continue
        img_path = ""
        img = pic.get_image(doc)
        if img is not None:
            p = settings.figures_dir / f"{ref.paper_id}_fig{k + 1}.png"
            img.save(p)
            img_path = str(p)
        chunks.append({**base, "id": f"{ref.paper_id}::fig{k + 1}", "type": "figure",
                       "section": caption[:120], "page": _page(pic), "image_path": img_path,
                       "description": "", "text": f"Figure. {caption}"})
    return chunks


def chunk_file(paper_id: str) -> Path:
    return settings.chunks_dir / f"{paper_id}.jsonl"


def write_chunks(paper_id: str, chunks: list[dict]) -> None:
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    with chunk_file(paper_id).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_all_chunks() -> list[dict]:
    out = []
    for fp in sorted(settings.chunks_dir.glob("*.jsonl")):
        with fp.open(encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def parse_all(refs: list[PaperRef], force: bool = False, ocr: bool = False) -> None:
    converter, chunker = make_converter(ocr), make_chunker()
    for n, ref in enumerate(refs, 1):
        if chunk_file(ref.paper_id).exists() and not force:
            print(f"[{n}/{len(refs)}] skip (already parsed) {ref.paper_id}")
            continue
        try:
            chunks = parse_paper(ref, converter, chunker)
            write_chunks(ref.paper_id, chunks)
            counts = {t: sum(c["type"] == t for c in chunks) for t in ("text", "table", "figure")}
            print(f"[{n}/{len(refs)}] {ref.paper_id}: {counts}")
        except Exception as e:  # one bad PDF must not stop the batch
            print(f"[{n}/{len(refs)}] FAILED {ref.paper_id}: {e}")
