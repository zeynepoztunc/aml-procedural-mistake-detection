from __future__ import annotations

import sys
import re
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pragma: no cover
    pdfminer_extract_text = None


def normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> None:
    _configure_stdout()

    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "AML-2025_Mistake_Detection_Project.pdf"
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = repo_root / "tools" / "_extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pypdf is fast, but some PDFs fail extraction due to font metadata issues.
    # We fall back to pdfminer.six which is slower but more robust.
    pages_text: list[str] = []
    num_pages: int | None = None
    try:
        if PdfReader is None:
            raise ImportError("pypdf is not installed")
        reader = PdfReader(str(pdf_path), strict=False)  # type: ignore[misc]
        num_pages = len(reader.pages)  # type: ignore[union-attr]
        for i, page in enumerate(reader.pages):  # type: ignore[union-attr]
            text = page.extract_text() or ""
            pages_text.append(f"\n\n--- PAGE {i+1} ---\n\n{text}")
    except Exception as e:
        if pdfminer_extract_text is None:
            raise SystemExit(
                "PDF text extraction requires either `pypdf` or `pdfminer.six`.\n"
                "Tip: run this script with the repo root venv: `..\\.venv\\Scripts\\python.exe`"
            ) from e
        extracted = pdfminer_extract_text(str(pdf_path)) or ""
        pages_text = [extracted]
        print(f"[warn] pypdf extraction failed ({type(e).__name__}: {e}); used pdfminer.six fallback")

    full_text = normalize_ws("\n".join(pages_text))
    out_txt = out_dir / "AML-2025_Mistake_Detection_Project.txt"
    out_txt.write_text(full_text, encoding="utf-8")

    # Heuristic: pull segments around likely requirement headings.
    heading_patterns = [
        r"requirements",
        r"deliverables",
        r"submission",
        r"evaluation",
        r"grading",
        r"task",
        r"baseline",
        r"dataset",
    ]

    lowered = full_text.lower()
    hits: list[tuple[int, str]] = []
    for pat in heading_patterns:
        for m in re.finditer(pat, lowered):
            hits.append((m.start(), pat))

    hits.sort(key=lambda x: x[0])
    # Deduplicate nearby hits
    filtered: list[tuple[int, str]] = []
    last = -10_000
    for pos, pat in hits:
        if pos - last > 800:
            filtered.append((pos, pat))
            last = pos

    print(f"PDF: {pdf_path.name}")
    if num_pages is not None:
        print(f"Pages: {num_pages}")
    print(f"Extracted text: {out_txt}")

    # Print first page and the requirement-like snippets for quick inspection.
    print("\n==== FIRST PAGE (preview) ====")
    print("\n".join(full_text.splitlines()[:120]))

    print("\n==== REQUIREMENT-LIKE SNIPPETS ====")
    window = 900
    for pos, pat in filtered[:25]:
        snippet = full_text[max(0, pos - 200) : pos + window]
        snippet = normalize_ws(snippet)
        print("\n---")
        print(f"[match: {pat} @ {pos}]")
        print(snippet)


if __name__ == "__main__":
    main()
