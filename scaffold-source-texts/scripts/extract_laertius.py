#!/usr/bin/env python3
"""Laertius Extraction Script.

This lightweight wrapper reuses the generic extraction logic from
``extract_texts.py`` but overrides the fragment–to–philosopher mapping for
Diogenes Laertius texts following the Roman numeral book convention (I, II, III, etc.)
with section numbers (1, 2, 3, etc.) and optional Italian translations (1it, 2it, etc.).

The rest of the pipeline (HTTP fetching, language separation, checkpointing,
output generation) is inherited unchanged.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import sys

# ---------------------------------------------------------------------------
# Dynamically load the generic extractor module so we do not duplicate ~650
# lines of implementation while avoiding package-name issues with the hyphen in
# ``scaffold-source-texts``.
# ---------------------------------------------------------------------------

GENERIC_PATH = Path(__file__).with_name("extract_texts.py")
MODULE_NAME = "generic_extractor"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, GENERIC_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover – sanity check
    raise ImportError(f"Unable to load generic extractor at {GENERIC_PATH}")
_generic = importlib.util.module_from_spec(SPEC)
# Register in sys.modules *before* execution so dataclasses sees the module
sys.modules[MODULE_NAME] = _generic
SPEC.loader.exec_module(_generic)  # type: ignore[arg-type]

ExtractedText = _generic.ExtractedText  # Alias for type hints

# Re-export logger so we get consistent formatting
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LaertiusExtractor(_generic.UnifiedExtractor):
    """Extractor with Roman numeral support for Laertius book identifiers."""

    def __init__(self) -> None:
        super().__init__()

        # Load book → name mapping from CSV (if available)
        self.book_name_map: Dict[str, str] = {}
        csv_path = (
            Path(__file__).parent.parent
            / "data"
            / "input"
            / "laertius_books_with_links.csv"
        )
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8-sig") as fh:  # Handle BOM
                    reader = csv.DictReader(fh)
                    for row in reader:
                        book = row.get("Book", "").strip()
                        order = row.get("Order", "").strip()
                        if book and order:
                            # Map Roman numeral (e.g., "Liber VII") to order number (e.g., "7")
                            roman_num = book.replace("Liber ", "")
                            self.book_name_map[roman_num] = f"Book {order}"
            except Exception as exc:  # pragma: no cover – robust to format issues
                logger.warning("Failed loading %s: %s", csv_path, exc)

    # ------------------------------------------------------------------
    # Catalogue sorting helpers
    # ------------------------------------------------------------------

    _roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}

    @classmethod
    def _sort_key(cls, ref: str) -> tuple[int, int, bool]:
        """Return a tuple that sorts references naturally (I,1 < I,2 < I,1it < I,2it)."""
        
        parts = ref.split(",")
        if len(parts) < 2:
            return (99, 0, False)  # Invalid format
        
        book_roman = parts[0]  # e.g., "VII"
        section_part = parts[1]  # e.g., "1" or "1it"
        
        # Extract section number and check for Italian
        is_italian = section_part.endswith("it")
        section_num = int(section_part.replace("it", "")) if section_part.replace("it", "").isdigit() else 0
        
        book_num = cls._roman_map.get(book_roman, 99)
        return book_num, section_num, is_italian

    # Override to apply natural ordering
    def load_catalogue(self, collection: str):  # type: ignore[override]
        fragments = super().load_catalogue(collection)
        return sorted(fragments, key=lambda f: self._sort_key(f["reference"]))

    async def extract_fragment(self, fragment: Dict[str, Any]) -> Optional[ExtractedText]:
        """Extract a single Laertius fragment with Roman numeral awareness."""

        reference: str = fragment["reference"]

        # Skip if already processed
        if reference in self.processed_refs:
            return None

        # Derive book mapping from reference (e.g., "VII,1" -> "VII")
        parts = reference.split(",")
        if len(parts) >= 1:
            book_roman = parts[0]  # e.g., "VII"
            book_info = self.book_name_map.get(book_roman, f"Book {book_roman}")
        else:
            book_info = "Unknown Book"

        # For Laertius, we use the book info as the "philosopher"
        philosopher: str = book_info

        # Build URL preference list
        urls_to_try: List[str] = []
        if fragment.get("plain_url"):
            urls_to_try.append(fragment["plain_url"])  # type: ignore[arg-type]
        if fragment.get("transcription_url"):
            urls_to_try.append(fragment["transcription_url"])  # type: ignore[arg-type]
        if fragment.get("html_url"):
            urls_to_try.append(fragment["html_url"])  # type: ignore[arg-type]

        extracted_data: Optional[Dict[str, str]] = None
        successful_url: Optional[str] = None

        for url in urls_to_try:
            data = await self.extract_from_url(url)
            if data:
                extracted_data = data  # type: ignore[assignment]
                successful_url = url
                break
            await asyncio.sleep(self.config["delay"] / 2)  # type: ignore[operator]

        # Handle modern dict format (Greek/Latin/Italian separated)
        if extracted_data and isinstance(extracted_data, dict):
            greek_text = extracted_data.get("greek_text", "")
            latin_text = extracted_data.get("latin_text", "")

            if greek_text or latin_text:  # Require ancient text
                main_text = greek_text if greek_text else latin_text
                extracted = ExtractedText(
                    reference=reference,
                    collection=fragment["collection"],
                    philosopher=philosopher,
                    greek_text=greek_text,
                    paragraphs=main_text.split("\n\n") if main_text else [],
                    url=successful_url or "",
                    extraction_date=datetime.now().isoformat(),
                )

                if latin_text:
                    extracted.latin_text = latin_text  # type: ignore[attr-defined]
                if extracted_data.get("italian_text"):
                    extracted.italian_text = extracted_data["italian_text"]  # type: ignore[attr-defined]
                if extracted_data.get("source_citation"):
                    extracted.source_citation = extracted_data["source_citation"]  # type: ignore[attr-defined]

                self.processed_refs.add(reference)
                return extracted

        # Legacy plain-string format fallback
        if extracted_data and isinstance(extracted_data, str):
            extracted = ExtractedText(
                reference=reference,
                collection=fragment["collection"],
                philosopher=philosopher,
                greek_text=extracted_data,
                paragraphs=extracted_data.split("\n\n"),
                url=successful_url or "",
                extraction_date=datetime.now().isoformat(),
            )
            self.processed_refs.add(reference)
            return extracted

        logger.warning("No ancient text (Greek or Latin) found for %s", reference)
        self.processed_refs.add(reference)
        return None

    def save_results(self, collection: str):
        """Save extraction results organized by book folders."""
        from pathlib import Path
        import json
        
        OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Group results by book
        books_data = {}
        for r in self.results:
            # Extract book from reference (e.g., "VII,1" -> "VII")
            parts = r.reference.split(",")
            if len(parts) >= 1:
                book_roman = parts[0]
                book_name = self.book_name_map.get(book_roman, f"Book_{book_roman}")
                
                if book_roman not in books_data:
                    books_data[book_roman] = {
                        'name': book_name,
                        'results': []
                    }
                books_data[book_roman]['results'].append(r)
        
        # Create output for each book
        for book_roman, book_info in books_data.items():
            book_name = book_info['name'].replace(" ", "_")  # "Book 7" -> "Book_7"
            book_dir = OUTPUT_DIR / f"laertius_{book_name.lower()}"
            book_dir.mkdir(parents=True, exist_ok=True)
            
            results = book_info['results']
            
            # JSON output
            json_file = book_dir / f"{book_name.lower()}_texts.json"
            json_data = []
            for r in results:
                entry = {
                    'reference': r.reference,
                    'collection': r.collection,
                    'philosopher': r.philosopher,
                    'greek_text': r.greek_text,
                    'paragraphs': r.paragraphs,
                    'url': r.url,
                    'extraction_date': r.extraction_date
                }
                if hasattr(r, 'latin_text') and r.latin_text:
                    entry['latin_text'] = r.latin_text
                if hasattr(r, 'italian_text') and r.italian_text:
                    entry['italian_text'] = r.italian_text
                if hasattr(r, 'source_citation') and r.source_citation:
                    entry['source_citation'] = r.source_citation
                json_data.append(entry)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Greek-only text output
            greek_only_file = book_dir / f"{book_name.lower()}_greek_only.txt"
            greek_count = 0
            with open(greek_only_file, 'w', encoding='utf-8') as f:
                for r in results:
                    if r.greek_text:
                        greek_count += 1
                        f.write(f"=== {r.reference} ({r.philosopher}) ===\n")
                        if hasattr(r, 'source_citation') and r.source_citation:
                            f.write(f"Source: {r.source_citation}\n")
                        f.write(f"{r.greek_text}\n\n")
            
            # Ancient texts (Greek + Latin, no Italian)
            ancient_file = book_dir / f"{book_name.lower()}_ancient.txt"
            with open(ancient_file, 'w', encoding='utf-8') as f:
                for r in results:
                    if r.greek_text or (hasattr(r, 'latin_text') and r.latin_text):
                        f.write(f"=== {r.reference} ({r.philosopher}) ===\n")
                        if hasattr(r, 'source_citation') and r.source_citation:
                            f.write(f"Source: {r.source_citation}\n")
                        
                        if r.greek_text:
                            f.write(f"{r.greek_text}\n")
                        
                        if hasattr(r, 'latin_text') and r.latin_text:
                            if r.greek_text:
                                f.write("---\n")
                            f.write(f"{r.latin_text}\n")
                        
                        f.write("\n")
            
            # Full output (everything including Italian)
            full_file = book_dir / f"{book_name.lower()}_full.txt"
            with open(full_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(f"=== {r.reference} ({r.philosopher}) ===\n")
                    if hasattr(r, 'source_citation') and r.source_citation:
                        f.write(f"Source: {r.source_citation}\n")
                    
                    if r.greek_text:
                        f.write(f"Greek: {r.greek_text}\n")
                    if hasattr(r, 'latin_text') and r.latin_text:
                        f.write(f"Latin: {r.latin_text}\n")
                    if hasattr(r, 'italian_text') and r.italian_text:
                        f.write(f"Italian: {r.italian_text}\n")
                    
                    f.write("\n")
            
            # Book-specific statistics
            latin_count = sum(1 for r in results if hasattr(r, 'latin_text') and r.latin_text)
            italian_count = sum(1 for r in results if hasattr(r, 'italian_text') and r.italian_text)
            
            stats_file = book_dir / f"{book_name.lower()}_stats.json"
            stats = {
                'book': book_info['name'],
                'book_roman': book_roman,
                'collection': collection,
                'extraction_date': datetime.now().isoformat(),
                'total_extracted': len(results),
                'with_greek': greek_count,
                'with_latin': latin_count,
                'with_italian': italian_count,
                'greek_only': sum(1 for r in results if r.greek_text and not (hasattr(r, 'latin_text') and r.latin_text)),
                'latin_only': sum(1 for r in results if not r.greek_text and hasattr(r, 'latin_text') and r.latin_text),
                'both_greek_latin': sum(1 for r in results if r.greek_text and hasattr(r, 'latin_text') and r.latin_text),
                'references': [r.reference for r in results]
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {book_info['name']} to {book_dir} ({len(results)} fragments)")
        
        # Overall collection statistics
        overall_stats_file = OUTPUT_DIR / f"{collection}_overall_stats.json"
        total_greek = sum(1 for r in self.results if r.greek_text)
        total_latin = sum(1 for r in self.results if hasattr(r, 'latin_text') and r.latin_text)
        total_italian = sum(1 for r in self.results if hasattr(r, 'italian_text') and r.italian_text)
        
        overall_stats = {
            'collection': collection,
            'extraction_date': datetime.now().isoformat(),
            'total_books': len(books_data),
            'books': list(books_data.keys()),
            'total_extracted': len(self.results),
            'with_greek': total_greek,
            'with_latin': total_latin,
            'with_italian': total_italian,
            'fragments_per_book': {
                book_roman: len(book_info['results']) 
                for book_roman, book_info in books_data.items()
            }
        }
        
        with open(overall_stats_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved overall statistics to {overall_stats_file}")
        logger.info(f"Greek: {total_greek}, Latin: {total_latin}, Italian: {total_italian}")


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


async def _main() -> None:  # noqa: D401
    import argparse

    parser = argparse.ArgumentParser(description="Laertius Daphnet Extraction")
    parser.add_argument(
        "--catalogue",
        default=str(Path(__file__).parent.parent / "data" / "catalogues" / "laertius_catalogue.json"),
        help="Path to the Laertius catalogue JSON (auto-detected by default).",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and start fresh.")

    args = parser.parse_args()

    extractor = LaertiusExtractor()
    # Run using fixed collection name 'laertius'
    await extractor.run(collection="laertius", resume=not args.no_resume)


if __name__ == "__main__":
    asyncio.run(_main())
