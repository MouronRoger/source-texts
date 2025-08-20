#!/usr/bin/env python3
"""Socratics (SSR) Extraction Script.

This lightweight wrapper reuses the generic extraction logic from
``extract_texts.py`` but overrides the fragment–to–philosopher mapping so that
Roman-numeral chapter identifiers (``I``, ``II`` …) are handled gracefully.

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


class SocraticsExtractor(_generic.UnifiedExtractor):
    """Extractor with Roman numeral support for chapter identifiers."""

    def __init__(self) -> None:
        super().__init__()

        # Load chapter → name mapping from CSV (if available)
        self.chapter_name_map: Dict[str, str] = {}
        csv_path = (
            Path(__file__).parent.parent
            / "data"
            / "input"
            / "socratics_name_hyperlinks.csv"
        )
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        ref = row.get("ref", "").strip()
                        name = row.get("name", "").strip()
                        if ref and name:
                            self.chapter_name_map[ref] = name
            except Exception as exc:  # pragma: no cover – robust to format issues
                logger.warning("Failed loading %s: %s", csv_path, exc)

    # ------------------------------------------------------------------
    # Catalogue sorting helpers
    # ------------------------------------------------------------------

    _roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}

    @classmethod
    def _sort_key(cls, ref: str) -> tuple[int, str, int]:
        """Return a tuple that sorts references naturally (1,2,3…)."""

        chapter, *rest = ref.split(",")
        roman, letter = chapter.split("-")
        frag_num = int(rest[0]) if rest else 0
        return cls._roman_map.get(roman, 99), letter, frag_num

    # Override to apply natural ordering
    def load_catalogue(self, collection: str):  # type: ignore[override]
        fragments = super().load_catalogue(collection)
        return sorted(fragments, key=lambda f: self._sort_key(f["reference"]))

    async def extract_fragment(self, fragment: Dict[str, Any]) -> Optional[ExtractedText]:
        """Extract a single Socratics fragment with Roman numeral awareness."""

        reference: str = fragment["reference"]

        # Skip if already processed
        if reference in self.processed_refs:
            return None

        # Derive philosopher mapping key without assuming numeric chapter refs
        chapter_part = reference.split("-")[0]
        chap_key: Union[int, str] = int(chapter_part) if chapter_part.isdigit() else chapter_part

        philosopher: str = self.chapter_name_map.get(
            chapter_part,
            self.config.get("philosopher_map", {}).get(
                chap_key, fragment.get("philosopher", f"Chapter {chapter_part}")
            ),
        )

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


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


async def _main() -> None:  # noqa: D401
    import argparse

    parser = argparse.ArgumentParser(description="Socratics Daphnet Extraction")
    parser.add_argument(
        "--catalogue",
        default=str(Path(__file__).parent.parent / "data" / "catalogues" / "socratics_catalogue.json"),
        help="Path to the Socratics catalogue JSON (auto-detected by default).",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and start fresh.")

    args = parser.parse_args()

    extractor = SocraticsExtractor()
    # Run using fixed collection name 'socratics'
    await extractor.run(collection="socratics", resume=not args.no_resume)


if __name__ == "__main__":
    asyncio.run(_main())
