#!/usr/bin/env python3
"""Generic Daphnet catalogue scanner.

This utility crawls book-transcription pages listed in the *_hyperlinks.csv files
and produces a fragment catalogue JSON for a chosen collection.  The generated
catalogue rows are compatible with :pymeth:`extract.DaphnetExtractor` which will
consume them when ``--from-catalogue`` is used (or when the JSON file already
exists).

Usage (from project root)::

    python scaffold-source-texts/scaffold/scan_catalogue.py --collection socratics

Available collections and their corresponding CSV inputs are:

* presocratics – ``scaffold-source-texts/data/presocratic_hyperlinks.csv``
* socratics – ``scaffold-source-texts/data/socratics_hyperlinks.csv``
* laertius – ``scaffold-source-texts/data/laertius_hyperlinks.csv``
* sextus – ``scaffold-source-texts/data/sextus_hyperlink.csv``

All other logic (request delay, user-agent) is read from the shared
``config.yaml`` via :pyfunc:`scaffold.config_loader.load_config`.
"""
from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

from config_loader import get_source_base_url, load_config
import signal
from http_utils import build_default_headers

# Global run-state for interrupt handling
_RUN_STATE: Dict[str, Any] = {"rows": [], "out_file": None}

# ---------------------------------------------------------------------------
# Signal handling helpers
# ---------------------------------------------------------------------------

def _handle_interrupt(signum, frame):  # noqa: D401 – signal handler
    """On Ctrl-C write partial catalogue before exiting."""

    out_file: Optional[Path] = _RUN_STATE.get("out_file")
    rows: List[FragmentRow] = _RUN_STATE.get("rows", [])
    if out_file and rows:
        print("\nInterrupt – saving partial catalogue …")
        out_file.write_text(
            json.dumps([r.to_dict() for r in rows], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Partial catalogue saved → {out_file}")
    raise SystemExit(130)

# ---------------------------------------------------------------------------
# Collection-specific metadata
# ---------------------------------------------------------------------------

_COLLECTION_META: MutableMapping[str, Dict[str, str]] = {
    "presocratics": {
        "csv": "scaffold-source-texts/data/presocratic_hyperlinks.csv",
        "output": "presocratics_catalogue.json",
        "path": "Presocratics",
    },
    "socratics": {
        "csv": "scaffold-source-texts/data/socratics_hyperlinks.csv",
        "output": "socratics_catalogue.json",
        "path": "Socratics",
    },
    "laertius": {
        "csv": "scaffold-source-texts/data/laertius_hyperlinks.csv",
        "output": "laertius_catalogue.json",
        "path": "Laertius",
    },
    "sextus": {
        "csv": "scaffold-source-texts/data/sextus_hyperlink.csv",
        "output": "sextus_catalogue.json",
        "path": "Sextus",
    },
}

SIGLUM_RE: re.Pattern[str] = re.compile(r"siglum=([^&]+)")
ID_RE: re.Pattern[str] = re.compile(r"id=(\d+)")
REQUEST_DELAY: float = float(CONFIG["extraction"]["rate_limit"]["base_delay"])

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FragmentRow:  # noqa: D101 – simple data container
    reference: str
    siglum: str
    id: Optional[int]
    html_url: str
    plain_url: str
    rdf_url: str

    def to_dict(self) -> Dict[str, Any]:  # noqa: D401 – dictionary converter
        """Return a JSON-serialisable mapping."""

        return asdict(self)


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

CACHE_DIR: Path = Path(CONFIG["output"]["directory"]) / "page_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path_for(url: str) -> Path:
    """Return a filesystem path for cached *url* HTML."""

    safe = quote_plus(url)
    return CACHE_DIR / f"{safe}.html"


async def fetch_html(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch *url* (with retry) and return body text when status == 200.

    If *url* has been cached previously this hits the local filesystem instead.
    """

    cpath = _cache_path_for(url)
    if cpath.exists():
        try:
            return cpath.read_text(encoding="utf-8")
        except Exception:
            pass  # fall through to network fetch on read error

    max_attempts = 3
    backoff = 1.0
    for attempt in range(max_attempts):
        await asyncio.sleep(REQUEST_DELAY)
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    # Save to cache
                    try:
                        cpath.write_text(text, encoding="utf-8")
                    except Exception:
                        pass
                    return text
                if resp.status in {500, 502, 503, 504}:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
        except asyncio.TimeoutError:
            await asyncio.sleep(backoff)
            backoff *= 2
            continue
        except Exception:
            return None
    return None


async def parse_book_page(
    session: aiohttp.ClientSession, url: str, collection_path: str
) -> List[FragmentRow]:
    """Return fragment rows discovered on a single book-transcription page."""

    html = await fetch_html(session, url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    rows: List[FragmentRow] = []

    for a in soup.select("a.boxview_link"):
        siglum_attr = a.get("data-url", "")
        m_sig = SIGLUM_RE.search(siglum_attr)
        if not m_sig:
            continue
        siglum = m_sig.group(1).replace("%2C", ",")
        m_id = ID_RE.search(siglum_attr)
        num_id = int(m_id.group(1)) if m_id else None

        html_url = f"{BASE_URL}{a.get('href')}"
        plain_url = f"{BASE_URL}/texts/{collection_path}/{siglum}.plain.html"
        rdf_url = f"{BASE_URL}/texts/{collection_path}/{siglum}.rdf"
        rows.append(
            FragmentRow(
                reference=siglum,
                siglum=siglum,
                id=num_id,
                html_url=html_url,
                plain_url=plain_url,
                rdf_url=rdf_url,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Main processing logic
# ---------------------------------------------------------------------------


async def process_collection(collection: str) -> None:  # noqa: D401 – coroutine
    """Crawl all book pages listed for *collection* and write the catalogue."""

    if collection not in _COLLECTION_META:
        raise ValueError(f"Unknown collection '{collection}'.")

    meta = _COLLECTION_META[collection]
    csv_path = Path(meta["csv"])
    out_dir = Path(CONFIG["output"]["directory"])
    out_file = out_dir / meta["output"]

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    book_urls: List[str] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            url = row.get("url")
            if url and url.startswith("http"):
                book_urls.append(url)

    if not book_urls:
        raise RuntimeError(f"No book URLs found in {csv_path}")

    async with aiohttp.ClientSession(
        headers=build_default_headers(CONFIG["source"]["user_agent"])
    ) as session:
        all_frags: List[FragmentRow] = []
        _RUN_STATE["rows"] = all_frags
        _RUN_STATE["out_file"] = out_file
        start = time.time()
        for i, url in enumerate(book_urls, 1):
            print(f"[{i}/{len(book_urls)}] {url}")
            rows = await parse_book_page(session, url, meta["path"])
            all_frags.extend(rows)

        elapsed = time.time() - start
        print(f"Discovered {len(all_frags)} fragments in {elapsed:.1f}s")

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in all_frags], fh, ensure_ascii=False, indent=2)

    print(f"Catalogue saved → {out_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> Namespace:
    """Return parsed command-line arguments."""

    p = ArgumentParser(description="Daphnet catalogue scanner (generic)")
    p.add_argument(
        "--collection",
        choices=list(_COLLECTION_META.keys()),
        required=True,
        help="Target collection to scan",
    )
    return p.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_interrupt)
    args = _parse_args()
    asyncio.run(process_collection(args.collection))
