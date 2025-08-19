#!/usr/bin/env python3
"""Generate a Presocratics fragment catalogue straight from book-transcription pages.

This avoids brute-force probing by leveraging the hyperlinks CSV extracted
from the Muruca interface (``data/presocratic_hyperlinks.csv``).

Output: ``data/daphnet/presocratics_catalogue.json`` – list of dicts::

    {
        "reference": "22-B,30a",
        "siglum": "22-B,30a",
        "id": 2785,
        "html_url": "http://.../agora_show_transcription?id=2785&siglum=22-B%2C30a",
        "plain_url": "http://.../texts/Presocratics/22-B,30a.plain.html",
        "rdf_url": "http://.../texts/Presocratics/22-B,30a.rdf"
    }

Run from project root:

    python scaffold-source-texts/scaffold/scan_presocratic_books.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import aiohttp
from bs4 import BeautifulSoup

from config_loader import get_source_base_url, load_config

BASE_URL: str = get_source_base_url()
CONFIG = load_config()

CSV_PATH: Path = Path("scaffold-source-texts/data/presocratic_hyperlinks.csv")
OUT_DIR: Path = Path(CONFIG["output"]["directory"])
OUT_FILE: Path = OUT_DIR / "presocratics_catalogue.json"

REQUEST_DELAY: float = CONFIG["extraction"]["rate_limit"]["base_delay"]

SIGLUM_RE = re.compile(r"siglum=([^&]+)")
ID_RE = re.compile(r"id=(\d+)")


@dataclass
class FragmentRow:
    reference: str
    siglum: str
    id: Optional[int]
    html_url: str
    plain_url: str
    rdf_url: str

    def to_dict(self):  # noqa: D401
        """Return as serialisable dict."""

        return asdict(self)


async def fetch_html(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch a URL and return text on 200."""

    await asyncio.sleep(REQUEST_DELAY)
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception:
        return None
    return None


async def parse_book_page(session: aiohttp.ClientSession, url: str) -> List[FragmentRow]:
    """Return all fragments listed on a book-transcription page."""

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
        plain_url = f"{BASE_URL}/texts/Presocratics/{siglum}.plain.html"
        rdf_url = f"{BASE_URL}/texts/Presocratics/{siglum}.rdf"
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


async def process_csv() -> None:
    """Read CSV, crawl pages, write JSON catalogue."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect unique book URLs
    book_urls: List[str] = []
    with CSV_PATH.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            url = row.get("url")
            if url and url.startswith("http"):
                book_urls.append(url)

    async with aiohttp.ClientSession(headers={"User-Agent": CONFIG["source"]["user_agent"]}) as session:
        all_frags: List[FragmentRow] = []
        start = time.time()
        for i, url in enumerate(book_urls, 1):
            print(f"[{i}/{len(book_urls)}] {url}")
            rows = await parse_book_page(session, url)
            all_frags.extend(rows)

        print(f"Discovered {len(all_frags)} fragments in {time.time()-start:.1f}s")

    with OUT_FILE.open("w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in all_frags], fh, ensure_ascii=False, indent=2)
    print(f"Catalogue saved → {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(process_csv())
