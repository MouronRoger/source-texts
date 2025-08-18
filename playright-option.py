# play_crawl.py
# pyright: reportMissingImports=false
# Minimal polite crawler using Playwright to discover dynamic links on the
# Daphnet Presocratics site. Intended as a fall-back when static HTML or
# plain-text transcriptions are unavailable.
from __future__ import annotations

import asyncio
import json
import time
from typing import List, Set
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright, Page

# ---------------------------------------------------------------------------
# Constants (kept local – crawler is experimental)
# ---------------------------------------------------------------------------

BASE: str = "http://ancientsource.daphnet.iliesi.cnr.it"
DOMAIN: str = urlparse(BASE).netloc
OUT_LINKS: str = "crawl_links.jsonl"
REQUEST_DELAY: float = 2.0

# Crawl politeness limits
MAX_PAGES: int = 2_000  # hard cap
CONCURRENCY: int = 1  # serial crawling for maximal respect

CLICK_SELECTORS: List[str] = [
    "a",  # normal anchors
    "nav a",
    "ul.menu a",
    "li a",
    "a[href*='texts/']",
]


def same_site(u: str) -> bool:
    """Return *True* when ``u`` is on the same host as BASE."""

    try:
        netloc: str = urlparse(u).netloc
        return netloc == "" or netloc == DOMAIN
    except Exception:
        return False


def should_visit(u: str) -> bool:
    """Basic filter against off-site or non-HTTP targets."""

    if not same_site(u):
        return False
    if u.startswith("mailto:") or u.startswith("javascript:"):
        return False
    return True


async def extract_links(page: Page) -> List[str]:
    """Return a list of on-site links found within the current page."""

    anchors = await page.eval_on_selector_all(
        "a",
        "els => els.map(e => e.getAttribute('href')).filter(Boolean)",
    )
    links: Set[str] = set()
    for href in anchors:
        full = urljoin(page.url, href)
        if should_visit(full):
            links.add(full.split("#")[0])
    return sorted(links)


async def crawl() -> None:
    """Breadth-first crawl within domain, emitting a JSONL link map."""

    seen: Set[str] = set()
    queue: List[str] = [BASE]
    saved: int = 0
    t0 = time.time()

    async with async_playwright() as pw:
        browser = await pw.firefox.launch(headless=True)
        context = await browser.new_context(
            user_agent="Daphnet-Crawler/0.2 (+academic research; contact: you@example.org)"
        )
        page = await context.new_page()

        while queue and len(seen) < MAX_PAGES:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)

            try:
                await page.goto(url, wait_until="networkidle", timeout=60_000)
            except Exception:
                continue

            # Attempt to expand nav menus that reveal additional links
            for sel in [
                "button[aria-controls]",
                ".menu-toggle",
                ".navbar-burger",
                ".hamburger",
            ]:
                try:
                    if await page.is_visible(sel):
                        await page.click(sel)
                        await page.wait_for_timeout(300)
                except Exception:
                    pass

            links: List[str] = await extract_links(page)

            # Persist targets that look like textual resources
            interesting = [
                u
                for u in links
                if "/texts/" in u or u.endswith(".html") or u.endswith(".plain.html")
            ]
            with open(OUT_LINKS, "a", encoding="utf-8") as f:
                for u in interesting:
                    f.write(json.dumps({"from": url, "to": u}) + "\n")
                    saved += 1

            # Enqueue newly discovered pages
            for u in links:
                if u not in seen and should_visit(u):
                    queue.append(u)

            await page.wait_for_timeout(int(REQUEST_DELAY * 1_000))

        await browser.close()

    print(
        f"Visited {len(seen)} pages, saved {saved} link records in {OUT_LINKS}, "
        f"elapsed {time.time() - t0:.1f}s",
    )


if __name__ == "__main__":
    asyncio.run(crawl())
