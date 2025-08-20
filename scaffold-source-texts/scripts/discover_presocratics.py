#!/usr/bin/env python3
"""Presocratics Discovery Script.

This script discovers all available Presocratics fragments by using the
presocratic_name_hyperlinks.csv file which contains philosopher names and
chapter references, then crawling the book transcription pages to find
all available fragments.

This follows the principle that each collection should have its own discovery
script tailored to its specific structure and conventions.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import quote_plus

import aiohttp
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
CONFIG_FILE = Path(__file__).parent.parent / "config" / "config.yaml"
CATALOGUE_DIR = Path(__file__).parent.parent / "data" / "catalogues"
INPUT_DIR = Path(__file__).parent.parent / "data" / "input"

# Load config
import yaml
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

BASE_URL = CONFIG["source"]["base_url"]
REQUEST_DELAY = float(CONFIG["extraction"]["rate_limit"]["base_delay"])

@dataclass
class PresocraticFragment:
    """Represents a discovered Presocratic fragment."""
    reference: str
    collection: str
    philosopher: str
    chapter: str
    fragment_type: str  # A, B, C
    rdf_url: str
    html_url: str
    transcription_url: str
    discovery_method: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PresocraticsDiscovery:
    """Discover all Presocratics fragments from the book transcription pages."""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.fragments: Dict[str, PresocraticFragment] = {}
        self.processed_chapters: Set[str] = set()
        
        # Load chapter-to-URL mapping from CSV
        self.chapter_urls: Dict[str, str] = {}
        self.chapter_philosophers: Dict[str, str] = {}
        self.load_chapter_mappings()

    def load_chapter_mappings(self):
        """Load chapter mappings from the name hyperlinks CSV."""
        csv_path = INPUT_DIR / "presocratic_name_hyperlinks.csv"
        
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return
        
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    text = row.get("Text", "").strip()  # e.g., "22-B"
                    philosopher = row.get("Philosopher", "").strip()  # e.g., "Herakleitos"
                    hyperlink = row.get("Hyperlink", "").strip()
                    
                    if text and philosopher and hyperlink:
                        self.chapter_urls[text] = hyperlink
                        self.chapter_philosophers[text] = philosopher
            
            logger.info(f"Loaded {len(self.chapter_urls)} chapter mappings")
            
        except Exception as exc:
            logger.error(f"Failed loading {csv_path}: {exc}")

    async def setup(self):
        """Initialize HTTP session."""
        headers = {
            'User-Agent': CONFIG["source"]["user_agent"]
        }
        self.session = aiohttp.ClientSession(headers=headers)

    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()

    async def fetch_chapter_page(self, chapter: str, url: str) -> Optional[str]:
        """Fetch the HTML content of a chapter transcription page."""
        if not self.session:
            await self.setup()

        try:
            logger.info(f"Fetching chapter {chapter} from {url}")
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.debug(f"Successfully fetched chapter {chapter} ({len(content)} chars)")
                    return content
                else:
                    logger.warning(f"HTTP {response.status} for chapter {chapter}: {url}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching chapter {chapter}: {e}")
            return None

    def parse_chapter_fragments(self, chapter: str, html_content: str) -> List[PresocraticFragment]:
        """Parse fragments from a chapter transcription page."""
        fragments = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        philosopher = self.chapter_philosophers.get(chapter, f"Chapter {chapter}")
        
        # Extract chapter number and type (e.g., "22-B" -> chapter="22", type="B")
        if "-" in chapter:
            chapter_num, fragment_type = chapter.split("-", 1)
        else:
            chapter_num, fragment_type = chapter, "A"
        
        # Look for fragment references in the HTML
        # Pattern: chapter-type,number (e.g., "22-B,1", "22-B,2", etc.)
        reference_pattern = re.compile(rf'{re.escape(chapter)},\d+(?:[abc])?')
        
        # Find all text that matches our pattern
        all_text = soup.get_text()
        matches = reference_pattern.findall(all_text)
        
        # Also look in links and list items more specifically
        for element in soup.find_all(['a', 'li', 'div', 'span']):
            text = element.get_text(strip=True)
            if reference_pattern.match(text):
                matches.append(text)
        
        # Remove duplicates and sort
        unique_refs = sorted(set(matches), key=lambda x: (
            int(x.split(',')[1].rstrip('abc')),
            x.split(',')[1][-1] if x.split(',')[1][-1] in 'abc' else ''
        ))
        
        logger.info(f"Found {len(unique_refs)} references for chapter {chapter}")
        if unique_refs:
            logger.debug(f"Sample references: {unique_refs[:5]}...{unique_refs[-5:] if len(unique_refs) > 5 else ''}")
        
        for ref in unique_refs:
            # Build URLs
            encoded_ref = quote_plus(ref)
            rdf_url = f"{BASE_URL}/texts/Presocratics/{ref}.rdf"
            html_url = f"{BASE_URL}/texts/Presocratics/{ref}.plain.html"
            transcription_url = f"{BASE_URL}/agora_show_transcription?siglum={encoded_ref}"
            
            fragment = PresocraticFragment(
                reference=ref,
                collection="presocratics",
                philosopher=philosopher,
                chapter=chapter_num,
                fragment_type=fragment_type,
                rdf_url=rdf_url,
                html_url=html_url,
                transcription_url=transcription_url,
                discovery_method="book_transcription"
            )
            
            fragments.append(fragment)
        
        return fragments

    async def discover_chapter(self, chapter: str) -> List[PresocraticFragment]:
        """Discover all fragments for a specific chapter."""
        if chapter in self.processed_chapters:
            logger.info(f"Chapter {chapter} already processed, skipping")
            return []
        
        url = self.chapter_urls.get(chapter)
        if not url:
            logger.error(f"No URL found for chapter {chapter}")
            return []
        
        # Fetch the chapter page
        html_content = await self.fetch_chapter_page(chapter, url)
        if not html_content:
            logger.error(f"Failed to fetch content for chapter {chapter}")
            return []
        
        # Parse fragments
        fragments = self.parse_chapter_fragments(chapter, html_content)
        
        # Add to our collection
        for fragment in fragments:
            self.fragments[fragment.reference] = fragment
        
        self.processed_chapters.add(chapter)
        
        # Respectful delay
        await asyncio.sleep(REQUEST_DELAY)
        
        return fragments

    async def discover_all_chapters(self) -> Dict[str, PresocraticFragment]:
        """Discover fragments from all Presocratics chapters."""
        logger.info("Starting Presocratics fragment discovery...")
        
        all_fragments = []
        chapters = sorted(self.chapter_urls.keys())
        
        for chapter in chapters:
            try:
                fragments = await self.discover_chapter(chapter)
                all_fragments.extend(fragments)
                logger.info(f"Chapter {chapter}: discovered {len(fragments)} fragments")
            except Exception as e:
                logger.error(f"Error discovering chapter {chapter}: {e}")
                continue
        
        logger.info(f"Total fragments discovered: {len(self.fragments)}")
        return self.fragments

    def save_catalogue(self) -> Path:
        """Save the discovered fragments to a catalogue file."""
        CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
        
        catalogue_file = CATALOGUE_DIR / "presocratics_catalogue.json"
        
        # Group by philosopher for statistics
        philosopher_stats = {}
        for fragment in self.fragments.values():
            philosopher = fragment.philosopher
            if philosopher not in philosopher_stats:
                philosopher_stats[philosopher] = 0
            philosopher_stats[philosopher] += 1
        
        # Create catalogue structure
        catalogue = {
            "collection": "presocratics",
            "discovery_date": datetime.now().isoformat(),
            "total_fragments": len(self.fragments),
            "total_philosophers": len(philosopher_stats),
            "philosophers_discovered": list(philosopher_stats.keys()),
            "discovery_methods": {
                "book_transcription": len(self.fragments)
            },
            "philosopher_statistics": philosopher_stats,
            "fragments": [fragment.to_dict() for fragment in sorted(
                self.fragments.values(), 
                key=lambda f: (int(f.chapter), f.fragment_type, int(f.reference.split(',')[1].rstrip('abc')))
            )]
        }
        
        with open(catalogue_file, 'w', encoding='utf-8') as f:
            json.dump(catalogue, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved catalogue to {catalogue_file}")
        logger.info(f"Total fragments: {len(self.fragments)}")
        
        # Print philosopher statistics
        logger.info("Fragments per philosopher:")
        for philosopher in sorted(philosopher_stats.keys()):
            count = philosopher_stats[philosopher]
            logger.info(f"  {philosopher}: {count} fragments")
        
        return catalogue_file

    async def run(self) -> Path:
        """Run the complete discovery process."""
        try:
            await self.setup()
            await self.discover_all_chapters()
            catalogue_file = self.save_catalogue()
            return catalogue_file
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Presocratics Fragment Discovery")
    logger.info("Discovers all available Presocratics fragments from Daphnet")
    logger.info("using the presocratic_name_hyperlinks.csv file")
    logger.info("=" * 60)
    
    discovery = PresocraticsDiscovery()
    catalogue_file = await discovery.run()
    
    logger.info("=" * 60)
    logger.info(f"Discovery complete! Catalogue saved to: {catalogue_file}")
    logger.info("Use this catalogue with extract_presocratics.py for extraction")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
