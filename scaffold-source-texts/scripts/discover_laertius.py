#!/usr/bin/env python3
"""Laertius Discovery Script.

This script discovers all available Laertius fragments by crawling the book
transcription pages for each of the 10 books (I-X) and extracting the actual
available references, rather than guessing the ranges.

This follows the principle that each collection should have its own discovery
script tailored to its specific structure and conventions.
"""

from __future__ import annotations

import asyncio
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
CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "checkpoints"

# Load config
import yaml
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

BASE_URL = CONFIG["source"]["base_url"]
REQUEST_DELAY = float(CONFIG["extraction"]["rate_limit"]["base_delay"])

@dataclass
class LaertiusFragment:
    """Represents a discovered Laertius fragment."""
    reference: str
    collection: str
    book: str
    section: str
    is_italian: bool
    rdf_url: str
    html_url: str
    transcription_url: str
    discovery_method: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class LaertiusDiscovery:
    """Discover all Laertius fragments from the book transcription pages."""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.fragments: Dict[str, LaertiusFragment] = {}
        self.processed_books: Set[str] = set()
        
        # Book URLs from the CSV mapping
        self.book_urls = {
            "I": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/1",
            "II": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/2", 
            "III": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/3",
            "IV": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/4",
            "V": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/5",
            "VI": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/6",
            "VII": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/7",
            "VIII": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/8",
            "IX": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/9",
            "X": "http://ancientsource.daphnet.iliesi.cnr.it/agora_show_book_transcription/10"
        }

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

    async def fetch_book_page(self, book: str, url: str) -> Optional[str]:
        """Fetch the HTML content of a book transcription page."""
        if not self.session:
            await self.setup()

        try:
            logger.info(f"Fetching book {book} from {url}")
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.debug(f"Successfully fetched book {book} ({len(content)} chars)")
                    return content
                else:
                    logger.warning(f"HTTP {response.status} for book {book}: {url}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching book {book}: {e}")
            return None

    def parse_book_fragments(self, book: str, html_content: str) -> List[LaertiusFragment]:
        """Parse fragments from a book transcription page."""
        fragments = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for the reference patterns in the HTML
        # The references appear as list items or links with patterns like "VII,1", "VII,1it", etc.
        reference_pattern = re.compile(rf'{book},\d+(?:it)?')
        
        # Find all text that matches our pattern
        all_text = soup.get_text()
        matches = reference_pattern.findall(all_text)
        
        # Also look in links and list items more specifically
        for element in soup.find_all(['a', 'li', 'div', 'span']):
            text = element.get_text(strip=True)
            if reference_pattern.match(text):
                matches.append(text)
        
        # Remove duplicates and sort
        unique_refs = []
        seen_refs = set()
        
        for match in matches:
            if match not in seen_refs:
                seen_refs.add(match)
                unique_refs.append(match)
        
        # Sort with better error handling
        def sort_key(ref):
            try:
                section_part = ref.replace(f'{book},', '').replace('it', '')
                section_num = int(section_part)
                is_italian = ref.endswith('it')
                return (section_num, is_italian)
            except ValueError:
                # If we can't parse the number, put it at the end
                logger.warning(f"Could not parse section number from reference: {ref}")
                return (99999, False)
        
        unique_refs = sorted(unique_refs, key=sort_key)
        
        logger.info(f"Found {len(unique_refs)} references for book {book}")
        if unique_refs:
            logger.debug(f"Sample references: {unique_refs[:5]}...{unique_refs[-5:] if len(unique_refs) > 5 else ''}")
        
        for ref in unique_refs:
            # Parse reference components
            parts = ref.split(',')
            if len(parts) != 2:
                continue
                
            book_part = parts[0]  # e.g., "VII"
            section_part = parts[1]  # e.g., "1" or "1it"
            
            is_italian = section_part.endswith('it')
            section = section_part.replace('it', '')
            
            # Build URLs
            encoded_ref = quote_plus(ref)
            rdf_url = f"{BASE_URL}/texts/Laertius/{ref}.rdf"
            html_url = f"{BASE_URL}/texts/Laertius/{ref}.plain.html"
            transcription_url = f"{BASE_URL}/agora_show_transcription?siglum={encoded_ref}"
            
            fragment = LaertiusFragment(
                reference=ref,
                collection="laertius",
                book=book_part,
                section=section,
                is_italian=is_italian,
                rdf_url=rdf_url,
                html_url=html_url,
                transcription_url=transcription_url,
                discovery_method="book_transcription"
            )
            
            fragments.append(fragment)
        
        return fragments

    def _sort_section_key(self, section: str) -> tuple[int, str]:
        """Create a sort key for section numbers that handles special cases like '33a', '120b'."""
        try:
            # Try to parse as a simple integer first
            return (int(section), "")
        except ValueError:
            # Handle cases like '33a', '120b', etc.
            import re
            match = re.match(r'^(\d+)([a-z]*)$', section)
            if match:
                num_part = int(match.group(1))
                letter_part = match.group(2)
                return (num_part, letter_part)
            else:
                # If we can't parse it at all, put it at the end
                return (99999, section)

    async def discover_book(self, book: str) -> List[LaertiusFragment]:
        """Discover all fragments for a specific book."""
        if book in self.processed_books:
            logger.info(f"Book {book} already processed, skipping")
            return []
        
        url = self.book_urls.get(book)
        if not url:
            logger.error(f"No URL found for book {book}")
            return []
        
        # Fetch the book page
        html_content = await self.fetch_book_page(book, url)
        if not html_content:
            logger.error(f"Failed to fetch content for book {book}")
            return []
        
        # Parse fragments
        fragments = self.parse_book_fragments(book, html_content)
        
        # Add to our collection
        for fragment in fragments:
            self.fragments[fragment.reference] = fragment
        
        self.processed_books.add(book)
        
        # Respectful delay
        await asyncio.sleep(REQUEST_DELAY)
        
        return fragments

    async def discover_all_books(self) -> Dict[str, LaertiusFragment]:
        """Discover fragments from all Laertius books."""
        logger.info("Starting Laertius fragment discovery...")
        
        all_fragments = []
        for book in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]:
            try:
                fragments = await self.discover_book(book)
                all_fragments.extend(fragments)
                logger.info(f"Book {book}: discovered {len(fragments)} fragments")
            except Exception as e:
                logger.error(f"Error discovering book {book}: {e}")
                continue
        
        logger.info(f"Total fragments discovered: {len(self.fragments)}")
        return self.fragments

    def save_catalogue(self) -> Path:
        """Save the discovered fragments to a catalogue file."""
        CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
        
        catalogue_file = CATALOGUE_DIR / "laertius_catalogue.json"
        
        # Group by book for statistics
        books_stats = {}
        for fragment in self.fragments.values():
            book = fragment.book
            if book not in books_stats:
                books_stats[book] = {'total': 0, 'italian': 0, 'regular': 0}
            books_stats[book]['total'] += 1
            if fragment.is_italian:
                books_stats[book]['italian'] += 1
            else:
                books_stats[book]['regular'] += 1
        
        # Create catalogue structure
        catalogue = {
            "collection": "laertius",
            "discovery_date": datetime.now().isoformat(),
            "total_fragments": len(self.fragments),
            "total_books": len(books_stats),
            "books_discovered": list(books_stats.keys()),
            "discovery_methods": {
                "book_transcription": len(self.fragments)
            },
            "book_statistics": books_stats,
            "fragments": [fragment.to_dict() for fragment in sorted(
                self.fragments.values(), 
                key=lambda f: (f.book, self._sort_section_key(f.section), f.is_italian)
            )]
        }
        
        with open(catalogue_file, 'w', encoding='utf-8') as f:
            json.dump(catalogue, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved catalogue to {catalogue_file}")
        logger.info(f"Total fragments: {len(self.fragments)}")
        
        # Print book statistics
        logger.info("Fragments per book:")
        for book in sorted(books_stats.keys()):
            stats = books_stats[book]
            logger.info(f"  Book {book}: {stats['total']} total ({stats['regular']} regular + {stats['italian']} Italian)")
        
        return catalogue_file

    async def run(self) -> Path:
        """Run the complete discovery process."""
        try:
            await self.setup()
            await self.discover_all_books()
            catalogue_file = self.save_catalogue()
            return catalogue_file
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Laertius Fragment Discovery")
    logger.info("Discovers all available Laertius fragments from Daphnet")
    logger.info("by crawling the actual book transcription pages")
    logger.info("=" * 60)
    
    discovery = LaertiusDiscovery()
    catalogue_file = await discovery.run()
    
    logger.info("=" * 60)
    logger.info(f"Discovery complete! Catalogue saved to: {catalogue_file}")
    logger.info("Use this catalogue with extract_laertius.py for extraction")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
