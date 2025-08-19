#!/usr/bin/env python3
"""
Unified Discovery Script for Daphnet Texts
Combines two complementary approaches:
1. CSV-based discovery from book transcription pages
2. RDF-based discovery through systematic probing
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
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, quote

import aiohttp
import yaml
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
CONFIG_FILE = Path(__file__).parent.parent / "config" / "config.yaml"
INPUT_DIR = Path(__file__).parent.parent / "data" / "input"
CATALOGUE_DIR = Path(__file__).parent.parent / "data" / "catalogues"


@dataclass
class FragmentRecord:
    """Unified fragment record from either discovery method."""
    reference: str  # e.g., "22-B,30"
    collection: str  # e.g., "presocratics"
    philosopher: str  # e.g., "Heraclitus"
    fragment_id: Optional[int] = None  # From book transcription
    plain_url: Optional[str] = None
    rdf_url: Optional[str] = None
    html_url: Optional[str] = None
    transcription_url: Optional[str] = None  # agora_show_transcription URL
    discovery_method: str = ""  # "csv" or "rdf"
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class UnifiedDiscovery:
    """Combines CSV and RDF discovery methods."""
    
    def __init__(self):
        self.config = self.load_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.fragments: Dict[str, FragmentRecord] = {}
        
    def load_config(self) -> Dict:
        """Load configuration from YAML."""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                config = yaml.safe_load(f)
                # Flatten nested config structure for compatibility
                flattened = self.default_config()
                if 'source' in config:
                    flattened.update({
                        'base_url': config['source'].get('base_url', flattened['base_url']),
                        'user_agent': config['source'].get('user_agent', flattened['user_agent'])
                    })
                if 'extraction' in config and 'rate_limit' in config['extraction']:
                    rate_limit = config['extraction']['rate_limit']
                    flattened.update({
                        'delay': rate_limit.get('base_delay', flattened['delay']),
                        'timeout': rate_limit.get('timeout', flattened['timeout']),
                        'max_retries': rate_limit.get('max_retries', flattened['max_retries'])
                    })
                if 'philosopher_map' in config:
                    flattened['philosopher_map'].update(config['philosopher_map'])
                return flattened
        return self.default_config()
    
    def default_config(self) -> Dict:
        """Default configuration if config.yaml not found."""
        return {
            'base_url': 'http://ancientsource.daphnet.iliesi.cnr.it',
            'delay': 2.0,
            'timeout': 30,
            'max_retries': 2,  # Reduced for 500 errors
            'user_agent': 'DaphnetExtractor/2.0 (Academic Research)',
            'philosopher_map': {
                '11': 'Thales',
                '12': 'Anaximander',
                '13': 'Anaximenes',
                '14': 'Pythagoras',
                '21': 'Xenophanes',
                '22': 'Heraclitus',
                '28': 'Parmenides',
                '31': 'Empedocles',
                '59': 'Anaxagoras',
                '68': 'Democritus',
                # Add more as needed
            }
        }
    
    async def setup(self):
        """Setup HTTP session."""
        headers = {
            'User-Agent': self.config['user_agent'],
            'Accept': 'text/html,application/xml',
            'Accept-Encoding': 'gzip, deflate',
        }
        timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
    
    async def cleanup(self):
        """Cleanup HTTP session."""
        if self.session:
            await self.session.close()
    
    # ===== CSV-based Discovery =====
    
    async def discover_from_csv(self, collection: str = 'presocratics') -> List[FragmentRecord]:
        """Discover fragments from CSV hyperlinks."""
        # Handle filename variations for each collection
        filename_map = {
            'presocratics': 'presocratic_hyperlinks.csv',  # Note: singular form
            'socratics': 'socratics_hyperlinks.csv',
            'laertius': 'laertius_hyperlinks.csv', 
            'sextus': 'sextus_hyperlink.csv'  # Note: singular form
        }
        
        csv_filename = filename_map.get(collection, f"{collection}_hyperlinks.csv")
        csv_file = INPUT_DIR / csv_filename
        if not csv_file.exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return []
        
        records = []
        book_urls = self.parse_csv_urls(csv_file, collection)
        
        for chapter_ref, url in book_urls:
            logger.info(f"Fetching book transcription: {chapter_ref}")
            fragments = await self.fetch_book_transcription(url, chapter_ref, collection)
            records.extend(fragments)
            await asyncio.sleep(self.config['delay'])
        
        return records
    
    def parse_csv_urls(self, csv_file: Path, collection: str) -> List[Tuple[str, str]]:
        """Parse book transcription URLs from CSV with collection-specific patterns."""
        urls = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'agora_show_book_transcription' in row.get('url', ''):
                    text = row.get('text', '')
                    chapter_ref = None
                    
                    # Collection-specific pattern matching
                    if collection == 'presocratics':
                        # Pattern: "1-A transcription", "2-B transcription"
                        match = re.search(r'(\d+-[A-Z])\s+transcription', text)
                        if match:
                            chapter_ref = match.group(1)
                    
                    elif collection == 'socratics':
                        # Pattern: "I-A transcription", "II-B transcription"
                        match = re.search(r'([IVX]+-[A-Z])\s+transcription', text)
                        if match:
                            chapter_ref = match.group(1)
                    
                    elif collection == 'laertius':
                        # Pattern: "VIII transcription", "IX transcription"
                        match = re.search(r'([IVX]+)\s+transcription', text)
                        if match:
                            chapter_ref = match.group(1)
                    
                    elif collection == 'sextus':
                        # Pattern: "PH-1 transcription", "Math-4 transcription"
                        match = re.search(r'((?:PH|Math)-\d+)\s+transcription', text)
                        if match:
                            chapter_ref = match.group(1)
                    
                    if chapter_ref:
                        urls.append((chapter_ref, row['url']))
        return urls
    
    async def fetch_book_transcription(self, url: str, chapter_ref: str, 
                                      collection: str) -> List[FragmentRecord]:
        """Fetch and parse a book transcription page."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract fragment links
                fragments = []
                for link in soup.find_all('a', class_='boxview_link'):
                    data_url = link.get('data-url', '')
                    match = re.search(r'id=(\d+)&siglum=([^&]+)', data_url)
                    if match:
                        frag_id = int(match.group(1))
                        siglum = match.group(2).replace('%2C', ',')
                        
                        # Get philosopher from chapter
                        chapter_num = chapter_ref.split('-')[0]
                        philosopher = self.config['philosopher_map'].get(
                            chapter_num, f"Chapter {chapter_num}"
                        )
                        
                        record = FragmentRecord(
                            reference=siglum,
                            collection=collection,
                            philosopher=philosopher,
                            fragment_id=frag_id,
                            transcription_url=urljoin(self.config['base_url'], data_url),
                            discovery_method='csv'
                        )
                        fragments.append(record)
                
                return fragments
                
        except Exception as e:
            logger.error(f"Error fetching book transcription {url}: {e}")
            return []
    
    # ===== RDF-based Discovery =====
    
    async def discover_from_rdf(self, collection: str = 'presocratics') -> List[FragmentRecord]:
        """Discover fragments by systematic RDF probing with collection-specific patterns."""
        records = []
        
        if collection == 'presocratics':
            # Use CSV data to determine what chapter references actually exist
            csv_file = INPUT_DIR / self.get_csv_filename(collection)
            if csv_file.exists():
                csv_chapters = self._extract_csv_chapters(csv_file, collection)
                logger.info(f"Found {len(csv_chapters)} chapter references from CSV")
                
                for chapter_ref in csv_chapters:
                    logger.info(f"Probing {collection} chapter {chapter_ref}")
                    records.extend(await self._probe_chapter_range(
                        collection, chapter_ref, 1, 200
                    ))
            else:
                logger.warning(f"CSV file not found for {collection}, using fallback ranges")
                # Fallback to original broad ranges
                for chapter in range(1, 91):
                    for frag_type in ['A', 'B', 'C']:
                        logger.info(f"Probing {collection} chapter {chapter}-{frag_type}")
                        records.extend(await self._probe_chapter_range(
                            collection, f"{chapter}-{frag_type}", 1, 200
                        ))
        
        elif collection == 'socratics':
            # Use CSV data to determine what chapter references actually exist
            csv_file = INPUT_DIR / self.get_csv_filename(collection)
            if csv_file.exists():
                csv_chapters = self._extract_csv_chapters(csv_file, collection)
                logger.info(f"Found {len(csv_chapters)} chapter references from CSV")
                
                for chapter_ref in csv_chapters:
                    logger.info(f"Probing {collection} chapter {chapter_ref}")
                    records.extend(await self._probe_chapter_range(
                        collection, chapter_ref, 1, 200
                    ))
            else:
                logger.warning(f"CSV file not found for {collection}, using fallback ranges")
                # Fallback to smaller, more realistic ranges
                roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI']
                letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                
                for roman in roman_numerals:
                    for letter in letters:
                        chapter_ref = f"{roman}-{letter}"
                        logger.info(f"Probing {collection} chapter {chapter_ref}")
                        records.extend(await self._probe_chapter_range(
                            collection, chapter_ref, 1, 200
                        ))
        
        elif collection == 'laertius':
            # Use CSV data to determine what chapter references actually exist
            csv_file = INPUT_DIR / self.get_csv_filename(collection)
            if csv_file.exists():
                csv_chapters = self._extract_csv_chapters(csv_file, collection)
                logger.info(f"Found {len(csv_chapters)} chapter references from CSV")
                
                for chapter_ref in csv_chapters:
                    logger.info(f"Probing {collection} chapter {chapter_ref}")
                    records.extend(await self._probe_chapter_range(
                        collection, chapter_ref, 1, 200
                    ))
            else:
                logger.warning(f"CSV file not found for {collection}, using fallback ranges")
                # Fallback ranges
                roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
                
                for roman in roman_numerals:
                    logger.info(f"Probing {collection} chapter {roman}")
                    records.extend(await self._probe_chapter_range(
                        collection, roman, 1, 200
                    ))
        
        elif collection == 'sextus':
            # Use CSV data to determine what chapter references actually exist
            csv_file = INPUT_DIR / self.get_csv_filename(collection)
            if csv_file.exists():
                csv_chapters = self._extract_csv_chapters(csv_file, collection)
                logger.info(f"Found {len(csv_chapters)} chapter references from CSV")
                
                for chapter_ref in csv_chapters:
                    logger.info(f"Probing {collection} chapter {chapter_ref}")
                    records.extend(await self._probe_chapter_range(
                        collection, chapter_ref, 1, 200
                    ))
            else:
                logger.warning(f"CSV file not found for {collection}, using fallback ranges")
                # Fallback ranges
                prefixes = ['PH-1', 'PH-2', 'PH-3'] + [f'Math-{i}' for i in range(1, 12)]
                
                for prefix in prefixes:
                    logger.info(f"Probing {collection} chapter {prefix}")
                    records.extend(await self._probe_chapter_range(
                        collection, prefix, 1, 200
                    ))
        
        return records
    
    def get_csv_filename(self, collection: str) -> str:
        """Get the CSV filename for a collection."""
        filename_map = {
            'presocratics': 'presocratic_hyperlinks.csv',
            'socratics': 'socratics_hyperlinks.csv',
            'laertius': 'laertius_hyperlinks.csv', 
            'sextus': 'sextus_hyperlink.csv'
        }
        return filename_map.get(collection, f"{collection}_hyperlinks.csv")
    
    def _extract_csv_chapters(self, csv_file: Path, collection: str) -> List[str]:
        """Extract chapter references from CSV file."""
        chapters = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'agora_show_book_transcription' in row.get('url', ''):
                    text = row.get('text', '')
                    chapter_ref = None
                    
                    # Collection-specific pattern matching
                    if collection == 'presocratics':
                        match = re.search(r'(\d+-[A-Z])\s+transcription', text)
                        if match:
                            chapter_ref = match.group(1)
                    
                    elif collection == 'socratics':
                        match = re.search(r'([IVX]+-[A-Z])\s+transcription', text)
                        if match:
                            chapter_ref = match.group(1)
                    
                    elif collection == 'laertius':
                        match = re.search(r'([IVX]+)\s+transcription', text)
                        if match:
                            chapter_ref = match.group(1)
                    
                    elif collection == 'sextus':
                        match = re.search(r'((?:PH|Math)-\d+)\s+transcription', text)
                        if match:
                            chapter_ref = match.group(1)
                    
                    if chapter_ref and chapter_ref not in chapters:
                        chapters.append(chapter_ref)
        
        return sorted(chapters)
    
    async def _probe_chapter_range(self, collection: str, chapter_ref: str, 
                                  start_num: int, max_num: int) -> List[FragmentRecord]:
        """Probe a range of fragment numbers for a given chapter reference."""
        records = []
        consecutive_failures = 0
        
        for num in range(start_num, max_num + 1):
            ref = f"{chapter_ref},{num}"
            record = await self.probe_rdf(ref, collection)
            
            if record:
                records.append(record)
                consecutive_failures = 0
                
                # Try sub-fragments (a, b, c, etc.)
                for suffix in 'abcdefgh':
                    sub_ref = f"{chapter_ref},{num}{suffix}"
                    sub_record = await self.probe_rdf(sub_ref, collection)
                    if sub_record:
                        records.append(sub_record)
                    else:
                        break  # No more sub-fragments
            else:
                consecutive_failures += 1
                # After 10 consecutive failures, assume no more fragments in this chapter
                if consecutive_failures >= 10:
                    break
            
            await asyncio.sleep(self.config['delay'])
        
        return records
    
    async def probe_rdf(self, reference: str, collection: str) -> Optional[FragmentRecord]:
        """Probe a single RDF URL and extract URLs if it exists."""
        rdf_url = f"{self.config['base_url']}/texts/{collection.title()}/{reference}.rdf"
        
        try:
            async with self.session.get(rdf_url) as response:
                if response.status != 200:
                    return None
                
                rdf_content = await response.text()
                urls = self.parse_rdf(rdf_content)
                
                if urls:
                    # Get philosopher from reference
                    chapter_num = reference.split('-')[0]
                    philosopher = self.config['philosopher_map'].get(
                        chapter_num, f"Chapter {chapter_num}"
                    )
                    
                    return FragmentRecord(
                        reference=reference,
                        collection=collection,
                        philosopher=philosopher,
                        rdf_url=rdf_url,
                        plain_url=urls.get('plain'),
                        html_url=urls.get('html'),
                        discovery_method='rdf'
                    )
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout probing {rdf_url}")
        except Exception as e:
            logger.debug(f"Error probing {rdf_url}: {e}")
        
        return None
    
    def parse_rdf(self, rdf_content: str) -> Dict[str, str]:
        """Parse RDF content to extract URLs."""
        urls = {}
        try:
            root = ET.fromstring(rdf_content)
            
            # Define namespaces
            ns = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'edm': 'http://www.europeana.eu/schemas/edm/',
                'korbo': 'http://purl.org/net7/korbo/vocab#'
            }
            
            # Find the Description element
            desc = root.find('.//rdf:Description', ns)
            if desc is not None:
                # Extract URLs
                shown_by = desc.find('edm:isShownBy', ns)
                if shown_by is not None:
                    urls['plain'] = shown_by.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                
                shown_at = desc.find('edm:isShownAt', ns)
                if shown_at is not None:
                    urls['html'] = shown_at.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                
        except ET.ParseError as e:
            logger.debug(f"Failed to parse RDF: {e}")
        
        return urls
    
    # ===== Merge and Save =====
    
    def merge_discoveries(self, csv_records: List[FragmentRecord], 
                         rdf_records: List[FragmentRecord]) -> Dict[str, FragmentRecord]:
        """Merge discoveries from both methods."""
        merged = {}
        
        # Add CSV discoveries
        for record in csv_records:
            key = f"{record.collection}:{record.reference}"
            merged[key] = record
        
        # Merge RDF discoveries
        for record in rdf_records:
            key = f"{record.collection}:{record.reference}"
            if key in merged:
                # Merge information
                existing = merged[key]
                if record.plain_url:
                    existing.plain_url = record.plain_url
                if record.rdf_url:
                    existing.rdf_url = record.rdf_url
                if record.html_url:
                    existing.html_url = record.html_url
            else:
                merged[key] = record
        
        return merged
    
    def save_catalogue(self, records: Dict[str, FragmentRecord], collection: str):
        """Save discovery catalogue to JSON."""
        CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
        
        output_file = CATALOGUE_DIR / f"{collection}_catalogue.json"
        
        catalogue = {
            'metadata': {
                'collection': collection,
                'discovery_date': datetime.now().isoformat(),
                'total_fragments': len(records),
                'discovery_methods': ['csv', 'rdf']
            },
            'fragments': [record.to_dict() for record in records.values()]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(catalogue, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved catalogue: {output_file} ({len(records)} fragments)")
    
    async def run(self, collection: str = 'presocratics', use_csv: bool = True, 
                  use_rdf: bool = True):
        """Run unified discovery."""
        await self.setup()
        
        try:
            csv_records = []
            rdf_records = []
            
            if use_csv:
                logger.info("Starting CSV-based discovery...")
                csv_records = await self.discover_from_csv(collection)
                logger.info(f"Found {len(csv_records)} fragments via CSV")
            
            if use_rdf:
                logger.info("Starting RDF-based discovery...")
                # Start from chapter 1 for full discovery
                rdf_records = await self.discover_from_rdf(collection)
                logger.info(f"Found {len(rdf_records)} fragments via RDF")
            
            # Merge discoveries
            merged = self.merge_discoveries(csv_records, rdf_records)
            
            # Save catalogue
            self.save_catalogue(merged, collection)
            
            return merged
            
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Daphnet Discovery')
    parser.add_argument('--collection', default='presocratics',
                       choices=['presocratics', 'socratics', 'laertius', 'sextus'])
    parser.add_argument('--csv', action='store_true', default=True,
                       help='Use CSV-based discovery')
    parser.add_argument('--rdf', action='store_true', default=False,
                       help='Use RDF-based discovery')
    parser.add_argument('--both', action='store_true',
                       help='Use both discovery methods')
    
    args = parser.parse_args()
    
    if args.both:
        args.csv = True
        args.rdf = True
    
    discovery = UnifiedDiscovery()
    fragments = await discovery.run(
        collection=args.collection,
        use_csv=args.csv,
        use_rdf=args.rdf
    )
    
    print(f"\nDiscovery complete: {len(fragments)} fragments found")
    print(f"Catalogue saved to: {CATALOGUE_DIR}/{args.collection}_catalogue.json")


if __name__ == '__main__':
    asyncio.run(main())
