#!/usr/bin/env python3
"""
Socratics (SSR) Discovery Script for Daphnet Texts
Optimized for Roman numerals I-VI with sections A-N
Handles the specific patterns of the Socratics collection efficiently
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

# Configure logging - set to DEBUG temporarily to diagnose issues
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"
INPUT_DIR = BASE_DIR / "data" / "input"
CATALOGUE_DIR = BASE_DIR / "data" / "catalogues"

# Socratics-specific constants
COLLECTION = "socratics"
CSV_FILE = "socratics_hyperlinks.csv"
CATALOGUE_FILE = "socratics_catalogue.json"

# Known structure of Socratics collection (SSR)
ROMAN_CHAPTERS = ['I', 'II', 'III', 'IV', 'V', 'VI']  # Only I-VI exist
LETTER_SECTIONS = list('ABCDEFGHIKLMNOPQRS')  # Extended to cover all seen sections

# URL patterns
BASE_URL = 'http://ancientsource.daphnet.iliesi.cnr.it'
RDF_PATTERN = '/texts/Socratics/{reference}.rdf'
HTML_PATTERN = '/texts/Socratics/{reference}.plain.html'
BOOK_TRANSCRIPTION_PATTERN = '/agora_show_book_transcription/{book_id}'
TRANSCRIPTION_PATTERN = '/agora_show_transcription/{fragment_id}'


@dataclass
class FragmentRecord:
    """Unified fragment record for Socratics collection."""
    reference: str  # e.g., "I-A,15" or "VI-B,101"
    collection: str = "socratics"
    philosopher: str = ""  # To be determined from reference
    fragment_id: Optional[int] = None  # From book transcription
    plain_url: Optional[str] = None
    rdf_url: Optional[str] = None
    html_url: Optional[str] = None
    transcription_url: Optional[str] = None
    discovery_method: str = ""  # "csv" or "rdf"
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class SocraticsDiscovery:
    """Optimized discovery for Socratics collection."""
    
    def __init__(self):
        self.config = self.load_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.fragments: Dict[str, FragmentRecord] = {}
        self.csv_chapters: Set[str] = set()  # Track chapters found in CSV
        
    def load_config(self) -> Dict:
        """Load configuration from YAML."""
        config = {
            'base_url': BASE_URL,
            'delay': 2.0,
            'timeout': 30,
            'max_retries': 2,
            'user_agent': 'DaphnetExtractor/2.0 (Academic Research - Socratics)',
            'philosopher_map': {}  # Will be populated if available
        }
        
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    yaml_config = yaml.safe_load(f)
                    if 'source' in yaml_config:
                        config.update({
                            'base_url': yaml_config['source'].get('base_url', config['base_url']),
                            'user_agent': yaml_config['source'].get('user_agent', config['user_agent'])
                        })
                    if 'extraction' in yaml_config and 'rate_limit' in yaml_config['extraction']:
                        rate_limit = yaml_config['extraction']['rate_limit']
                        config.update({
                            'delay': rate_limit.get('base_delay', config['delay']),
                            'timeout': rate_limit.get('timeout', config['timeout']),
                            'max_retries': rate_limit.get('max_retries', config['max_retries'])
                        })
                    # Socratics philosopher mapping if available
                    if 'socratics_philosopher_map' in yaml_config:
                        config['philosopher_map'] = yaml_config['socratics_philosopher_map']
            except Exception as e:
                logger.warning(f"Could not load config.yaml: {e}, using defaults")
        
        return config
    
    def get_philosopher_name(self, reference: str) -> str:
        """Get philosopher name from reference."""
        # For Socratics, the chapter often indicates the school
        # This would need proper mapping based on SSR conventions
        chapter = reference.split('-')[0] if '-' in reference else reference.split(',')[0]
        
        # Default mapping for major Socratic schools
        default_map = {
            'I': 'Socratic School',
            'II': 'Megarian School',
            'III': 'Elean-Eretrian School',
            'IV': 'Cyrenaic School',
            'V': 'Cynic School',
            'VI': 'Other Socratics'
        }
        
        return self.config['philosopher_map'].get(chapter, default_map.get(chapter, 'Unknown Socratic'))
    
    async def setup_session(self):
        """Setup aiohttp session with headers."""
        if not self.session:
            headers = {
                'User-Agent': self.config['user_agent'],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
    
    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_with_retry(self, url: str, retries: int = None) -> Optional[str]:
        """Fetch URL with retry logic."""
        if retries is None:
            retries = self.config['max_retries']
        
        for attempt in range(retries + 1):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 404:
                        return None
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        if response.status == 500 and attempt == retries:
                            return None  # Don't retry 500 errors multiple times
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
            
            if attempt < retries:
                await asyncio.sleep(self.config['delay'] * (attempt + 1))
        
        return None
    
    async def discover_from_csv(self) -> List[FragmentRecord]:
        """Discover fragments from CSV hyperlinks file."""
        csv_file = INPUT_DIR / CSV_FILE
        
        if not csv_file.exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return []
        
        logger.info(f"Reading CSV file: {csv_file}")
        records = []
        
        # First pass: collect all chapter references and URLs
        chapters_to_fetch = []  # List of (chapter_ref, url) tuples
        
        # Pattern for Socratics: Roman numeral + letter
        pattern = re.compile(r'([IVX]+-[A-Z])\s+transcription')
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                url = row.get('url', '').strip()
                if not url:
                    continue
                
                # Extract chapter reference from text
                text = row.get('text', '')
                match = pattern.search(text)
                
                if match:
                    chapter_ref = match.group(1)  # e.g., "I-A"
                    self.csv_chapters.add(chapter_ref)
                    
                    # Extract fragment ID from URL
                    if 'agora_show_book_transcription' in url:
                        # Extract just the base URL without query parameters
                        base_url = url.split('?')[0] if '?' in url else url
                        full_url = base_url if base_url.startswith('http') else urljoin(self.config['base_url'], base_url)
                        chapters_to_fetch.append((chapter_ref, full_url))
        
        logger.info(f"Found {len(chapters_to_fetch)} chapter transcription pages to fetch")
        logger.info(f"Chapters found: {sorted([c[0] for c in chapters_to_fetch])}")
        
        # Second pass: fetch transcription pages (now that session is ready)
        for idx, (chapter_ref, transcription_url) in enumerate(chapters_to_fetch, 1):
            fragment_id = transcription_url.split('/')[-1]
            
            record = FragmentRecord(
                reference=chapter_ref,
                collection=COLLECTION,
                philosopher=self.get_philosopher_name(chapter_ref),
                fragment_id=int(fragment_id) if fragment_id.isdigit() else None,
                transcription_url=transcription_url,
                discovery_method='csv'
            )
            
            # Generate additional URLs
            record.rdf_url = urljoin(
                self.config['base_url'],
                RDF_PATTERN.format(reference=chapter_ref)
            )
            record.html_url = urljoin(
                self.config['base_url'],
                HTML_PATTERN.format(reference=chapter_ref)
            )
            
            records.append(record)
            
            # Fetch the transcription page to find individual fragments
            logger.info(f"[{idx}/{len(chapters_to_fetch)}] Fetching transcription page for {chapter_ref}")
            await asyncio.sleep(self.config['delay'])
            html = await self.fetch_with_retry(transcription_url)
            
            if html:
                fragment_records = self.parse_transcription_page(html, chapter_ref)
                if fragment_records:
                    logger.info(f"Found {len(fragment_records)} fragments in {chapter_ref}")
                    records.extend(fragment_records)
                else:
                    logger.warning(f"No fragments found in {chapter_ref} transcription page")
            else:
                logger.warning(f"Could not fetch transcription page for {chapter_ref}")
        
        logger.info(f"Discovered {len(records)} total records from CSV")
        logger.info(f"Found chapters in CSV: {sorted(self.csv_chapters)}")
        
        return records
    
    def parse_transcription_page(self, html: str, chapter_ref: str) -> List[FragmentRecord]:
        """Parse a book transcription page to extract fragment references."""
        records = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Debug: Check if we got valid HTML
        if len(html) < 100:
            logger.warning(f"HTML content too short for {chapter_ref}: {len(html)} chars")
            return records
        
        # Look for all links
        all_links = soup.find_all('a', href=True)
        logger.debug(f"Found {len(all_links)} links in {chapter_ref} transcription page")
        
        # Track unique fragments to avoid duplicates
        seen_fragments = set()
        
        # Try multiple patterns to find fragments
        # Pattern 1: "I-A,1" or "I-A, 1" format
        fragment_pattern1 = re.compile(rf'{re.escape(chapter_ref)},\s*(\d+)')
        # Pattern 2: "I-A 1" format (space instead of comma)
        fragment_pattern2 = re.compile(rf'{re.escape(chapter_ref)}\s+(\d+)')
        # Pattern 3: Just the fragment number in link text if href contains the chapter
        fragment_pattern3 = re.compile(r'^(\d+)$')
        
        for i, link in enumerate(all_links):
            href = link['href']
            text = link.get_text().strip()
            
            # Debug first few links
            if i < 3:
                logger.debug(f"Link {i}: text='{text}', href='{href}'")
            
            fragment_num = None
            
            # Try pattern 1 and 2 first
            match = fragment_pattern1.search(text) or fragment_pattern2.search(text)
            if match:
                fragment_num = match.group(1)
            # If no match and href contains chapter ref, try pattern 3
            elif chapter_ref in href and fragment_pattern3.match(text):
                fragment_num = text
            # Also check if href contains the fragment reference
            elif chapter_ref in href:
                # Try to extract fragment number from href
                href_match = fragment_pattern1.search(href) or fragment_pattern2.search(href)
                if href_match:
                    fragment_num = href_match.group(1)
            
            if fragment_num and fragment_num not in seen_fragments:
                seen_fragments.add(fragment_num)
                full_ref = f"{chapter_ref},{fragment_num}"
                logger.debug(f"Found fragment: {full_ref}")
                
                record = FragmentRecord(
                    reference=full_ref,
                    collection=COLLECTION,
                    philosopher=self.get_philosopher_name(chapter_ref),
                    discovery_method='csv-transcription'
                )
                
                # Generate URLs
                record.rdf_url = urljoin(
                    self.config['base_url'],
                    RDF_PATTERN.format(reference=full_ref)
                )
                record.html_url = urljoin(
                    self.config['base_url'],
                    HTML_PATTERN.format(reference=full_ref)
                )
                
                # If the link has a useful href, save it
                if 'agora_show_transcription' in href or 'agora_show_fragment' in href or '/texts/' in href:
                    record.transcription_url = urljoin(self.config['base_url'], href)
                
                records.append(record)
        
        if not records and all_links:
            # Log some sample links to help debug
            logger.debug(f"No fragments found for {chapter_ref}. Sample links:")
            for i in range(min(5, len(all_links))):
                logger.debug(f"  Link: '{all_links[i].get_text().strip()}' -> '{all_links[i]['href']}'")
        
        return records
    
    async def probe_rdf(self, reference: str) -> Optional[FragmentRecord]:
        """Probe a single RDF URL to check if fragment exists."""
        rdf_url = urljoin(self.config['base_url'], RDF_PATTERN.format(reference=reference))
        
        content = await self.fetch_with_retry(rdf_url)
        if content:
            try:
                # Parse RDF/XML to verify it's valid
                root = ET.fromstring(content)
                
                record = FragmentRecord(
                    reference=reference,
                    collection=COLLECTION,
                    philosopher=self.get_philosopher_name(reference),
                    rdf_url=rdf_url,
                    html_url=urljoin(self.config['base_url'], HTML_PATTERN.format(reference=reference)),
                    discovery_method='rdf'
                )
                
                return record
            except ET.ParseError:
                logger.warning(f"Invalid RDF content for {reference}")
        
        return None
    
    async def discover_from_rdf_optimized(self) -> List[FragmentRecord]:
        """
        Optimized RDF discovery for Socratics collection.
        Uses CSV data as guide and implements intelligent stopping.
        """
        records = []
        
        # If we have CSV chapters, use them as authoritative
        if self.csv_chapters:
            logger.info(f"Using {len(self.csv_chapters)} chapters from CSV as guide for RDF discovery")
            
            for chapter_idx, chapter_ref in enumerate(sorted(self.csv_chapters), 1):
                logger.info(f"[{chapter_idx}/{len(self.csv_chapters)}] Probing RDF fragments for chapter {chapter_ref}")
                
                # Probe fragments 1-200 for this chapter
                consecutive_missing = 0
                found_count = 0
                
                for fragment_num in range(1, 1001):
                    reference = f"{chapter_ref},{fragment_num}"
                    
                    await asyncio.sleep(self.config['delay'])
                    record = await self.probe_rdf(reference)
                    
                    if record:
                        records.append(record)
                        consecutive_missing = 0
                        found_count += 1
                        if found_count % 10 == 0:
                            logger.debug(f"Found {found_count} fragments so far in {chapter_ref}")
                    else:
                        consecutive_missing += 1
                        
                        # Stop after 5 consecutive missing fragments
                        if consecutive_missing >= 5 and fragment_num > 10:
                            logger.info(f"Stopping {chapter_ref} at fragment {fragment_num} (5 consecutive missing, found {found_count} total)")
                            break
                
                if found_count > 0:
                    logger.info(f"Found {found_count} fragments via RDF for {chapter_ref}")
        
        else:
            # Fallback: intelligent probing with early stopping
            logger.info("No CSV data, using intelligent RDF probing")
            
            # CRITICAL: Only probe known chapters I-VI
            for roman in ROMAN_CHAPTERS:
                logger.info(f"Probing chapter {roman}")
                
                chapter_has_content = False
                empty_sections = 0
                
                # First, do a quick test to see if this chapter exists
                test_sections = ['A', 'B', 'C']  # Quick sample
                
                for letter in test_sections:
                    reference = f"{roman}-{letter},1"
                    
                    await asyncio.sleep(self.config['delay'])
                    if await self.probe_rdf(reference):
                        chapter_has_content = True
                        break
                
                if not chapter_has_content:
                    logger.info(f"Chapter {roman} appears empty, skipping")
                    continue
                
                # Chapter exists, probe all sections
                for letter in LETTER_SECTIONS:
                    chapter_ref = f"{roman}-{letter}"
                    section_has_content = False
                    consecutive_missing = 0
                    
                    # Probe fragments 1-200 for this section
                    for fragment_num in range(1, 1001):
                        reference = f"{chapter_ref},{fragment_num}"
                        
                        await asyncio.sleep(self.config['delay'])
                        record = await self.probe_rdf(reference)
                        
                        if record:
                            records.append(record)
                            section_has_content = True
                            consecutive_missing = 0
                        else:
                            consecutive_missing += 1
                            
                            if consecutive_missing >= 5 and fragment_num > 5:
                                break
                    
                    if not section_has_content:
                        empty_sections += 1
                        
                        # Stop after 3 consecutive empty sections
                        if empty_sections >= 3:
                            logger.info(f"Stopping chapter {roman} at section {letter} (3 empty sections)")
                            break
                    else:
                        empty_sections = 0
        
        logger.info(f"Discovered {len(records)} fragments via RDF")
        return records
    
    async def discover(self) -> None:
        """Main discovery process combining CSV and RDF methods."""
        await self.setup_session()
        
        try:
            # Phase 1: CSV discovery (fast)
            logger.info("Phase 1: Starting CSV discovery")
            csv_records = await self.discover_from_csv()
            
            for record in csv_records:
                self.fragments[record.reference] = record
            
            logger.info(f"CSV discovery complete: {len(csv_records)} fragments")
            
            # Phase 2: RDF discovery (comprehensive but optimized)
            logger.info("Phase 2: Starting optimized RDF discovery")
            rdf_records = await self.discover_from_rdf_optimized()
            
            # Merge RDF discoveries (don't overwrite CSV records)
            new_discoveries = 0
            for record in rdf_records:
                if record.reference not in self.fragments:
                    self.fragments[record.reference] = record
                    new_discoveries += 1
            
            logger.info(f"RDF discovery complete: {new_discoveries} new fragments")
            
            # Save catalogue
            self.save_catalogue()
            
        finally:
            await self.close_session()
    
    def save_catalogue(self):
        """Save discovered fragments to JSON catalogue."""
        CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
        
        catalogue_file = CATALOGUE_DIR / CATALOGUE_FILE
        
        # Convert to list and sort by reference
        catalogue_data = {
            'collection': COLLECTION,
            'discovery_date': datetime.now().isoformat(),
            'total_fragments': len(self.fragments),
            'discovery_methods': {
                'csv': sum(1 for f in self.fragments.values() if 'csv' in f.discovery_method),
                'rdf': sum(1 for f in self.fragments.values() if f.discovery_method == 'rdf')
            },
            'fragments': [
                record.to_dict() 
                for record in sorted(self.fragments.values(), key=lambda x: x.reference)
            ]
        }
        
        with open(catalogue_file, 'w', encoding='utf-8') as f:
            json.dump(catalogue_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved catalogue to {catalogue_file}")
        logger.info(f"Total fragments discovered: {len(self.fragments)}")
        
        # Print summary statistics
        chapters = {}
        for ref in self.fragments.keys():
            chapter = ref.split(',')[0] if ',' in ref else ref
            chapters[chapter] = chapters.get(chapter, 0) + 1
        
        logger.info("Fragments per chapter:")
        for chapter in sorted(chapters.keys()):
            logger.info(f"  {chapter}: {chapters[chapter]} fragments")


async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Socratics Collection Discovery (Optimized)")
    logger.info("=" * 60)
    
    discovery = SocraticsDiscovery()
    await discovery.discover()
    
    logger.info("Discovery complete!")


if __name__ == '__main__':
    asyncio.run(main())
