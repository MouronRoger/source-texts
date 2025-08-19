#!/usr/bin/env python3
"""
Enhanced Daphnet Extraction using RDF catalogue discovery.
Leverages the plain.html format for cleaner text extraction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import quote

import aiohttp
import yaml
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Fragment:
    """Enhanced fragment with full reference support."""
    reference: str  # Full reference like "22-B,30a"
    collection: str  # Presocratics, Laertius, etc.
    philosopher: str
    greek_text: str
    paragraphs: List[str]
    urls: Dict[str, str]  # html, plain, rdf URLs
    metadata: Dict[str, any]
    extraction_date: str
    
    def to_dict(self) -> Dict:
        return {
            'reference': self.reference,
            'collection': self.collection,
            'philosopher': self.philosopher,
            'greek_text': self.greek_text,
            'paragraphs': self.paragraphs,
            'urls': self.urls,
            'metadata': self.metadata,
            'extraction_date': self.extraction_date
        }


class EnhancedDaphnetExtractor:
    """Extractor using catalogue map and plain.html for cleaner extraction."""
    
    def __init__(self, config_path: Path = Path("scaffold/config.yaml"),
                 catalogue_path: Optional[Path] = None):
        """Initialize with config and optional catalogue."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.base_url = self.config['source']['base_url']
        self.philosopher_map = self.config['philosopher_map']
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0.0
        self.fragments: List[Fragment] = []
        
        # Load catalogue if provided
        self.catalogue: List[Dict] = []
        if catalogue_path and catalogue_path.exists():
            with open(catalogue_path, 'r') as f:
                self.catalogue = json.load(f)
            logger.info(f"Loaded {len(self.catalogue)} entries from catalogue")
    
    async def setup(self) -> None:
        """Setup HTTP session."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Academic Research Bot',
                'Accept': 'text/html,application/xhtml+xml,application/rdf+xml'
            },
            timeout=timeout
        )
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def rate_limit(self) -> None:
        """Enforce rate limiting."""
        delay = self.config['extraction']['rate_limit']['delay']
        elapsed = time.time() - self.last_request_time
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self.last_request_time = time.time()
    
    async def extract_from_catalogue(self) -> None:
        """Extract using catalogue entries."""
        if not self.catalogue:
            logger.warning("No catalogue loaded, falling back to discovery mode")
            await self.extract_with_discovery()
            return
        
        logger.info(f"Extracting {len(self.catalogue)} catalogue entries")
        
        # Group by collection for organized extraction
        by_collection = {}
        for entry in self.catalogue:
            coll = entry.get('collection', 'Unknown')
            if coll not in by_collection:
                by_collection[coll] = []
            by_collection[coll].append(entry)
        
        # Process Presocratics first (priority)
        if 'Presocratics' in by_collection:
            await self._process_collection('Presocratics', by_collection['Presocratics'])
        
        # Then other collections if configured
        for coll in ['Socratics', 'Laertius', 'Sextus']:
            if coll in by_collection:
                logger.info(f"Processing {coll}: {len(by_collection[coll])} entries")
                # await self._process_collection(coll, by_collection[coll])
    
    async def _process_collection(self, collection: str, entries: List[Dict]) -> None:
        """Process a collection of catalogue entries."""
        logger.info(f"Processing {collection}: {len(entries)} entries")
        
        for i, entry in enumerate(entries):
            # Build URLs from catalogue
            urls = {
                'html': entry['html_url'],
                'rdf': entry['rdf_url']
            }
            
            # Try plain.html first (cleaner)
            plain_url = entry['html_url'].replace('.html', '.plain.html')
            if not entry['html_url'].endswith('.html'):
                plain_url = entry['html_url'] + '.plain.html'
            urls['plain'] = plain_url
            
            # Extract content
            fragment = await self._extract_entry(collection, entry['reference'], urls)
            if fragment:
                self.fragments.append(fragment)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(entries)} from {collection}")
                self.save_progress()
    
    async def _extract_entry(self, collection: str, reference: str, 
                            urls: Dict[str, str]) -> Optional[Fragment]:
        """Extract a single entry trying plain.html first."""
        await self.rate_limit()
        
        # Try plain.html first for cleaner extraction
        content = await self._fetch_content(urls.get('plain'))
        if not content:
            # Fall back to regular HTML
            content = await self._fetch_content(urls.get('html'))
        
        if not content:
            logger.warning(f"Failed to fetch {reference}")
            return None
        
        # Parse content
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract Greek text with structure
        paragraphs = self._extract_greek_paragraphs(soup)
        if not paragraphs:
            return None
        
        greek_text = '\n\n'.join(paragraphs)
        greek_text = unicodedata.normalize('NFC', greek_text)
        
        # Get philosopher name
        philosopher = self._get_philosopher(collection, reference)
        
        # Extract metadata from RDF if available
        metadata = {}
        if 'rdf' in urls:
            rdf_metadata = await self._extract_rdf_metadata(urls['rdf'])
            if rdf_metadata:
                metadata.update(rdf_metadata)
        
        return Fragment(
            reference=reference,
            collection=collection,
            philosopher=philosopher,
            greek_text=greek_text,
            paragraphs=paragraphs,
            urls=urls,
            metadata=metadata,
            extraction_date=datetime.now().isoformat()
        )
    
    async def _fetch_content(self, url: str) -> Optional[str]:
        """Fetch content from URL."""
        if not url:
            return None
        
        try:
            assert self.session
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                return None
        except Exception as e:
            logger.debug(f"Error fetching {url}: {e}")
            return None
    
    async def _extract_rdf_metadata(self, rdf_url: str) -> Dict:
        """Extract metadata from RDF."""
        content = await self._fetch_content(rdf_url)
        if not content:
            return {}
        
        metadata = {}
        try:
            # Parse RDF for DC metadata
            soup = BeautifulSoup(content, 'xml')
            
            # Extract DC fields
            for field in ['title', 'creator', 'subject', 'description']:
                elem = soup.find(f'dc:{field}')
                if elem and elem.text:
                    metadata[f'dc_{field}'] = elem.text.strip()
            
            # Extract rdfs:label
            label = soup.find('rdfs:label')
            if label and label.text:
                metadata['label'] = label.text.strip()
                
        except Exception as e:
            logger.debug(f"RDF parsing error: {e}")
        
        return metadata
    
    def _extract_greek_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract Greek text preserving structure."""
        paragraphs = []
        
        # For plain.html, content is usually cleaner
        if 'plain' in str(soup):
            # Plain format often has minimal markup
            text = soup.get_text(separator='\n', strip=True)
            if self._has_greek(text):
                # Split on double newlines to get paragraphs
                parts = text.split('\n\n')
                for part in parts:
                    if part.strip() and self._has_greek(part):
                        paragraphs.append(part.strip())
        
        # Standard extraction for regular HTML
        if not paragraphs:
            for selector in self.config['extraction']['selectors']['primary']:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(strip=True)
                    if self._validate_greek_content(text):
                        paragraphs.append(text)
        
        return paragraphs
    
    def _has_greek(self, text: str) -> bool:
        """Check if text contains Greek characters."""
        return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in text)
    
    def _validate_greek_content(self, text: str) -> bool:
        """Validate Greek content meets requirements."""
        if not text or len(text) < 5:
            return False
        
        greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
        ratio = greek_chars / len(text) if text else 0
        
        return ratio >= 0.1
    
    def _get_philosopher(self, collection: str, reference: str) -> str:
        """Get philosopher name from reference."""
        if collection == 'Presocratics':
            # Extract chapter from reference like "22-B,30"
            if '-' in reference:
                chapter_str = reference.split('-')[0]
                try:
                    chapter = int(chapter_str)
                    return self.philosopher_map.get(chapter, f"Unknown Ch.{chapter}")
                except ValueError:
                    pass
        elif collection == 'Socratics':
            return "Socratic School"
        elif collection == 'Laertius':
            return "Diogenes Laertius"
        elif collection == 'Sextus':
            return "Sextus Empiricus"
        
        return f"{collection} Author"
    
    async def extract_with_discovery(self) -> None:
        """Fallback: Extract with URL discovery (original method)."""
        logger.info("Running in discovery mode...")
        
        # Use suffix-aware discovery
        presocratics = self.config['source']['collections']['presocratics']
        
        for chapter in range(1, 91):
            for ftype in ['A', 'B', 'C']:
                consecutive_404s = 0
                
                for number in range(1, 201):
                    if consecutive_404s >= 5:
                        break
                    
                    # Check base number
                    reference = f"{chapter}-{ftype},{number}"
                    urls = {
                        'html': f"{self.base_url}/texts/Presocratics/{reference}",
                        'plain': f"{self.base_url}/texts/Presocratics/{reference}.plain.html",
                        'rdf': f"{self.base_url}/texts/Presocratics/{reference}.rdf"
                    }
                    
                    fragment = await self._extract_entry('Presocratics', reference, urls)
                    if fragment:
                        self.fragments.append(fragment)
                        consecutive_404s = 0
                        
                        # Check for suffixes a-z
                        for suffix in 'abcdefghij':  # Usually only a-j exist
                            ref_suffix = f"{chapter}-{ftype},{number}{suffix}"
                            urls_suffix = {
                                'html': f"{self.base_url}/texts/Presocratics/{ref_suffix}",
                                'plain': f"{self.base_url}/texts/Presocratics/{ref_suffix}.plain.html",
                                'rdf': f"{self.base_url}/texts/Presocratics/{ref_suffix}.rdf"
                            }
                            
                            frag_suffix = await self._extract_entry('Presocratics', ref_suffix, urls_suffix)
                            if frag_suffix:
                                self.fragments.append(frag_suffix)
                            else:
                                break  # No more suffixes
                    else:
                        consecutive_404s += 1
                    
                    if len(self.fragments) % 50 == 0 and self.fragments:
                        logger.info(f"Discovered {len(self.fragments)} fragments")
                        self.save_progress()
    
    def save_progress(self) -> None:
        """Save current extraction progress."""
        output_dir = Path(self.config['output']['directory'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced JSON with all fields
        json_path = output_dir / 'daphnet_enhanced.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                [frag.to_dict() for frag in self.fragments],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        # Text corpus for RAG
        text_path = output_dir / 'daphnet_corpus.txt'
        with open(text_path, 'w', encoding='utf-8') as f:
            for frag in self.fragments:
                f.write(f"[{frag.reference}] {frag.philosopher} ({frag.collection})\n\n")
                f.write(frag.greek_text)
                f.write("\n\n---\n\n")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive extraction report."""
        # Group fragments by collection and type
        by_collection = {}
        suffixed_count = 0
        
        for frag in self.fragments:
            coll = frag.collection
            if coll not in by_collection:
                by_collection[coll] = []
            by_collection[coll].append(frag.reference)
            
            # Count suffixed fragments
            if any(c.isalpha() for c in frag.reference[-1:]):
                suffixed_count += 1
        
        report = {
            'extraction_date': datetime.now().isoformat(),
            'total_fragments': len(self.fragments),
            'suffixed_fragments': suffixed_count,
            'by_collection': {k: len(v) for k, v in by_collection.items()},
            'collections_detail': by_collection,
            'unique_philosophers': len(set(f.philosopher for f in self.fragments))
        }
        
        # Save report
        output_dir = Path(self.config['output']['directory'])
        report_path = output_dir / 'extraction_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    async def run(self, use_catalogue: bool = True) -> None:
        """Run extraction with catalogue support."""
        try:
            await self.setup()
            
            if use_catalogue and Path('./data/catalogue_map.json').exists():
                self.catalogue = json.load(open('./data/catalogue_map.json'))
                logger.info(f"Using catalogue with {len(self.catalogue)} entries")
                await self.extract_from_catalogue()
            else:
                logger.info("No catalogue found, using discovery mode")
                await self.extract_with_discovery()
            
            self.save_progress()
            report = self.generate_report()
            
            logger.info(f"Extraction complete:")
            logger.info(f"  Total: {report['total_fragments']} fragments")
            logger.info(f"  Suffixed: {report['suffixed_fragments']} sub-fragments")
            logger.info(f"  Collections: {report['by_collection']}")
            
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    import sys
    
    # Check for catalogue scan mode
    if '--scan-catalogue' in sys.argv:
        logger.info("Running catalogue scanner first...")
        # Import and run the scanner
        from scan_rdf_catalogue import main as scan_main
        await scan_main()
    
    # Run extraction
    extractor = EnhancedDaphnetExtractor()
    
    if '--test' in sys.argv:
        logger.info("Test mode: Limited extraction")
        extractor.config['source']['collections']['presocratics']['chapters'] = [22, 22]
        extractor.config['source']['collections']['presocratics']['max_number'] = 30
    
    use_catalogue = '--no-catalogue' not in sys.argv
    await extractor.run(use_catalogue=use_catalogue)


if __name__ == "__main__":
    asyncio.run(main())
