#!/usr/bin/env python3
"""
Unified Extraction Script for Daphnet Texts
Uses discovered catalogue to extract actual Greek texts
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiohttp
import yaml
from bs4 import BeautifulSoup, NavigableString

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
CONFIG_FILE = Path(__file__).parent.parent / "config" / "config.yaml"
CATALOGUE_DIR = Path(__file__).parent.parent / "data" / "catalogues"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "checkpoints"


@dataclass
class ExtractedText:
    """Extracted text with metadata."""
    reference: str
    collection: str
    philosopher: str
    greek_text: str
    paragraphs: List[str]
    url: str
    extraction_date: str
    italian_text: Optional[str] = None
    latin_text: Optional[str] = None
    source_citation: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None


class UnifiedExtractor:
    """Extract texts using discovered catalogue."""
    
    def __init__(self):
        self.config = self.load_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[ExtractedText] = []
        self.checkpoint_file = None
        self.processed_refs: Set[str] = set()
        
    def load_config(self) -> Dict:
        """Load configuration from YAML."""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                full_config = yaml.safe_load(f)
                # Build config dict with philosopher_map
                config = self.default_config()
                if 'philosopher_map' in full_config:
                    config['philosopher_map'] = full_config['philosopher_map']
                # Update with other settings
                if 'source' in full_config:
                    config.update(full_config['source'])
                if 'extraction' in full_config:
                    if 'rate_limit' in full_config['extraction']:
                        config.update(full_config['extraction']['rate_limit'])
                return config
        return self.default_config()
    
    def default_config(self) -> Dict:
        """Default configuration."""
        return {
            'base_url': 'http://ancientsource.daphnet.iliesi.cnr.it',
            'delay': 2.0,
            'timeout': 30,
            'max_retries': 3,
            'user_agent': 'DaphnetExtractor/2.0 (Academic Research)',
            'min_greek_ratio': 0.1,  # Minimum ratio of Greek characters
        }
    
    def load_catalogue(self, collection: str) -> List[Dict]:
        """Load fragment catalogue."""
        catalogue_file = CATALOGUE_DIR / f"{collection}_catalogue.json"
        if not catalogue_file.exists():
            raise FileNotFoundError(f"Catalogue not found: {catalogue_file}")
        
        with open(catalogue_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['fragments']
    
    def load_checkpoint(self, collection: str) -> Set[str]:
        """Load checkpoint if it exists."""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Find latest checkpoint
        checkpoints = list(CHECKPOINT_DIR.glob(f"{collection}_checkpoint_*.json"))
        if not checkpoints:
            return set()
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading checkpoint: {latest}")
        
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('processed_refs', []))
    
    def save_checkpoint(self, collection: str):
        """Save extraction checkpoint."""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = CHECKPOINT_DIR / f"{collection}_checkpoint_{timestamp}.json"
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'collection': collection,
            'processed_refs': list(self.processed_refs),
            'extracted_count': len(self.results)
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
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
    
    def is_greek_text(self, text: str) -> bool:
        """Check if text contains sufficient Greek characters."""
        if not text:
            return False
        
        greek_count = sum(
            1 for char in text
            if '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF'
        )
        
        ratio = greek_count / len(text)
        return ratio >= self.config['min_greek_ratio']
    
    def normalize_greek(self, text: str) -> str:
        """Normalize Greek text to NFC form."""
        return unicodedata.normalize('NFC', text)
    
    async def extract_from_url(self, url: str) -> Optional[Dict[str, str]]:
        """Extract Greek and Italian text from a URL."""
        for attempt in range(self.config['max_retries']):
            try:
                async with self.session.get(url) as response:
                    if response.status != 200:
                        logger.debug(f"HTTP {response.status} for {url}")
                        return None
                    
                    content = await response.text()
                    
                    # Parse based on URL type
                    if '.plain.html' in url:
                        return self.extract_from_plain_html(content)
                    elif 'agora_show_transcription' in url:
                        return self.extract_from_transcription(content)
                    else:
                        return self.extract_from_html(content)
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error extracting from {url}: {e}")
                return None
        
        return None
    
    def extract_from_plain_html(self, html: str) -> Optional[Dict[str, str]]:
        """Extract and separate Greek, Italian, and Latin text from plain HTML format."""
        soup = BeautifulSoup(html, 'html.parser')
        
        result = {
            'greek_text': '',
            'italian_text': '',
            'latin_text': '',
            'source_citation': ''
        }
        
        # Look for the main content div
        content_div = soup.find('div', class_='pundit-content')
        if not content_div:
            content_div = soup.find('body')
        
        if content_div:
            # Find table cells - usually column 1 is Italian, column 2 is Greek/Latin
            table = content_div.find('table')
            if table:
                cells = table.find_all('td')
                if len(cells) >= 2:
                    # Left column (Italian)
                    italian_cell = cells[0]
                    italian_text = self.extract_text_from_element(italian_cell, 'italian')
                    
                    # Right column (Greek/Latin + apparatus)
                    ancient_cell = cells[1]
                    ancient_text = self.extract_text_from_element(ancient_cell, 'ancient')
                    
                    # Extract source citation (e.g., "ALCAEUS fr. 80 Diehl")
                    citation_match = re.search(r'([A-Z]+(?:\s+\w+)*\s+fr\.\s+\d+\s+\w+)', italian_text)
                    if citation_match:
                        result['source_citation'] = citation_match.group(1)
                    
                    result['italian_text'] = self.clean_italian_text(italian_text)
                    
                    # Separate Greek and Latin from ancient text
                    greek, latin = self.separate_greek_and_latin(ancient_text)
                    result['greek_text'] = self.clean_greek_text(greek)
                    result['latin_text'] = self.clean_latin_text(latin)
            else:
                # Fallback: try to separate by language detection
                full_text = content_div.get_text(strip=True)
                result['greek_text'], result['latin_text'], result['italian_text'] = self.separate_all_languages(full_text)
        
        # Return if any ancient text found (Greek OR Latin)
        if result['greek_text'] or result['latin_text']:
            return result
        
        return None
    
    def extract_text_from_element(self, element, target_lang='ancient'):
        """Extract text from HTML element, focusing on target language."""
        texts = []
        
        # Look for elements with class="greek" for Greek text
        if target_lang in ['greek', 'ancient']:
            # Get all text that might be Greek or Latin
            greek_elements = element.find_all(class_='greek')
            for elem in greek_elements:
                text = elem.get_text(strip=True)
                if text:
                    texts.append(text)
            
            # Also check for Greek/Latin in font tags
            for font in element.find_all('font'):
                if font.get('face') == '' and font.get('class') == 'greek':
                    texts.append(font.get_text(strip=True))
                # Some Latin might be unmarked
                elif not font.get('class'):
                    text = font.get_text(strip=True)
                    if text and not text.startswith('['):
                        texts.append(text)
        else:
            # For Italian, get text but exclude Greek/Latin elements
            for item in element.descendants:
                if isinstance(item, NavigableString):
                    parent = item.parent
                    if not (parent and parent.get('class') == ['greek']):
                        text = str(item).strip()
                        if text and not text.startswith('['):
                            texts.append(text)
        
        return ' '.join(texts)
    
    def separate_greek_and_latin(self, text: str) -> Tuple[str, str]:
        """Separate Greek from Latin text using character detection."""
        greek_parts = []
        latin_parts = []
        
        # Split into words/segments
        segments = text.split()
        
        for segment in segments:
            if self.has_greek_chars(segment):
                greek_parts.append(segment)
            elif self.has_latin_chars(segment):
                latin_parts.append(segment)
            # If mixed, prefer Greek
            elif self.has_greek_chars(segment) and self.has_latin_chars(segment):
                greek_parts.append(segment)
        
        return ' '.join(greek_parts), ' '.join(latin_parts)
    
    def separate_all_languages(self, text: str) -> Tuple[str, str, str]:
        """Separate Greek, Latin, and Italian text using character detection."""
        greek_parts = []
        latin_parts = []
        italian_parts = []
        
        # Split into words/segments
        segments = text.split()
        
        for segment in segments:
            if self.has_greek_chars(segment):
                greek_parts.append(segment)
            elif self.is_likely_latin(segment):
                latin_parts.append(segment)
            else:
                italian_parts.append(segment)
        
        return ' '.join(greek_parts), ' '.join(latin_parts), ' '.join(italian_parts)
    
    def has_latin_chars(self, text: str) -> bool:
        """Check if text contains Latin characters (basic Latin alphabet)."""
        # Check for Latin characters without diacritics that are common in Latin texts
        latin_pattern = re.compile(r'[a-zA-Z]')
        return bool(latin_pattern.search(text)) and not self.has_greek_chars(text)
    
    def is_likely_latin(self, text: str) -> bool:
        """Determine if text is likely Latin vs Italian."""
        # Common Latin words and endings
        latin_indicators = [
            'que', 'enim', 'autem', 'sed', 'cum', 'quod', 'qui', 'quae',
            'ibus', 'orum', 'arum', 'erum', 'atur', 'itur', 'untur'
        ]
        text_lower = text.lower()
        
        # Check for Latin indicators
        for indicator in latin_indicators:
            if indicator in text_lower or text_lower.endswith(indicator):
                return True
        
        # If it's all caps or has specific patterns, might be Latin
        if text.isupper() and len(text) > 2:
            return True
            
        return False
    
    def has_greek_chars(self, text: str) -> bool:
        """Check if text contains Greek characters."""
        for char in text:
            if '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF':
                return True
        return False
    
    def clean_greek_text(self, text: str) -> str:
        """Clean and normalize Greek text."""
        # Remove reference numbers like [I 2. 20]
        text = re.sub(r'\[I\s*\d+\.\s*\d+.*?\]', '', text)
        # Remove apparatus markers
        text = re.sub(r'\bv\.\s*s\.\b', '', text)
        # Remove angle brackets (textual reconstructions)
        text = re.sub(r'[〈〉⌊⌋]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        # Normalize to NFC
        return unicodedata.normalize('NFC', text).strip()
    
    def clean_latin_text(self, text: str) -> str:
        """Clean Latin text."""
        # Remove reference numbers
        text = re.sub(r'\[I\s*\d+\.\s*\d+.*?\]', '', text)
        # Remove apparatus markers
        text = re.sub(r'\bv\.\s*s\.\b', '', text)
        # Remove angle brackets
        text = re.sub(r'[〈〉⌊⌋]', '', text)
        # Normalize whitespace
        return ' '.join(text.split()).strip()
    
    def clean_italian_text(self, text: str) -> str:
        """Clean Italian translation text."""
        # Remove reference numbers
        text = re.sub(r'\d+\s*[A-Z]\s*\d+\.?', '', text)
        # Remove font size markers
        text = re.sub(r'<font[^>]*>|</font>', '', text)
        # Normalize whitespace
        return ' '.join(text.split()).strip()
    
    def extract_from_transcription(self, html: str) -> Optional[Dict[str, str]]:
        """Extract text from agora transcription format."""
        # For now, use the same logic as plain HTML
        return self.extract_from_plain_html(html)
    
    def extract_from_html(self, html: str) -> Optional[Dict[str, str]]:
        """Extract text from regular HTML format."""
        return self.extract_from_plain_html(html)
    
    async def extract_fragment(self, fragment: Dict) -> Optional[ExtractedText]:
        """Extract a single fragment."""
        reference = fragment['reference']
        
        # Skip if already processed
        if reference in self.processed_refs:
            return None
        
        # Get philosopher from reference using config mapping
        chapter_num = reference.split('-')[0]
        philosopher = self.config.get('philosopher_map', {}).get(
            int(chapter_num), 
            fragment.get('philosopher', f'Chapter {chapter_num}')
        )
        
        # Try different URL sources in order of preference
        urls_to_try = []
        
        if fragment.get('plain_url'):
            urls_to_try.append(fragment['plain_url'])
        if fragment.get('transcription_url'):
            urls_to_try.append(fragment['transcription_url'])
        if fragment.get('html_url'):
            urls_to_try.append(fragment['html_url'])
        
        extracted_data = None
        successful_url = None
        
        for url in urls_to_try:
            data = await self.extract_from_url(url)
            if data:
                extracted_data = data
                successful_url = url
                break
            await asyncio.sleep(self.config['delay'] / 2)  # Short delay between attempts
        
        if extracted_data and isinstance(extracted_data, dict):
            # Create extracted record with separated texts
            greek_text = extracted_data.get('greek_text', '')
            latin_text = extracted_data.get('latin_text', '')
            
            # Accept if either Greek OR Latin is present
            if greek_text or latin_text:
                # For fragments with only Latin, use Latin as the main text
                main_text = greek_text if greek_text else latin_text
                
                extracted = ExtractedText(
                    reference=reference,
                    collection=fragment['collection'],
                    philosopher=philosopher,
                    greek_text=greek_text,  # May be empty
                    paragraphs=main_text.split('\n\n') if main_text else [],
                    url=successful_url,
                    extraction_date=datetime.now().isoformat()
                )
                
                # Store Latin and Italian translations if present
                if latin_text:
                    extracted.latin_text = latin_text
                if extracted_data.get('italian_text'):
                    extracted.italian_text = extracted_data['italian_text']
                if extracted_data.get('source_citation'):
                    extracted.source_citation = extracted_data['source_citation']
                
                self.processed_refs.add(reference)
                return extracted
        elif extracted_data and isinstance(extracted_data, str):
            # Old format - just Greek text
            extracted = ExtractedText(
                reference=reference,
                collection=fragment['collection'],
                philosopher=philosopher,
                greek_text=extracted_data,
                paragraphs=extracted_data.split('\n\n'),
                url=successful_url,
                extraction_date=datetime.now().isoformat()
            )
            self.processed_refs.add(reference)
            return extracted
        
        logger.warning(f"No ancient text (Greek or Latin) found for {reference}")
        self.processed_refs.add(reference)  # Mark as processed even if failed
        return None
    
    def save_results(self, collection: str):
        """Save extraction results."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # JSON output with separated texts
        json_file = OUTPUT_DIR / f"{collection}_texts.json"
        json_data = []
        for r in self.results:
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
        
        # 1. Greek-only text output (no Italian, no Latin)
        greek_only_file = OUTPUT_DIR / f"{collection}_greek_only.txt"
        greek_count = 0
        with open(greek_only_file, 'w', encoding='utf-8') as f:
            for r in self.results:
                if r.greek_text:  # Only if Greek text exists
                    greek_count += 1
                    f.write(f"=== {r.reference} ({r.philosopher}) ===\n")
                    if hasattr(r, 'source_citation') and r.source_citation:
                        f.write(f"Source: {r.source_citation}\n")
                    f.write(f"{r.greek_text}\n\n")
        
        # 2. Ancient texts only (Greek + Latin in order, no Italian)
        ancient_file = OUTPUT_DIR / f"{collection}_ancient.txt"
        with open(ancient_file, 'w', encoding='utf-8') as f:
            for r in self.results:
                # Only write if there's Greek or Latin (skip Italian-only entries)
                if r.greek_text or (hasattr(r, 'latin_text') and r.latin_text):
                    f.write(f"=== {r.reference} ({r.philosopher}) ===\n")
                    if hasattr(r, 'source_citation') and r.source_citation:
                        f.write(f"Source: {r.source_citation}\n")
                    
                    # Write Greek if present
                    if r.greek_text:
                        f.write(f"{r.greek_text}\n")
                    
                    # Write Latin if present (will appear after Greek if both exist)
                    if hasattr(r, 'latin_text') and r.latin_text:
                        if r.greek_text:  # Add separator if Greek was written
                            f.write("---\n")
                        f.write(f"{r.latin_text}\n")
                    
                    f.write("\n")
        
        # 3. Full output (everything including Italian translations)
        full_file = OUTPUT_DIR / f"{collection}_full.txt"
        with open(full_file, 'w', encoding='utf-8') as f:
            for r in self.results:
                f.write(f"=== {r.reference} ({r.philosopher}) ===\n")
                if hasattr(r, 'source_citation') and r.source_citation:
                    f.write(f"Source: {r.source_citation}\n")
                
                # Ancient texts first
                if r.greek_text:
                    f.write(f"Greek: {r.greek_text}\n")
                if hasattr(r, 'latin_text') and r.latin_text:
                    f.write(f"Latin: {r.latin_text}\n")
                
                # Italian translation last
                if hasattr(r, 'italian_text') and r.italian_text:
                    f.write(f"Italian: {r.italian_text}\n")
                
                f.write("\n")
        
        # Statistics
        latin_count = sum(1 for r in self.results if hasattr(r, 'latin_text') and r.latin_text)
        italian_count = sum(1 for r in self.results if hasattr(r, 'italian_text') and r.italian_text)
        
        stats_file = OUTPUT_DIR / f"{collection}_stats.json"
        stats = {
            'collection': collection,
            'extraction_date': datetime.now().isoformat(),
            'total_extracted': len(self.results),
            'with_greek': greek_count,
            'with_latin': latin_count,
            'with_italian': italian_count,
            'greek_only': sum(1 for r in self.results if r.greek_text and not (hasattr(r, 'latin_text') and r.latin_text)),
            'latin_only': sum(1 for r in self.results if not r.greek_text and hasattr(r, 'latin_text') and r.latin_text),
            'both_greek_latin': sum(1 for r in self.results if r.greek_text and hasattr(r, 'latin_text') and r.latin_text),
            'philosophers': list(set(r.philosopher for r in self.results)),
            'references': [r.reference for r in self.results]
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved {len(self.results)} texts to {OUTPUT_DIR}")
        logger.info(f"Greek: {greek_count}, Latin: {latin_count}, Italian: {italian_count}")
        logger.info(f"Greek only: {stats['greek_only']}, Latin only: {stats['latin_only']}, Both: {stats['both_greek_latin']}")
    
    async def run(self, collection: str = 'presocratics', resume: bool = True):
        """Run extraction."""
        await self.setup()
        
        try:
            # Load catalogue
            fragments = self.load_catalogue(collection)
            logger.info(f"Loaded catalogue with {len(fragments)} fragments")
            
            # Load checkpoint if resuming
            if resume:
                self.processed_refs = self.load_checkpoint(collection)
                logger.info(f"Resuming: {len(self.processed_refs)} already processed")
            
            # Process fragments
            for i, fragment in enumerate(fragments):
                if fragment['reference'] in self.processed_refs:
                    continue
                
                logger.info(f"[{i+1}/{len(fragments)}] Extracting {fragment['reference']}")
                
                extracted = await self.extract_fragment(fragment)
                if extracted:
                    self.results.append(extracted)
                
                # Save checkpoint periodically
                if (i + 1) % 50 == 0:
                    self.save_checkpoint(collection)
                    self.save_results(collection)
                
                await asyncio.sleep(self.config['delay'])
            
            # Final save
            self.save_checkpoint(collection)
            self.save_results(collection)
            
            logger.info(f"Extraction complete: {len(self.results)} texts extracted")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user, saving progress...")
            self.save_checkpoint(collection)
            self.save_results(collection)
            
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Daphnet Extraction')
    parser.add_argument('--collection', default='presocratics',
                       choices=['presocratics', 'socratics', 'laertius', 'sextus'])
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh, ignore checkpoint')
    
    args = parser.parse_args()
    
    extractor = UnifiedExtractor()
    await extractor.run(
        collection=args.collection,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    asyncio.run(main())
