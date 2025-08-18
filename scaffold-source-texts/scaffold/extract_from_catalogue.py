#!/usr/bin/env python3
"""
Daphnet Smart Extractor
Uses catalogue map to extract actual content efficiently.
"""

import asyncio
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup
import yaml

# Paths
CATALOGUE_FILE = "./data/daphnet/catalogue_map.json"
CONFIG_FILE = "./scaffold/config.yaml"
OUTPUT_DIR = "./data/daphnet"


class SmartExtractor:
    """Extract content using pre-discovered catalogue."""
    
    def __init__(self):
        self.catalogue = self.load_catalogue()
        self.config = self.load_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.results = []
        
    def load_catalogue(self) -> List[Dict]:
        """Load discovered catalogue map."""
        with open(CATALOGUE_FILE, 'r') as f:
            return json.load(f)
    
    def load_config(self) -> Dict:
        """Load extraction config."""
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    
    async def setup(self):
        """Setup session."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': 'Daphnet-Extractor/2.0'},
            timeout=timeout
        )
    
    async def cleanup(self):
        """Cleanup."""
        if self.session:
            await self.session.close()
    
    async def extract_fragment(self, entry: Dict) -> Optional[Dict]:
        """Extract content for a catalogue entry."""
        # Try plain HTML first (cleaner for extraction)
        url = entry.get('plain_url', entry['html_url'])
        
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                    
                html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract Greek text
                greek_text = self.extract_greek(soup)
                if not greek_text:
                    return None
                
                # Normalize Unicode
                greek_text = unicodedata.normalize('NFC', greek_text)
                
                return {
                    'reference': entry['reference'],
                    'collection': entry['collection'],
                    'greek_text': greek_text,
                    'url': entry['html_url'],
                    'philosopher': entry.get('metadata', {}).get('philosopher'),
                }
                
        except Exception as e:
            print(f"Error extracting {entry['reference']}: {e}")
            return None
    
    def extract_greek(self, soup: BeautifulSoup) -> str:
        """Extract Greek text from HTML."""
        # Remove script/style
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Filter for Greek content
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and self.has_significant_greek(line):
                lines.append(line)
        
        return '\n'.join(lines)
    
    def has_significant_greek(self, text: str) -> bool:
        """Check if text has significant Greek content."""
        if len(text) < 3:
            return False
        greek_chars = sum(1 for c in text 
                         if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
        return greek_chars / len(text) > 0.3  # 30% threshold
    
    async def run(self):
        """Run extraction on catalogue."""
        try:
            await self.setup()
            
            # Filter for Presocratics only (or adjust as needed)
            targets = [e for e in self.catalogue if e['collection'] == 'Presocratics']
            
            print(f"Extracting {len(targets)} fragments...")
            
            for i, entry in enumerate(targets):
                result = await self.extract_fragment(entry)
                if result:
                    self.results.append(result)
                
                # Progress
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(targets)}")
                    self.save_results()
                
                # Rate limit
                await asyncio.sleep(2)
            
            self.save_results()
            print(f"Extraction complete: {len(self.results)} fragments")
            
        finally:
            await self.cleanup()
    
    def save_results(self):
        """Save extraction results."""
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # JSON output
        with open(f"{OUTPUT_DIR}/extracted_corpus.json", 'w') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Text corpus
        with open(f"{OUTPUT_DIR}/greek_texts.txt", 'w') as f:
            for r in self.results:
                f.write(f"=== {r['reference']} ===\n")
                if r.get('philosopher'):
                    f.write(f"({r['philosopher']})\n\n")
                f.write(r['greek_text'])
                f.write("\n\n---\n\n")


async def main():
    extractor = SmartExtractor()
    await extractor.run()


if __name__ == "__main__":
    asyncio.run(main())
