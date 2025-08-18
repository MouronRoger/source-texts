#!/usr/bin/env python3
"""
Daphnet Presocratics Extraction Scaffold
Main execution module for systematic extraction of ancient philosophical texts
from the ILIESI-CNR Daphnet database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from urllib.parse import urljoin

import aiohttp
import yaml
from bs4 import BeautifulSoup, Tag
from typing_extensions import NotRequired

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExtractionResult(TypedDict):
    """Type definition for extraction results."""
    url: str
    status: str
    dk_reference: NotRequired[str]
    greek_text: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]
    error: NotRequired[str]
    timestamp: str


class ProgressData(TypedDict):
    """Type definition for progress tracking."""
    project: str
    started: Optional[str]
    last_updated: Optional[str]
    status: str
    phases: Dict[str, Any]
    extraction_stats: Dict[str, Any]
    checkpoints: List[str]
    errors: List[Dict[str, str]]


@dataclass
class DKReference:
    """Represents a Diels-Kranz reference."""
    chapter: int
    type: str  # A, B, or C
    number: int
    
    def __str__(self) -> str:
        return f"{self.chapter}-{self.type},{self.number}"
    
    @classmethod
    def from_string(cls, ref: str) -> Optional[DKReference]:
        """Parse a DK reference string."""
        import re
        match = re.match(r"(\d+)-([ABC]),(\d+)", ref)
        if match:
            return cls(
                chapter=int(match.group(1)),
                type=match.group(2),
                number=int(match.group(3))
            )
        return None


class DaphnetExtractor:
    """Main extractor class for Daphnet ancient texts."""
    
    def __init__(self, config_path: Path) -> None:
        """Initialize the extractor with configuration."""
        self.config = self._load_configs(config_path)
        self.base_url = "http://ancientsource.daphnet.iliesi.cnr.it"
        self.session: Optional[aiohttp.ClientSession] = None
        self.progress: ProgressData = self._load_progress()
        self.rate_limiter = asyncio.Semaphore(1)  # Single concurrent request
        self.last_request_time: float = 0
        
    def _load_configs(self, config_path: Path) -> Dict[str, Any]:
        """Load all configuration files."""
        configs: Dict[str, Any] = {}
        
        config_files = [
            "01_scaffold_specification.yaml",
            "02_scaffold_roadmap.yaml",
            "03_scaffold_runner.yaml"
        ]
        
        for config_file in config_files:
            file_path = config_path / config_file
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    configs[config_file.split('_')[1]] = yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {file_path}")
                
        return configs
    
    def _load_progress(self) -> ProgressData:
        """Load progress tracking data."""
        progress_file = Path("scaffold/progress.json")
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)  # type: ignore
        return self._initialize_progress()
    
    def _initialize_progress(self) -> ProgressData:
        """Initialize fresh progress tracking."""
        return {
            "project": "Daphnet Presocratics Complete Extraction",
            "started": datetime.now().isoformat(),
            "last_updated": None,
            "status": "in_progress",
            "phases": {},
            "extraction_stats": {
                "total_items_processed": 0,
                "successful_extractions": 0,
                "failed_extractions": 0,
                "dk_references_found": {},
                "collections": {}
            },
            "checkpoints": [],
            "errors": []
        }
    
    def _save_progress(self) -> None:
        """Save current progress to file."""
        self.progress["last_updated"] = datetime.now().isoformat()
        progress_file = Path("scaffold/progress.json")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    async def _rate_limit(self) -> None:
        """Implement rate limiting between requests."""
        delay = self.config.get("runner", {}).get("crawling", {}).get("rate_limiting", {}).get("requests_per_second", 0.5)
        min_interval = 1.0 / delay
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def setup_session(self) -> None:
        """Setup aiohttp session with proper headers."""
        headers = self.config.get("runner", {}).get("crawling", {}).get("headers", {})
        headers["User-Agent"] = self.config.get("runner", {}).get("crawling", {}).get("user_agent", "Research Bot")
        
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=1)  # Single connection
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=connector
        )
    
    async def close_session(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
    
    def generate_dk_urls(self) -> List[Tuple[DKReference, str]]:
        """Generate all DK reference URLs to check."""
        urls: List[Tuple[DKReference, str]] = []
        
        for chapter in range(1, 91):  # Chapters 1-90
            for dk_type in ['A', 'B', 'C']:
                for number in range(1, 201):  # Up to 200 per type
                    ref = DKReference(chapter, dk_type, number)
                    url = f"{self.base_url}/texts/Presocratics/{ref}"
                    urls.append((ref, url))
        
        return urls
    
    async def extract_content(self, url: str) -> ExtractionResult:
        """Extract content from a single URL."""
        async with self.rate_limiter:
            await self._rate_limit()
            
            try:
                if not self.session:
                    await self.setup_session()
                    
                assert self.session is not None
                
                async with self.session.get(url) as response:
                    if response.status == 404:
                        return {
                            "url": url,
                            "status": "not_found",
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    if response.status != 200:
                        return {
                            "url": url,
                            "status": "error",
                            "error": f"HTTP {response.status}",
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    html = await response.text()
                    return self._parse_html_content(url, html)
                    
            except asyncio.TimeoutError:
                return {
                    "url": url,
                    "status": "timeout",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error extracting {url}: {e}")
                return {
                    "url": url,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    def _parse_html_content(self, url: str, html: str) -> ExtractionResult:
        """Parse HTML content to extract Greek text and metadata."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract DK reference from URL
        dk_ref_match = DKReference.from_string(url.split('/')[-1])
        dk_reference = str(dk_ref_match) if dk_ref_match else ""
        
        # Extract Greek text
        greek_text = self._extract_greek_text(soup)
        
        # Extract metadata
        metadata = self._extract_metadata(soup)
        
        if greek_text:
            return {
                "url": url,
                "status": "success",
                "dk_reference": dk_reference,
                "greek_text": greek_text,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "url": url,
                "status": "no_content",
                "dk_reference": dk_reference,
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_greek_text(self, soup: BeautifulSoup) -> str:
        """Extract Greek text from HTML."""
        greek_parts: List[str] = []
        
        # Try various selectors
        selectors = [
            'div[lang="grc"]',
            '.greek-text',
            'p',
            'div'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                if isinstance(elem, Tag):
                    text = elem.get_text(strip=True)
                    if self._contains_greek(text):
                        greek_parts.append(text)
        
        return ' '.join(greek_parts)
    
    def _contains_greek(self, text: str) -> bool:
        """Check if text contains significant Greek content."""
        if not text:
            return False
        
        greek_count = sum(1 for char in text 
                         if '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF')
        
        return greek_count > len(text) * 0.1  # At least 10% Greek
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata: Dict[str, Any] = {}
        
        # Try to extract various metadata
        title_elem = soup.find('title')
        if title_elem:
            metadata['title'] = title_elem.get_text(strip=True)
        
        # Look for DC metadata
        for meta in soup.find_all('meta'):
            name = meta.get('name', '')
            if name.startswith('dc.'):
                metadata[name] = meta.get('content', '')
        
        return metadata
    
    async def run_extraction(self, test_mode: bool = False) -> None:
        """Run the main extraction process."""
        logger.info("Starting Daphnet extraction...")
        
        try:
            await self.setup_session()
            
            # Generate URLs
            all_urls = self.generate_dk_urls()
            
            if test_mode:
                all_urls = all_urls[:10]  # Test with first 10
            
            logger.info(f"Processing {len(all_urls)} URLs...")
            
            results: List[ExtractionResult] = []
            batch_size = 100
            
            for i, (dk_ref, url) in enumerate(all_urls):
                result = await self.extract_content(url)
                results.append(result)
                
                # Update progress
                self.progress["extraction_stats"]["total_items_processed"] += 1
                
                if result["status"] == "success":
                    self.progress["extraction_stats"]["successful_extractions"] += 1
                    dk_ref_str = str(dk_ref)
                    self.progress["extraction_stats"]["dk_references_found"][dk_ref_str] = True
                elif result["status"] == "error":
                    self.progress["extraction_stats"]["failed_extractions"] += 1
                
                # Save progress periodically
                if (i + 1) % batch_size == 0:
                    self._save_progress()
                    self._save_results(results)
                    logger.info(f"Processed {i + 1}/{len(all_urls)} items")
            
            # Final save
            self._save_progress()
            self._save_results(results)
            
            logger.info("Extraction complete!")
            self._generate_report(results)
            
        finally:
            await self.close_session()
    
    def _save_results(self, results: List[ExtractionResult]) -> None:
        """Save extraction results to file."""
        output_dir = Path("data/daphnet")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_file = output_dir / "daphnet_complete.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save as text corpus
        corpus_file = output_dir / "daphnet_corpus.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for result in results:
                if result["status"] == "success" and "greek_text" in result:
                    f.write(f"=== {result.get('dk_reference', 'Unknown')} ===\n")
                    f.write(result["greek_text"])
                    f.write("\n\n")
    
    def _generate_report(self, results: List[ExtractionResult]) -> None:
        """Generate extraction report."""
        report = {
            "total_processed": len(results),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "not_found": sum(1 for r in results if r["status"] == "not_found"),
            "errors": sum(1 for r in results if r["status"] == "error"),
            "dk_coverage": len(self.progress["extraction_stats"]["dk_references_found"])
        }
        
        logger.info(f"Extraction Report: {report}")
        
        report_file = Path("data/daphnet/extraction_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)


async def main() -> None:
    """Main entry point."""
    config_path = Path("scaffold")
    extractor = DaphnetExtractor(config_path)
    
    # Check command line arguments
    test_mode = "--test" in sys.argv
    
    await extractor.run_extraction(test_mode=test_mode)


if __name__ == "__main__":
    asyncio.run(main())
