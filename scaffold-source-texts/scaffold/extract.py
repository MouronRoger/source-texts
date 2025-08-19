#!/usr/bin/env python3
"""
Daphnet Presocratics Extraction - Enhanced Version
With improved crawling logic, RDF support, and robustness
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import signal
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
import yaml
from bs4 import BeautifulSoup
from http_utils import build_default_headers

# ---------------------------------------------------------------------------
# Plain-HTML helpers
# ---------------------------------------------------------------------------


def _derive_plain_url(url: str) -> str:
    """Return the corresponding ``.plain.html`` URL for a given Daphnet page.

    Rules follow Muruca conventions:

    1. ``*.rdf`` → replace extension with ``.plain.html``
    2. ``*.html`` → replace extension with ``.plain.html``
    3. no extension  → append ``.plain.html``
    4. Already ``.plain.html`` → unchanged
    """

    if url.endswith(".plain.html"):
        return url
    if url.endswith(".rdf"):
        return url[:-4] + ".plain.html"
    if url.endswith(".html"):
        return url[:-5] + ".plain.html"
    return url + ".plain.html"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for interrupt handling
RUN_STATE = {"extractor": None}


@dataclass
class Record:
    """Represents an extracted record with full metadata."""
    url: str
    rdf_url: Optional[str]
    collection: str
    reference: str
    dk_reference: Optional[str]
    philosopher: Optional[str]
    greek_text: Optional[str]
    paragraphs: List[str]
    status: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    final_url: Optional[str] = None
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class DaphnetExtractor:
    """Enhanced extractor with RDF support and better robustness."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        args: Optional[argparse.Namespace] = None,
    ):
        """Create extractor.

        Parameters
        ----------
        config_path
            Optional location of the YAML configuration. When *None* (the
            default) the file ``config.yaml`` located next to this script is
            used.  This makes the extractor runnable from any working
            directory without requiring *cd* into the `scaffold` folder.
        args
            Parsed command-line arguments.
        """

        if config_path is None:
            config_path = Path(__file__).with_name("config.yaml")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.args = args or argparse.Namespace()
        self.base_url = self.config['source']['base_url']
        self.user_agent = self.config['source']['user_agent']
        self.philosopher_map = self.config['philosopher_map']
        
        # Rate limiting with CLI override
        rate_config = self.config["extraction"]["rate_limit"]

        cli_delay = getattr(args, "delay", None)
        # If the CLI flag was omitted we keep the YAML default; otherwise use
        # the user-provided value (even if it is 0.0 for debugging).
        self.base_delay = rate_config["base_delay"] if cli_delay is None else cli_delay
        self.jitter = rate_config['jitter']
        self.max_backoff = rate_config['max_backoff']

        # Progress logging
        self.progress_freq: int = self.config["extraction"].get(
            "progress_log_frequency",
            10,
        )
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.records: List[Record] = []
        self.checkpoint_counter = 0
        
        # Ensure output directories exist in repo
        self.output_dir = Path(self.config['output']['directory'])
        self.checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set global reference for signal handler
        RUN_STATE["extractor"] = self
        
    async def setup(self) -> None:
        """Setup HTTP session with proper headers."""
        timeout = aiohttp.ClientTimeout(
            total=self.config['extraction']['rate_limit']['timeout']
        )
        self.session = aiohttp.ClientSession(
            headers=build_default_headers(self.user_agent),
            timeout=timeout
        )
        
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def rate_limit(self) -> None:
        """Enforce rate limiting with jitter."""
        delay = self.base_delay + random.random() * self.jitter
        await asyncio.sleep(delay)
    
    def is_rdf_response(self, headers: Dict[str, str]) -> bool:
        """Check if response is RDF based on content type."""
        ctype = headers.get('Content-Type', '').lower()
        return any(t in ctype for t in ['application/rdf+xml', 'text/xml', '+xml'])
    
    def looks_like_rdf_xml(self, body: bytes) -> bool:
        """Validate that content is actually RDF XML."""
        try:
            root = ET.fromstring(body)
            tag = root.tag.lower()
            return 'rdf' in tag or tag.endswith('rdf')
        except Exception:
            return False
    
    async def fetch_with_retry(self, url: str) -> Optional[Tuple[int, Dict[str, str], bytes]]:
        """Fetch URL with retry logic and backoff."""
        attempt = 0
        backoff = 2.0
        
        while attempt < self.config['extraction']['rate_limit']['max_retries']:
            try:
                await self.rate_limit()
                logger.debug("Fetching %s", url)
                
                assert self.session
                async with self.session.get(url, allow_redirects=True) as response:
                    body = await response.read()
                    logger.debug("%s -> %s", url, response.status)
                    headers = dict(response.headers)
                    
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = headers.get('Retry-After')
                        if retry_after and retry_after.isdigit():
                            wait = float(retry_after)
                        else:
                            wait = backoff + random.random()
                        
                        logger.warning(f"Rate limited, waiting {wait:.1f}s")
                        await asyncio.sleep(min(wait, self.max_backoff))
                        backoff = min(backoff * 2, self.max_backoff)
                        attempt += 1
                        continue
                    
                    # Handle server errors with retry
                    if response.status in self.config['extraction']['retry_on_status']:
                        await asyncio.sleep(backoff + random.random())
                        backoff = min(backoff * 2, self.max_backoff)
                        attempt += 1
                        continue
                    
                    return response.status, headers, body
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url}")
                attempt += 1
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
        
        return None
    
    async def extract_record(self, url: str, collection: str, reference: str) -> Optional[Record]:
        """Extract a single record (HTML or RDF)."""
        result = await self.fetch_with_retry(url)
        if not result:
            return None
        
        status_code, headers, body = result
        
        # Handle redirects as soft success
        if status_code in self.config['extraction']['redirect_status']:
            final_url = headers.get('Location', url)
        else:
            final_url = url

        # We'll potentially update this variable if we succeed in fetching
        # a corresponding ``.plain.html`` transcription.
        final_url_used = final_url
        
        if status_code != 200:
            return None
        
        # Determine if this is RDF
        is_rdf = url.endswith('.rdf')
        
        if is_rdf:
            # Validate RDF response
            if not self.is_rdf_response(headers) or not self.looks_like_rdf_xml(body):
                logger.warning(f"Invalid RDF response for {url}")
                return None
            
            # For RDF, we'll also try to get the HTML version
            html_url = url[:-4] if url.endswith('.rdf') else url
            greek_text, paragraphs = await self.extract_greek_from_html(html_url)
        else:
            # Prefer the lightweight plain transcription if available
            greek_text: Optional[str]
            paragraphs: List[str]

            plain_url = _derive_plain_url(url)
            final_url_used = url

            if plain_url != url:
                plain_result = await self.fetch_with_retry(plain_url)
                if plain_result:
                    status_p, hdr_p, body_p = plain_result
                    if (
                        status_p == 200
                        and "html" in hdr_p.get("Content-Type", "").lower()
                    ):
                        greek_text, paragraphs = self.parse_html_content(body_p)
                        if greek_text:
                            final_url_used = plain_url
                        else:
                            greek_text, paragraphs = self.parse_html_content(body)
                    else:
                        greek_text, paragraphs = self.parse_html_content(body)
            else:
                greek_text, paragraphs = self.parse_html_content(body)
        
        # Build record
        dk_reference = reference if collection == 'presocratics' else None
        philosopher = None
        
        if dk_reference and collection == 'presocratics':
            # Parse chapter from reference
            try:
                chapter = int(reference.split('-')[0])
                philosopher = self.philosopher_map.get(chapter, f"Unknown (Ch. {chapter})")
            except Exception:
                pass
        
        return Record(
            url=url if not is_rdf else url[:-4],
            rdf_url=url if is_rdf else None,
            collection=collection,
            reference=reference,
            dk_reference=dk_reference,
            philosopher=philosopher,
            greek_text=greek_text,
            paragraphs=paragraphs,
            status='ok' if greek_text else 'no_content',
            etag=headers.get('ETag'),
            last_modified=headers.get('Last-Modified'),
            final_url=final_url_used
        )
    
    async def extract_greek_from_html(self, url: str) -> Tuple[Optional[str], List[str]]:
        """Extract Greek text from HTML URL."""
        result = await self.fetch_with_retry(url)
        if not result:
            return None, []
        
        status_code, headers, body = result
        if status_code != 200:
            return None, []
        
        return self.parse_html_content(body)
    
    def parse_html_content(self, body: bytes) -> Tuple[Optional[str], List[str]]:
        """Parse HTML to extract Greek text."""
        try:
            soup = BeautifulSoup(body, 'html.parser')
            paragraphs = []
            
            # Try primary selectors
            for selector in self.config['extraction']['selectors']['primary']:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(strip=True)
                    if self.has_greek(text):
                        # Extract paragraphs
                        for p in elem.find_all(['p', 'div']):
                            p_text = p.get_text(strip=True)
                            if p_text and self.has_greek(p_text):
                                paragraphs.append(p_text)
                        
                        if not paragraphs and text:
                            paragraphs.append(text)
            
            # Fallback selectors
            if not paragraphs:
                for selector in self.config['extraction']['selectors']['fallback']:
                    elements = soup.select(selector)
                    for elem in elements:
                        text = elem.get_text(strip=True)
                        if self.has_greek(text):
                            paragraphs.append(text)
            
            if paragraphs:
                greek_text = '\n\n'.join(paragraphs)
                greek_text = unicodedata.normalize('NFC', greek_text)
                return greek_text, paragraphs
            
            return None, []
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return None, []
    
    def has_greek(self, text: str) -> bool:
        """Check if text contains Greek characters."""
        if not text:
            return False
        greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
        return greek_chars / len(text) >= self.config['extraction']['content_validation']['greek_threshold']
    
    def should_break_after_misses(
        self,
        found_any: bool,
        consecutive_misses: int,
        threshold: int,
    ) -> bool:
        """Stop only after *threshold* consecutive misses *after* at least one hit."""

        return found_any and consecutive_misses >= threshold
    
    async def scan_presocratics(self) -> None:
        """Scan Presocratics collection with smart stopping."""
        logger.info("Scanning Presocratics...")
        config = self.config['source']['collections']['presocratics']
        threshold = config['consecutive_miss_threshold']
        
        start_ch = getattr(self.args, 'start_ch', config['chapters'][0])
        end_ch = getattr(self.args, 'end_ch', config['chapters'][1])
        
        for chapter in range(start_ch, end_ch + 1):
            for ftype in config['types']:
                consecutive_misses = 0
                found_any = False
                number = 1
                
                while number <= config['max_number']:
                    # Build list of suffixes to probe ("" for base)
                    suffixes_to_check = [""] + config.get("letter_suffixes", [])

                    any_success_this_number = False

                    for suffix in suffixes_to_check:
                        ref_part = f"{number}{suffix}" if suffix else f"{number}"
                        reference = f"{chapter}-{ftype},{ref_part}"
                        url = f"{self.base_url}/texts/Presocratics/{reference}"

                        target_url = f"{url}.rdf" if self.config["output"]["formats"]["rdf"]["enabled"] else url

                        record = await self.extract_record(target_url, "presocratics", reference)

                        if record and record.status == "ok":
                            self.records.append(record)
                            found_any = True
                            any_success_this_number = True
                            consecutive_misses = 0
                        else:
                            # Only break suffix loop after first miss to avoid endless letters
                            if suffix:
                                break

                    if not any_success_this_number:
                        consecutive_misses += 1
                        if self.should_break_after_misses(found_any, consecutive_misses, threshold):
                            logger.info(
                                "Stopping %s-%s after %s consecutive misses",
                                chapter,
                                ftype,
                                consecutive_misses,
                            )
                            break
                    
                    # Checkpoint periodically
                    await self.checkpoint_if_needed()
                    
                    if number % self.progress_freq == 0:
                        logger.info(
                            "[Presocratics] ch=%s type=%s n=%s records=%s",
                            chapter,
                            ftype,
                            number,
                            len(self.records),
                        )
                    
                    number += 1
    
    async def scan_catalogue(self, collection: str) -> None:
        """Generic catalogue-based scanning for *collection*."""

        cat_path = Path(self.config["output"]["directory"]) / f"{collection}_catalogue.json"
        if not cat_path.exists():
            logger.error("Catalogue file not found: %s", cat_path)
            return

        logger.info("Scanning %s via catalogue …", collection.capitalize())
        with cat_path.open("r", encoding="utf-8") as fh:
            rows: List[Dict[str, Any]] = json.load(fh)

        total = len(rows)
        for idx, row in enumerate(rows, 1):
            reference = row.get("reference") or row.get("siglum")
            target_url = (
                row["rdf_url"]
                if self.config["output"]["formats"]["rdf"]["enabled"]
                else row["plain_url"]
            )

            record = await self.extract_record(target_url, collection, reference)
            if record and record.status == "ok":
                self.records.append(record)

            await self.checkpoint_if_needed()
            if idx % self.progress_freq == 0:
                logger.info(
                    "[Catalogue %s] %s/%s records=%s",
                    collection,
                    idx,
                    total,
                    len(self.records),
                )

    # Redirect old method to generic implementation
    async def scan_presocratics_catalogue(self) -> None:  # noqa: D401
        await self.scan_catalogue("presocratics")
    
    async def checkpoint_if_needed(self) -> None:
        """Save checkpoint if needed."""
        self.checkpoint_counter += 1
        frequency = self.config['output']['formats']['json']['checkpoint_frequency']
        
        if self.checkpoint_counter % frequency == 0:
            self.save_checkpoint()
    
    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(self.records),
            'records': [r.to_dict() for r in self.records]
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def save_final_output(self) -> None:
        """Save all output formats."""
        # JSON output
        if self.config['output']['formats']['json']['enabled']:
            json_path = self.output_dir / self.config['output']['formats']['json']['filename']
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [r.to_dict() for r in self.records],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        
        # Text corpus
        if self.config['output']['formats']['text']['enabled']:
            text_path = self.output_dir / self.config['output']['formats']['text']['filename']
            with open(text_path, 'w', encoding='utf-8') as f:
                for record in self.records:
                    if record.greek_text:
                        f.write(f"[{record.reference}] {record.philosopher or 'Unknown'}\n\n")
                        f.write(record.greek_text)
                        f.write("\n\n---\n\n")
        
        # Coverage report
        if self.config['output']['formats']['coverage']['enabled']:
            coverage = self.generate_coverage_report()
            coverage_path = self.output_dir / self.config['output']['formats']['coverage']['filename']
            with open(coverage_path, 'w', encoding='utf-8') as f:
                json.dump(coverage, f, indent=2)
    
    def _catalogue_length(self, collection: str) -> Optional[int]:
        """Return number of entries in <collection>_catalogue.json if present."""

        cat_path = (
            Path(self.config["output"]["directory"]) / f"{collection}_catalogue.json"
        )
        if not cat_path.exists():
            return None
        try:
            with cat_path.open("r", encoding="utf-8") as fh:
                return len(json.load(fh))
        except Exception:
            return None

    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate coverage statistics."""
        by_collection = {}
        for record in self.records:
            by_collection.setdefault(record.collection, 0)
            if record.status == "ok":
                by_collection[record.collection] += 1

        coverage_details: Dict[str, Any] = {}
        for coll, success_count in by_collection.items():
            total_expected = self._catalogue_length(coll)
            if total_expected is not None and total_expected > 0:
                ratio = success_count / total_expected
            else:
                ratio = None
            coverage_details[coll] = {
                "expected": total_expected,
                "extracted": success_count,
                "ratio": ratio,
            }

        return {
            "extraction_date": datetime.now().isoformat(),
            "total_records": len(self.records),
            "successful_records": sum(1 for r in self.records if r.status == "ok"),
            "by_collection": coverage_details,
            "collections_attempted": list(by_collection.keys()),
        }
    
    async def run(self) -> None:
        """Run the extraction process."""
        try:
            await self.setup()
            
            # Determine which collections to scan
            collections = getattr(self.args, "collections", ["presocratics"])

            for collection in collections:
                cat_path = Path(self.config["output"]["directory"]) / f"{collection}_catalogue.json"

                if getattr(self.args, "from_catalogue", False) or cat_path.exists():
                    await self.scan_catalogue(collection)
                else:
                    if collection == "presocratics":
                        await self.scan_presocratics()
                    else:
                        logger.warning(
                            "No catalogue found and no brute-force scanner for %s; skipping.",
                            collection,
                        )
            
            self.save_final_output()
            
            coverage = self.generate_coverage_report()
            logger.info("Extraction complete:")
            logger.info(f"  Total records: {coverage['total_records']}")
            logger.info(f"  Successful: {coverage['successful_records']}")
            logger.info(f"  Collections: {coverage['collections_attempted']}")
            
        finally:
            await self.cleanup()


def handle_interrupt(signum, frame):
    """Handle interrupt signal."""
    logger.info("\nInterrupt received, saving checkpoint...")
    if RUN_STATE["extractor"]:
        RUN_STATE["extractor"].save_checkpoint()
        RUN_STATE["extractor"].save_final_output()
    raise SystemExit(130)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Daphnet Extraction')
    parser.add_argument('--collections', nargs='+', 
                       default=['presocratics'],
                       choices=['presocratics', 'laertius', 'socratics', 'sextus'],
                       help='Collections to extract')
    parser.add_argument('--delay', type=float, 
                       help='Override request delay (seconds)')
    parser.add_argument('--start-ch', type=int, default=1,
                       help='Starting chapter for Presocratics')
    parser.add_argument('--end-ch', type=int, default=90,
                       help='Ending chapter for Presocratics')
    parser.add_argument('--test', action='store_true',
                       help='Test mode (limited extraction)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable DEBUG logging')
    parser.add_argument('--from-catalogue', action='store_true',
                       help='Use catalogue JSON instead of brute-force probing')
    return parser.parse_args()


async def main():
    """Main entry point."""
    # Setup signal handler
    signal.signal(signal.SIGINT, handle_interrupt)
    
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if args.test:
        logger.info("Running in test mode")
        args.start_ch = 22
        args.end_ch = 22
        args.collections = ['presocratics']
    
    extractor = DaphnetExtractor(args=args)
    await extractor.run()


if __name__ == "__main__":
    asyncio.run(main())
