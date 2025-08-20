#!/usr/bin/env python3
"""
Greek Text Site Scraper
Scrapes http://217.71.231.54:8080/indices/indexA.htm and converts to JSON

Usage:
    python extract_greek_site.py [--discover-only] [--extract-only] [--max-pages N]
"""

import requests
import json
import time
import re
import logging
import argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Set
import unicodedata

class GreekSiteExtractor:
    def __init__(self, base_url: str = "http://217.71.231.54:8080", delay: float = 2.0):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Greek Text Academic Extractor/1.0 (Research)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Greek Unicode ranges
        self.greek_ranges = [
            (0x0370, 0x03FF),  # Greek and Coptic
            (0x1F00, 0x1FFF),  # Greek Extended
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize data containers
        self.discovered_urls: Set[str] = set()
        self.extracted_texts: List[Dict] = []
        self.failed_urls: List[str] = []
        
    def is_greek_text(self, text: str, min_ratio: float = 0.1) -> bool:
        """Check if text contains sufficient Greek characters"""
        if not text or len(text) < 10:
            return False
            
        greek_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                for start, end in self.greek_ranges:
                    if start <= ord(char) <= end:
                        greek_chars += 1
                        break
        
        if total_chars == 0:
            return False
            
        ratio = greek_chars / total_chars
        return ratio >= min_ratio
    
    def clean_greek_text(self, text: str) -> str:
        """Clean and normalize Greek text"""
        if not text:
            return ""
            
        # Normalize to NFC (composed form)
        text = unicodedata.normalize('NFC', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_text_from_soup(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """Extract Greek text content from BeautifulSoup object"""
        
        # Try different selectors for Greek content
        selectors = [
            '[lang="grc"]',           # Standard Greek language attribute
            '.greek-text',            # Common Greek text class
            '.text-greek',            # Alternative Greek text class
            '.transcription',         # Transcription content
            'p',                      # Paragraph tags
            'div.content',            # Content divs
            'div.text'                # Text divs
        ]
        
        best_content = None
        best_greek_ratio = 0
        
        for selector in selectors:
            elements = soup.select(selector)
            if not elements:
                continue
                
            # Extract text from elements
            text_parts = []
            for elem in elements:
                text = elem.get_text()
                if text.strip():
                    text_parts.append(text.strip())
            
            combined_text = ' '.join(text_parts)
            
            if self.is_greek_text(combined_text):
                # Calculate Greek ratio for this selector
                greek_chars = sum(1 for char in combined_text 
                                if any(start <= ord(char) <= end 
                                     for start, end in self.greek_ranges))
                total_alpha = sum(1 for char in combined_text if char.isalpha())
                greek_ratio = greek_chars / max(total_alpha, 1)
                
                if greek_ratio > best_greek_ratio:
                    best_greek_ratio = greek_ratio
                    best_content = {
                        'text': self.clean_greek_text(combined_text),
                        'selector_used': selector,
                        'greek_ratio': greek_ratio,
                        'paragraphs': [self.clean_greek_text(p.get_text()) 
                                     for p in elements if p.get_text().strip()]
                    }
        
        if best_content:
            # Extract metadata
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Look for reference information
            ref_patterns = [
                r'(\d+-[A-Z],\d+)',           # DK style: 22-B,30
                r'([IVX]+-[A-Z],\d+)',        # Roman: II-A,5
                r'(Book\s+[IVX]+,\s*\d+)',    # Book VII, 123
                r'(\d+\.\d+)',                # Decimal: 1.23
            ]
            
            reference = ""
            page_text = soup.get_text()
            for pattern in ref_patterns:
                match = re.search(pattern, page_text)
                if match:
                    reference = match.group(1)
                    break
            
            return {
                'url': url,
                'title': title,
                'reference': reference,
                'greek_text': best_content['text'],
                'paragraphs': best_content['paragraphs'],
                'selector_used': best_content['selector_used'],
                'greek_ratio': best_content['greek_ratio'],
                'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return None
    
    def discover_urls(self, start_url: str, max_depth: int = 3) -> Set[str]:
        """Discover URLs by crawling from the start page"""
        urls_to_crawl = {start_url}
        discovered = set()
        depth = 0
        
        while urls_to_crawl and depth < max_depth:
            current_level = set(urls_to_crawl)
            urls_to_crawl.clear()
            
            self.logger.info(f"Discovery depth {depth}: crawling {len(current_level)} URLs")
            
            for url in current_level:
                if url in discovered:
                    continue
                    
                try:
                    self.logger.info(f"Discovering: {url}")
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    discovered.add(url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find all links
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        
                        # Convert relative URLs to absolute
                        full_url = urljoin(url, href)
                        
                        # Only include URLs from the same domain
                        if urlparse(full_url).netloc == urlparse(self.base_url).netloc:
                            # Filter for likely content pages
                            if any(pattern in full_url.lower() for pattern in 
                                  ['transcription', 'text', 'fragment', 'book', '.htm', '.html']):
                                if full_url not in discovered:
                                    urls_to_crawl.add(full_url)
                    
                    time.sleep(self.delay)
                    
                except Exception as e:
                    self.logger.error(f"Error discovering {url}: {e}")
                    self.failed_urls.append(url)
            
            depth += 1
        
        self.discovered_urls = discovered
        self.logger.info(f"Discovery complete: found {len(discovered)} URLs")
        return discovered
    
    def extract_from_url(self, url: str) -> Optional[Dict]:
        """Extract Greek text from a single URL"""
        try:
            self.logger.info(f"Extracting: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Try different encodings if needed
            if response.encoding.lower() != 'utf-8':
                response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return self.extract_text_from_soup(soup, url)
            
        except Exception as e:
            self.logger.error(f"Error extracting {url}: {e}")
            self.failed_urls.append(url)
            return None
    
    def extract_all_discovered(self, max_pages: Optional[int] = None) -> List[Dict]:
        """Extract text from all discovered URLs"""
        urls_to_extract = list(self.discovered_urls)
        
        if max_pages:
            urls_to_extract = urls_to_extract[:max_pages]
        
        self.logger.info(f"Extracting from {len(urls_to_extract)} URLs")
        
        extracted_count = 0
        for i, url in enumerate(urls_to_extract):
            self.logger.info(f"Progress: {i+1}/{len(urls_to_extract)} - {url}")
            
            extracted = self.extract_from_url(url)
            if extracted:
                self.extracted_texts.append(extracted)
                extracted_count += 1
                self.logger.info(f"✓ Extracted Greek text (ratio: {extracted['greek_ratio']:.2f})")
            else:
                self.logger.warning(f"✗ No Greek text found")
            
            time.sleep(self.delay)
        
        self.logger.info(f"Extraction complete: {extracted_count}/{len(urls_to_extract)} successful")
        return self.extracted_texts
    
    def save_results(self, output_dir: str = "data/output"):
        """Save extracted results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main JSON file
        json_file = output_path / "greek_site_corpus.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'source_site': self.base_url,
                    'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_texts': len(self.extracted_texts),
                    'total_discovered': len(self.discovered_urls),
                    'failed_urls': len(self.failed_urls)
                },
                'texts': self.extracted_texts
            }, f, ensure_ascii=False, indent=2)
        
        # Save Greek-only text corpus
        greek_file = output_path / "greek_site_texts.txt"
        with open(greek_file, 'w', encoding='utf-8') as f:
            for text in self.extracted_texts:
                f.write(f"# {text['reference']} - {text['title']}\n")
                f.write(f"# Source: {text['url']}\n")
                f.write(f"{text['greek_text']}\n\n")
        
        # Save discovery results
        discovery_file = output_path / "discovered_urls.json"
        with open(discovery_file, 'w', encoding='utf-8') as f:
            json.dump({
                'discovered_urls': list(self.discovered_urls),
                'failed_urls': self.failed_urls,
                'total_discovered': len(self.discovered_urls)
            }, f, indent=2)
        
        # Save statistics
        stats_file = output_path / "extraction_stats.json"
        greek_ratios = [t['greek_ratio'] for t in self.extracted_texts]
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_texts_extracted': len(self.extracted_texts),
                'total_urls_discovered': len(self.discovered_urls),
                'success_rate': len(self.extracted_texts) / max(len(self.discovered_urls), 1),
                'average_greek_ratio': sum(greek_ratios) / max(len(greek_ratios), 1),
                'min_greek_ratio': min(greek_ratios) if greek_ratios else 0,
                'max_greek_ratio': max(greek_ratios) if greek_ratios else 0,
                'failed_extractions': len(self.failed_urls)
            }, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        self.logger.info(f"  - {json_file}: Main JSON corpus")
        self.logger.info(f"  - {greek_file}: Greek text only")
        self.logger.info(f"  - {discovery_file}: URL discovery results")
        self.logger.info(f"  - {stats_file}: Extraction statistics")

def main():
    parser = argparse.ArgumentParser(description='Extract Greek texts from site and convert to JSON')
    parser.add_argument('--discover-only', action='store_true', 
                       help='Only discover URLs, do not extract content')
    parser.add_argument('--extract-only', action='store_true',
                       help='Only extract from previously discovered URLs')
    parser.add_argument('--max-pages', type=int,
                       help='Maximum number of pages to extract (for testing)')
    parser.add_argument('--start-url', default='http://217.71.231.54:8080/indices/indexA.htm',
                       help='Starting URL for discovery')
    parser.add_argument('--output-dir', default='data/output',
                       help='Output directory for results')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between requests in seconds')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = GreekSiteExtractor(delay=args.delay)
    
    # Load previous discovery if extract-only
    if args.extract_only:
        discovery_file = Path(args.output_dir) / "discovered_urls.json"
        if discovery_file.exists():
            with open(discovery_file, 'r') as f:
                data = json.load(f)
                extractor.discovered_urls = set(data['discovered_urls'])
                extractor.logger.info(f"Loaded {len(extractor.discovered_urls)} previously discovered URLs")
        else:
            extractor.logger.error("No previous discovery file found. Run discovery first.")
            return
    else:
        # Discover URLs
        extractor.logger.info(f"Starting discovery from: {args.start_url}")
        extractor.discover_urls(args.start_url)
    
    # Extract content (unless discover-only)
    if not args.discover_only:
        extractor.extract_all_discovered(max_pages=args.max_pages)
    
    # Save results
    extractor.save_results(args.output_dir)
    
    # Print summary
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"URLs discovered: {len(extractor.discovered_urls)}")
    print(f"Texts extracted: {len(extractor.extracted_texts)}")
    print(f"Failed extractions: {len(extractor.failed_urls)}")
    if extractor.extracted_texts:
        avg_ratio = sum(t['greek_ratio'] for t in extractor.extracted_texts) / len(extractor.extracted_texts)
        print(f"Average Greek ratio: {avg_ratio:.2f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
