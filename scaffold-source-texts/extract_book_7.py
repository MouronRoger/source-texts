#!/usr/bin/env python3
"""Quick Book 7 extraction for immediate use."""

import asyncio
import json
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from extract_laertius import LaertiusExtractor

async def extract_book_7():
    """Extract only Book VII fragments."""
    
    extractor = LaertiusExtractor()
    
    # Load the catalogue
    catalogue_path = Path(__file__).parent / "data" / "catalogues" / "laertius_catalogue.json"
    
    if not catalogue_path.exists():
        print(f"❌ Catalogue not found: {catalogue_path}")
        return
    
    with open(catalogue_path, 'r', encoding='utf-8') as f:
        catalogue_data = json.load(f)
    
    all_fragments = catalogue_data.get("fragments", [])
    
    # Filter for Book VII fragments only
    book_7_fragments = [f for f in all_fragments if f.get("reference", "").startswith("VII,")]
    
    print(f"Found {len(book_7_fragments)} Book VII fragments to extract")
    print("Starting Book VII extraction...")
    print("=" * 50)
    
    # Set up the extractor with Book VII fragments only
    extractor.results = []
    extractor.processed_refs = set()
    
    # Process each Book VII fragment
    for i, fragment in enumerate(book_7_fragments, 1):
        reference = fragment.get("reference", "")
        
        if i % 10 == 0:  # Progress update every 10 fragments
            print(f"Progress: {i}/{len(book_7_fragments)} - {reference}")
        
        try:
            extracted = await extractor.extract_fragment(fragment)
            if extracted:
                extractor.results.append(extracted)
        except Exception as e:
            print(f"Error extracting {reference}: {e}")
            continue
    
    print(f"\nExtraction complete: {len(extractor.results)} texts extracted")
    
    # Save results (this will create the book_7 folder)
    extractor.save_results("laertius")
    
    print(f"✅ Book 7 saved to: data/output/laertius_book_7/")
    print(f"📊 Files created:")
    print(f"  - book_7_texts.json")
    print(f"  - book_7_ancient.txt") 
    print(f"  - book_7_full.txt")
    print(f"  - book_7_greek_only.txt")
    print(f"  - book_7_stats.json")

if __name__ == "__main__":
    asyncio.run(extract_book_7())
