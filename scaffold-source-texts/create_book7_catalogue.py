#!/usr/bin/env python3
"""Create a Book VII only catalogue for quick extraction."""

import json
from pathlib import Path

def create_book7_catalogue():
    """Create a catalogue with only Book VII fragments."""
    
    # Load the full catalogue
    full_catalogue_path = Path("data/catalogues/laertius_catalogue.json")
    
    with open(full_catalogue_path, 'r', encoding='utf-8') as f:
        full_catalogue = json.load(f)
    
    # Filter for Book VII fragments only
    all_fragments = full_catalogue.get("fragments", [])
    book7_fragments = [f for f in all_fragments if f.get("reference", "").startswith("VII,")]
    
    # Create Book VII catalogue
    book7_catalogue = {
        "collection": "laertius",
        "discovery_date": full_catalogue.get("discovery_date"),
        "total_fragments": len(book7_fragments),
        "total_books": 1,
        "books_discovered": ["VII"],
        "discovery_methods": {
            "book_transcription": len(book7_fragments)
        },
        "book_statistics": {
            "VII": {
                "total": len(book7_fragments),
                "regular": len([f for f in book7_fragments if not f.get("is_italian", False)]),
                "italian": len([f for f in book7_fragments if f.get("is_italian", False)])
            }
        },
        "fragments": book7_fragments
    }
    
    # Save Book VII catalogue
    book7_catalogue_path = Path("data/catalogues/laertius_book7_catalogue.json")
    with open(book7_catalogue_path, 'w', encoding='utf-8') as f:
        json.dump(book7_catalogue, f, ensure_ascii=False, indent=2)
    
    print(f"Created Book VII catalogue with {len(book7_fragments)} fragments")
    print(f"Saved to: {book7_catalogue_path}")
    
    return book7_catalogue_path

if __name__ == "__main__":
    create_book7_catalogue()
