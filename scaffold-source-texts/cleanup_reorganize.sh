#!/bin/bash

# Archive old scaffold scripts
# Run from project root: ./cleanup.sh

echo "Archiving legacy scaffold scripts..."

# Create archive directory
mkdir -p archive/scaffold_legacy

# Move legacy scripts
echo "Moving legacy extraction scripts..."
mv scaffold/extract_enhanced.py archive/scaffold_legacy/ 2>/dev/null
mv scaffold/scaffold.py archive/scaffold_legacy/ 2>/dev/null
mv scaffold/scan_presocratic_books.py archive/scaffold_legacy/ 2>/dev/null
mv scaffold/extract_from_catalogue.py archive/scaffold_legacy/ 2>/dev/null
mv scaffold/extract_book_transcriptions.py archive/scaffold_legacy/ 2>/dev/null

# Move YAML specifications to archive
echo "Moving YAML specifications..."
mv scaffold/01_scaffold_specification.yaml archive/scaffold_legacy/ 2>/dev/null
mv scaffold/02_scaffold_roadmap.yaml archive/scaffold_legacy/ 2>/dev/null
mv scaffold/03_scaffold_runner.yaml archive/scaffold_legacy/ 2>/dev/null

# Keep these in scaffold for now (may still be useful)
echo "Keeping utility scripts..."
# scaffold/extract.py - Keep as reference for RDF logic
# scaffold/scan_catalogue.py - Keep as reference for CSV parsing
# scaffold/config_loader.py - Still used
# scaffold/http_utils.py - Still used

# Move misplaced files
echo "Moving misplaced files..."
mv extracted_daphnet_links.csv data/ 2>/dev/null

# Create logs directory
mkdir -p logs
mv extraction.log logs/ 2>/dev/null

# Clean up .DS_Store files
find . -name ".DS_Store" -delete

echo "Cleanup complete!"
echo ""
echo "New structure:"
echo "  scripts/         - Main extraction scripts"
echo "  config/          - Configuration files"
echo "  data/input/      - Source CSV files"
echo "  data/catalogues/ - Discovery catalogues"
echo "  data/output/     - Extracted texts"
echo "  archive/         - Legacy scripts (for reference)"
echo ""
echo "To start extraction:"
echo "  python scripts/discover_fragments.py --collection presocratics --both"
echo "  python scripts/extract_texts.py --collection presocratics"
