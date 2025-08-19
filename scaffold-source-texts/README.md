# Daphnet Greek Texts Extraction Project

## Overview
Systematic extraction of ancient Greek philosophical texts from the ILIESI-CNR Daphnet database, including the complete Diels-Kranz collection and related philosophical works.

## Project Structure

```
scaffold-source-texts/
├── scripts/              # Main extraction scripts
│   ├── discover_fragments.py  # Fragment discovery (CSV + RDF methods)
│   └── extract_texts.py       # Text extraction using catalogue
├── config/
│   └── config.yaml      # Configuration settings
├── data/
│   ├── input/          # Source CSV files
│   │   ├── presocratic_hyperlinks.csv
│   │   ├── socratics_hyperlinks.csv
│   │   ├── laertius_hyperlinks.csv
│   │   └── sextus_hyperlink.csv
│   ├── catalogues/     # Discovered fragment catalogues
│   │   └── [collection]_catalogue.json
│   ├── output/         # Extracted Greek texts
│   │   ├── [collection]_texts.json
│   │   ├── [collection]_texts.txt
│   │   └── [collection]_stats.json
│   └── checkpoints/    # Resume points for extraction
└── logs/
    └── extraction.log

## Legacy Scripts (to be archived)
scaffold/
├── extract.py           # Original RDF-based extractor
├── extract_enhanced.py  # Enhanced version (redundant)
├── scaffold.py          # Original scaffold module
├── scan_catalogue.py    # Generic catalogue scanner
├── scan_presocratic_books.py  # Specific scanner (redundant)
└── extract_from_catalogue.py  # Catalogue-based extractor
```

## Two-Stage Extraction Process

### Stage 1: Discovery (`discover_fragments.py`)
Combines two complementary discovery methods:

1. **CSV-based Discovery**
   - Uses hyperlink CSV files from Muruca interface
   - Fetches book transcription pages
   - Extracts fragment IDs and URLs
   - Fast and comprehensive for known chapters

2. **RDF-based Discovery**
   - Systematic probing of RDF URLs
   - Discovers fragments not in CSV
   - Extracts plain.html URLs from RDF metadata
   - Handles sub-fragments (a, b, c suffixes)

### Stage 2: Extraction (`extract_texts.py`)
- Uses discovered catalogue from Stage 1
- Extracts actual Greek text from multiple URL types
- Validates Greek content (minimum 10% Greek characters)
- Saves in multiple formats (JSON, plain text)
- Supports checkpointing for resume capability

## Usage

### Quick Start
```bash
# Install dependencies
pip install aiohttp beautifulsoup4 lxml pyyaml

# Stage 1: Discover fragments (CSV method)
python scripts/discover_fragments.py --collection presocratics --csv

# Stage 1: Discover fragments (both methods)
python scripts/discover_fragments.py --collection presocratics --both

# Stage 2: Extract texts
python scripts/extract_texts.py --collection presocratics
```

### Available Collections
- `presocratics` - Diels-Kranz fragments (chapters 1-90)
- `socratics` - Socratic philosophers
- `laertius` - Diogenes Laertius
- `sextus` - Sextus Empiricus

### Command Options

#### Discovery
```bash
python scripts/discover_fragments.py [options]
  --collection {presocratics,socratics,laertius,sextus}
  --csv        Use CSV-based discovery
  --rdf        Use RDF-based discovery  
  --both       Use both methods (recommended)
```

#### Extraction
```bash
python scripts/extract_texts.py [options]
  --collection {presocratics,socratics,laertius,sextus}
  --no-resume  Start fresh, ignore checkpoint
```

## Key Features

### Robust Discovery
- **Dual methods**: CSV for speed, RDF for completeness
- **Smart probing**: Reduces retries on HTTP 500 errors
- **Sub-fragment detection**: Finds a, b, c variants
- **Catalogue merging**: Combines discoveries from both methods

### Reliable Extraction
- **Multiple URL fallbacks**: plain.html → transcription → html
- **Greek validation**: Ensures actual Greek content
- **Unicode normalization**: Consistent NFC encoding
- **Checkpoint/resume**: Handles interruptions gracefully
- **Rate limiting**: Respects server with configurable delays

### Output Formats
- **JSON**: Structured data with metadata
- **Plain text**: Simple text format for reading
- **Statistics**: Extraction summary and coverage

## Configuration

Edit `config/config.yaml`:
```yaml
base_url: http://ancientsource.daphnet.iliesi.cnr.it
delay: 2.0           # Seconds between requests
timeout: 30          # Request timeout
max_retries: 3       # Retry attempts for failures
min_greek_ratio: 0.1 # Minimum Greek character ratio
user_agent: DaphnetExtractor/2.0 (Academic Research)

philosopher_map:
  '11': Thales
  '12': Anaximander
  '13': Anaximenes
  '22': Heraclitus
  '28': Parmenides
  # ... etc
```

## Monitoring Progress

```bash
# Watch discovery progress
tail -f data/catalogues/presocratics_catalogue.json | jq '.metadata'

# Watch extraction progress  
tail -f data/output/presocratics_stats.json

# Check latest checkpoint
ls -la data/checkpoints/
```

## Recovery from Interruption

The extraction automatically saves checkpoints. To resume:
```bash
# Resumes from latest checkpoint by default
python scripts/extract_texts.py --collection presocratics

# Force fresh start
python scripts/extract_texts.py --collection presocratics --no-resume
```

## Technical Notes

### Why Two Discovery Methods?

1. **CSV method** provides:
   - Fast discovery via book transcription pages
   - Complete fragment IDs
   - Direct transcription URLs

2. **RDF method** provides:
   - Systematic coverage
   - Plain HTML URLs
   - Sub-fragment discovery
   - Fallback for missing CSV entries

### URL Priority

The extractor tries URLs in this order:
1. `plain.html` - Cleanest Greek text
2. `agora_show_transcription` - Transcription interface
3. Regular HTML - Fallback option

### Greek Text Validation

- Checks for Unicode ranges: U+0370-U+03FF, U+1F00-U+1FFF
- Requires minimum 10% Greek characters
- Normalizes to NFC for consistency

## License

Academic research use. Please cite the ILIESI-CNR Daphnet database in any publications.

## Contact

For questions about the extraction project, please refer to the project documentation.
For questions about the Daphnet database, contact ILIESI-CNR.
