# Daphnet Greek Texts Extraction Project

## Overview
Systematic extraction of ancient Greek philosophical texts from the ILIESI-CNR Daphnet database, including the complete Diels-Kranz collection and related philosophical works.

**Architecture Philosophy**: Each collection has its own specialized discovery and extraction scripts because they follow different patterns, reference systems, and organizational structures.

## Project Structure

```
scaffold-source-texts/
├── scripts/              # Collection-specific extraction scripts
│   ├── discover_fragments.py  # Generic Presocratics discovery
│   ├── discover_socratics.py  # Socratics-specific discovery
│   ├── discover_laertius.py   # Laertius-specific discovery
│   ├── extract_texts.py       # Generic text extractor
│   ├── extract_socratics.py   # Socratics extractor with philosopher mapping
│   └── extract_laertius.py    # Laertius extractor with book-specific output
├── config/
│   └── config.yaml      # Shared configuration settings
├── data/
│   ├── input/          # Collection-specific source CSV files
│   │   ├── presocratic_hyperlinks.csv
│   │   ├── socratics_name_hyperlinks.csv  # Philosopher name mappings
│   │   ├── laertius_books_with_links.csv  # Book-to-number mappings
│   │   └── sextus_hyperlink.csv
│   ├── catalogues/     # Collection-specific discovery catalogues
│   │   ├── presocratics_catalogue.json
│   │   ├── socratics_catalogue.json
│   │   └── laertius_catalogue.json
│   ├── output/         # Collection-specific extracted texts
│   │   ├── presocratics_*.{json,txt}      # Single files
│   │   ├── socratics_*.{json,txt}         # Single files
│   │   ├── laertius_book_*/book_*.*       # Book-specific folders
│   │   └── laertius_overall_stats.json
│   └── checkpoints/    # Resume points for extraction
├── docs/
└── logs/
```

## Collection-Specific Two-Stage Process

Each collection follows a tailored two-stage approach because they have different reference systems and organizational patterns:

### Stage 1: Collection-Specific Discovery

**Presocratics** (`discover_fragments.py`)
- Combines CSV-based and RDF-based discovery methods
- Handles Diels-Kranz numbering (1-A,1 to 90-C,n)
- Discovers sub-fragments (a, b, c suffixes)

**Socratics** (`discover_socratics.py`) 
- Optimized for Roman numerals I-VI with letter sections A-S
- Handles philosopher name mapping via CSV
- Discovers fragments with proper attribution

**Laertius** (`discover_laertius.py`)
- Crawls book transcription pages for Books I-X
- Discovers actual ranges (e.g., Book VII has 202 sections, not guessed 30)
- Handles special references like V,33a, X,120b
- Finds both regular and Italian translation variants

### Stage 2: Collection-Specific Extraction

**Presocratics** (`extract_texts.py`)
- Standard single-file output format
- Philosopher mapping via config file

**Socratics** (`extract_socratics.py`)
- Fixed philosopher attribution (was defaulting to "Socratic School")
- Uses `socratics_name_hyperlinks.csv` for proper philosopher names
- Handles BOM encoding in CSV files

**Laertius** (`extract_laertius.py`)
- **Book-specific output folders** (10 separate directories)
- Each book gets its own complete set of output files
- Overall statistics file for collection summary

## Usage

### Collection-Specific Commands

Each collection uses its own specialized scripts:

#### Presocratics (Diels-Kranz fragments)
```bash
# Discovery: Combined CSV + RDF methods
python scripts/discover_fragments.py --collection presocratics --both

# Extraction: Standard output format  
python scripts/extract_texts.py --collection presocratics
```

#### Socratics (Socratic philosophers with proper attribution)
```bash
# Discovery: Roman numeral optimization
python scripts/discover_socratics.py

# Extraction: Fixed philosopher names (not "Socratic School")
python scripts/extract_socratics.py --no-resume
```

#### Laertius (Diogenes Laertius with book-specific folders)
```bash
# Discovery: Finds actual ranges (e.g., Book VII = 202 sections)
python scripts/discover_laertius.py

# Extraction: Creates 10 book-specific output folders
python scripts/extract_laertius.py --no-resume
```

### Available Collections
- **`presocratics`** - Diels-Kranz fragments (chapters 1-90, ~3,000+ fragments)
- **`socratics`** - Socratic philosophers (2,415+ fragments with proper attribution)
- **`laertius`** - Diogenes Laertius (2,414+ fragments across 10 books)
- **`sextus`** - Sextus Empiricus (discovery script needed)

## Key Features

### Collection-Specific Architecture
- **Tailored discovery**: Each collection has optimized discovery patterns
- **Proper attribution**: Fixed philosopher/book mapping instead of generic labels
- **Accurate ranges**: Discovers actual fragment counts, not guessed estimates
- **Custom output**: Collection-appropriate organization (single files vs. book folders)

### Robust Discovery Methods
- **Presocratics**: Dual CSV+RDF methods for comprehensive coverage
- **Socratics**: Roman numeral optimization with philosopher name mapping
- **Laertius**: Book transcription page crawling for accurate ranges
- **Smart error handling**: Reduces retries on HTTP 500 errors

### Reliable Extraction
- **Multiple URL fallbacks**: plain.html → transcription → html
- **Greek validation**: Ensures actual Greek content (minimum 10% Greek characters)
- **Unicode normalization**: Consistent NFC encoding
- **Checkpoint/resume**: Handles interruptions gracefully
- **Rate limiting**: Respects server with configurable delays

### Flexible Output Formats
- **Standard collections**: JSON, ancient.txt, full.txt, greek_only.txt, stats.json
- **Laertius special**: 10 book-specific folders + overall statistics
- **Proper attribution**: Specific philosopher/book names instead of generic labels

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

### Why Collection-Specific Scripts?

Each collection has unique characteristics that require specialized handling:

1. **Reference Systems**:
   - **Presocratics**: Diels-Kranz numbering (22-B,30)
   - **Socratics**: Roman numerals + letters (II-A,5)
   - **Laertius**: Book + section + Italian variants (VII,1 / VII,1it)

2. **Data Organization**:
   - **Presocratics**: Single philosopher per chapter
   - **Socratics**: Multiple philosophers, needed CSV mapping to fix attribution
   - **Laertius**: 10 books requiring separate output folders

3. **Discovery Patterns**:
   - **Presocratics**: CSV + RDF hybrid approach
   - **Socratics**: Roman numeral optimization
   - **Laertius**: Book transcription page crawling for actual ranges

### URL Priority

All extractors try URLs in this order:
1. `plain.html` - Cleanest Greek text
2. `agora_show_transcription` - Transcription interface  
3. Regular HTML - Fallback option

### Greek Text Validation

- Checks for Unicode ranges: U+0370-U+03FF, U+1F00-U+1FFF
- Requires minimum 10% Greek characters
- Normalizes to NFC for consistency

### Key Improvements Made

- **Fixed Socratic attribution**: Was showing "Socratic School" for all fragments, now shows specific philosophers like "Euclides Megareus"
- **Accurate Laertius ranges**: Discovery found Book VII has 202 sections, not the guessed 30
- **Book-specific Laertius output**: 10 separate folders instead of mixed single files

## License

Academic research use. Please cite the ILIESI-CNR Daphnet database in any publications.

## Contact

For questions about the extraction project, please refer to the project documentation.
For questions about the Daphnet database, contact ILIESI-CNR.
