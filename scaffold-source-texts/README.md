# Daphnet Presocratics Extraction

## Enhanced Features

### Robust Crawling
- **Smart stopping**: Only stops after finding content, then hitting 25+ consecutive 404s
- **Letter suffixes**: Checks sub-fragments (22-B,1a, 22-B,1b, etc.)
- **Rate limiting**: Respects `Retry-After` headers, exponential backoff
- **Interrupt handling**: Saves checkpoint on Ctrl+C

### Data Storage
All data is stored within the repository:
```
data/daphnet/
├── daphnet_corpus.json      # Main corpus
├── daphnet_texts.txt        # Plain text
├── coverage.json            # Statistics
└── checkpoints/             # Progress saves
    └── checkpoint_*.json    # Periodic saves
```

### Command Line Options
```bash
# Full extraction
python scaffold/extract.py

# Test single chapter
python scaffold/extract.py --test

# Custom range
python scaffold/extract.py --start-ch 20 --end-ch 30

# Slower crawling
python scaffold/extract.py --delay 3.0

# Multiple collections
python scaffold/extract.py --collections presocratics laertius

# Resume from checkpoint
# (Automatically loads latest checkpoint if interrupted)
```

### Catalogue-driven Workflow

1. **Phase 0 – Build catalogues**
   Run one of the scanners once per collection to create a definitive fragment list:
   ```bash
   # Presocratics
   python scaffold/scan_catalogue.py --collection presocratics

   # Socratics, Laertius, Sextus
   python scaffold/scan_catalogue.py --collection socratics
   python scaffold/scan_catalogue.py --collection laertius
   python scaffold/scan_catalogue.py --collection sextus
   ```
   Each command writes a `<collection>_catalogue.json` file to `data/daphnet/`.

2. **Phase 1 – Extract texts**
   The extractor now auto-detects catalogue files (or use `--from-catalogue`) and iterates them instead of brute-force probing:
   ```bash
   # Fast extraction using catalogues (0.5 s delay by default)
   python scaffold/extract.py --collections presocratics socratics laertius sextus --verbose
   ```

   If a catalogue is missing for a requested collection, the extractor will skip it unless a legacy brute-force scanner exists (currently only Presocratics).

3. **Tuning**
   Base request delay has been lowered to **0.5 s** in `config.yaml` because the catalogue approach drastically reduces server load.

### RDF Support
The extractor can fetch RDF metadata alongside HTML content:
- Validates RDF XML structure
- Captures HTTP metadata (ETag, Last-Modified)
- Falls back to HTML if RDF unavailable

### Better Identification
- User-Agent includes contact email
- `From` header for server logs
- Respects all redirect codes
- Validates content types

## Installation
```bash
pip install aiohttp beautifulsoup4 lxml pyyaml
```

## Output Files

### JSON Corpus (`daphnet_corpus.json`)
```json
{
  "url": "http://...",
  "rdf_url": "http://.../.rdf",
  "collection": "presocratics",
  "reference": "22-B,30",
  "dk_reference": "22-B,30",
  "philosopher": "Heraclitus",
  "greek_text": "...",
  "paragraphs": ["..."],
  "etag": "...",
  "last_modified": "..."
}
```

### Coverage Report (`coverage.json`)
```json
{
  "extraction_date": "2025-01-20T...",
  "total_records": 2500,
  "successful_records": 2450,
  "by_collection": {
    "presocratics": 2450
  }
}
```

## Monitoring
Watch extraction progress:
```bash
tail -f data/daphnet/checkpoints/checkpoint_*.json | jq '.total_records'
```

## Recovery
If interrupted, the extractor saves state. To resume:
1. Latest checkpoint is in `data/daphnet/checkpoints/`
2. Final output always saved on interrupt
3. Can merge checkpoints if needed
