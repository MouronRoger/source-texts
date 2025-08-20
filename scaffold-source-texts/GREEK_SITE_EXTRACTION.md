# Greek Site Extraction Guide

## New Site: http://217.71.231.54:8080/indices/indexA.htm

This is a specialized extractor for a different Greek text source (not the Daphnet Presocratics site).

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install requests beautifulsoup4 lxml
   ```

2. **Test with limited extraction**:
   ```bash
   python extract_greek_site.py --max-pages 5
   ```

3. **Full extraction workflow**:
   ```bash
   # Step 1: Discover all URLs (safer to run separately)
   python extract_greek_site.py --discover-only
   
   # Step 2: Extract content from discovered URLs
   python extract_greek_site.py --extract-only
   ```

4. **One-shot extraction** (discover + extract):
   ```bash
   python extract_greek_site.py
   ```

## Output Files

The script creates several output files in `data/output/`:

- **`greek_site_corpus.json`** - Main JSON corpus with metadata
- **`greek_site_texts.txt`** - Clean Greek text only
- **`discovered_urls.json`** - List of discovered URLs
- **`extraction_stats.json`** - Extraction statistics

## Key Features

### Intelligent Greek Text Detection
- Validates minimum 10% Greek Unicode characters
- Tries multiple HTML selectors to find Greek content
- Handles different encoding issues

### Robust Extraction Strategy
- **Discovery Phase**: Crawls site systematically to find all content pages
- **Extraction Phase**: Downloads and parses each page for Greek text
- **Rate Limiting**: 2-second delays between requests (configurable)
- **Error Handling**: Continues on failures, logs all issues

### Flexible Output Formats
- **JSON**: Structured data with metadata, references, paragraphs
- **Text**: Clean Greek text corpus for processing
- **Statistics**: Success rates, Greek ratios, extraction metrics

## Configuration Options

```bash
# Customize crawling
python extract_greek_site.py --delay 3.0 --max-pages 10

# Different starting URL
python extract_greek_site.py --start-url "http://217.71.231.54:8080/some/other/page.htm"

# Custom output directory
python extract_greek_site.py --output-dir "my_extraction_results"
```

## Expected JSON Structure

```json
{
  "metadata": {
    "source_site": "http://217.71.231.54:8080",
    "extraction_date": "2025-08-21 15:30:00",
    "total_texts": 150,
    "total_discovered": 200,
    "failed_urls": 10
  },
  "texts": [
    {
      "url": "http://217.71.231.54:8080/some/page.htm",
      "title": "Page Title",
      "reference": "1.23",
      "greek_text": "Actual Greek text here...",
      "paragraphs": ["First paragraph...", "Second paragraph..."],
      "selector_used": "[lang=\"grc\"]",
      "greek_ratio": 0.85,
      "extraction_date": "2025-08-21 15:30:00"
    }
  ]
}
```

## Monitoring Progress

Watch the extraction in real-time:
```bash
# In another terminal
tail -f data/output/extraction_stats.json
```

## Troubleshooting

### No Greek text found
- Site may use different HTML structure
- Check `discovered_urls.json` to see what URLs were found
- Try accessing a specific page manually first

### Network errors
- Increase delay: `--delay 5.0`
- Server may be rate limiting

### Encoding issues
- Script handles UTF-8 conversion automatically
- Greek text validation helps filter out non-Greek content

## Integration with Your Project

This extractor follows the same patterns as your existing Daphnet extractors:

1. **Two-stage process**: Discovery → Extraction
2. **Greek text validation**: Unicode ratio checking
3. **JSON output format**: Compatible with your analysis tools
4. **Rate limiting**: Respectful crawling
5. **Resume capability**: Can restart from discovery phase

The output JSON format is designed to be compatible with your existing Greek text processing workflows.
