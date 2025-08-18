# Complete Daphnet Presocratics Extraction Plan

## Executive Summary
This document provides a comprehensive technical plan to extract all textual content from the Daphnet Presocratics database hosted by ILIESI-CNR, including the complete Diels-Kranz fragments, testimonies, and related philosophical collections.

## Site Architecture Analysis

### Primary Target Site
- **Base URL**: `http://ancientsource.daphnet.iliesi.cnr.it`
- **Documentation Site**: `http://presocratics-documentation.ancientsource.daphnet.iliesi.cnr.it`
- **Technology Stack**: Custom "Agora" CMS with systematic ID-based content management
- **Encoding**: UTF-8 Unicode (proper Greek text support)
- **Content Types**: HTML (human-readable) + RDF/XML (machine-readable)

### URL Pattern Structure
```
Base Pattern: http://ancientsource.daphnet.iliesi.cnr.it/texts/[COLLECTION]/[REFERENCE]

Collections Identified:
- Presocratics/     (DK fragments - primary target)
- Socratics/        (Socratic testimonies)
- Laertius/         (Diogenes Laertius Lives)
- Sextus/           (Sextus Empiricus works)
```

### Reference Number Patterns
**Presocratics Collection:**
- Format: `[CHAPTER]-[TYPE],[NUMBER]`
- Chapters: 1-90 (corresponding to DK numbering)
- Types: A (testimonia), B (fragments), C (imitations)
- Numbers: Sequential within each type
- Examples: `68-B,117`, `21-A,47`, `22-B,30`

**Alternative Access Methods:**
- Direct ID: `/agora_show_transcription?id=[NUMERIC_ID]`
- RDF format: `[BASE_URL].rdf`

## Technical Implementation Plan

### Phase 1: Site Mapping and Discovery
**Objective**: Identify all available content and URL patterns

**1.1 Collection Discovery**
```python
# Primary collections to map
collections = [
    'Presocratics',
    'Socratics', 
    'Laertius',
    'Sextus'
]

# URL discovery patterns
base_url = "http://ancientsource.daphnet.iliesi.cnr.it"
text_pattern = f"{base_url}/texts/{{collection}}/{{reference}}"
rdf_pattern = f"{base_url}/texts/{{collection}}/{{reference}}.rdf"
```

**1.2 DK Chapter Mapping**
```python
# For Presocratics: systematically test all DK combinations
dk_chapters = range(1, 91)  # DK chapters 1-90
dk_types = ['A', 'B', 'C']   # Testimonia, Fragments, Imitations
max_numbers = 200            # Test up to 200 items per type (adjust based on findings)

# Generate comprehensive URL list
urls_to_test = []
for chapter in dk_chapters:
    for dk_type in dk_types:
        for number in range(1, max_numbers + 1):
            reference = f"{chapter}-{dk_type},{number}"
            urls_to_test.append(f"{base_url}/texts/Presocratics/{reference}")
```

### Phase 2: Content Extraction Strategy
**Objective**: Develop robust extraction methods for multiple content types

**2.1 Multi-Format Extraction**
```python
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import time
import json

class DaphnetExtractor:
    def __init__(self):
        self.base_url = "http://ancientsource.daphnet.iliesi.cnr.it"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        
    def extract_html_content(self, url):
        """Extract structured content from HTML version"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract Greek text (main content)
                greek_text = self.extract_greek_text(soup)
                
                # Extract metadata
                metadata = self.extract_metadata(soup)
                
                return {
                    'url': url,
                    'greek_text': greek_text,
                    'metadata': metadata,
                    'html_content': str(soup),
                    'status': 'success'
                }
            else:
                return {'url': url, 'status': 'not_found', 'code': response.status_code}
        except Exception as e:
            return {'url': url, 'status': 'error', 'error': str(e)}
    
    def extract_rdf_content(self, url):
        """Extract structured data from RDF version"""
        rdf_url = f"{url}.rdf"
        try:
            response = self.session.get(rdf_url, timeout=30)
            if response.status_code == 200:
                # Parse RDF/XML
                root = ET.fromstring(response.content)
                # Extract structured semantic data
                return self.parse_rdf_content(root, rdf_url)
        except Exception as e:
            return {'url': rdf_url, 'status': 'error', 'error': str(e)}
```

**2.2 Greek Text Processing**
```python
def extract_greek_text(self, soup):
    """Specialized Greek text extraction"""
    # Look for common containers of Greek text
    selectors = [
        'div[lang="grc"]',      # Language-specific divs
        '.greek-text',          # Class-based selectors
        'p:contains("ἀ")',      # Paragraphs containing Greek characters
        '*:contains("φ")',      # Any element with Greek characters
    ]
    
    greek_content = []
    for selector in selectors:
        elements = soup.select(selector)
        for elem in elements:
            text = elem.get_text(strip=True)
            if self.is_greek_text(text):
                greek_content.append({
                    'text': text,
                    'element': elem.name,
                    'classes': elem.get('class', [])
                })
    
    return greek_content

def is_greek_text(self, text):
    """Detect if text contains substantial Greek content"""
    greek_chars = sum(1 for char in text if '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF')
    return greek_chars > len(text) * 0.3  # 30% Greek characters threshold
```

### Phase 3: Systematic Crawling Implementation
**Objective**: Execute comprehensive data extraction with proper etiquette

**3.1 Ethical Crawling Setup**
```python
class EthicalCrawler:
    def __init__(self):
        self.delay = 2.0  # 2-second delay between requests
        self.max_retries = 3
        self.timeout = 30
        self.concurrent_limit = 1  # Single-threaded to be respectful
        
    def crawl_with_delays(self, urls):
        """Crawl URLs with proper delays and error handling"""
        results = []
        
        for i, url in enumerate(urls):
            print(f"Processing {i+1}/{len(urls)}: {url}")
            
            # Respect robots.txt and implement delays
            time.sleep(self.delay)
            
            # Extract content
            result = self.extract_content_comprehensive(url)
            results.append(result)
            
            # Save progress every 100 items
            if (i + 1) % 100 == 0:
                self.save_progress(results, f"progress_{i+1}.json")
                
        return results
```

**3.2 Data Storage Strategy**
```python
def save_extracted_data(self, data, output_format='multiple'):
    """Save data in multiple formats for different use cases"""
    
    if output_format in ['multiple', 'json']:
        # JSON for structured data
        with open('daphnet_complete.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    if output_format in ['multiple', 'csv']:
        # CSV for tabular analysis
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv('daphnet_fragments.csv', index=False, encoding='utf-8')
    
    if output_format in ['multiple', 'xml']:
        # TEI-XML for scholarly use
        self.create_tei_xml(data, 'daphnet_tei.xml')
    
    if output_format in ['multiple', 'txt']:
        # Plain text Greek corpus
        self.create_text_corpus(data, 'daphnet_greek_corpus.txt')
```

### Phase 4: Advanced Extraction Techniques
**Objective**: Handle dynamic content and complex structures

**4.1 JavaScript-Rendered Content**
```python
# If the site requires JavaScript rendering
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def setup_selenium_crawler():
    """Setup headless browser for dynamic content"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(10)
    return driver

def extract_dynamic_content(self, url):
    """Extract content from JavaScript-rendered pages"""
    driver = self.setup_selenium_crawler()
    try:
        driver.get(url)
        
        # Wait for content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Extract rendered HTML
        html = driver.page_source
        return BeautifulSoup(html, 'html.parser')
    finally:
        driver.quit()
```

**4.2 Menu and Navigation Discovery**
```python
def discover_site_structure(self):
    """Discover additional content through menu navigation"""
    menu_url = f"{self.base_url}/agora_show_menu"
    
    # Extract menu structure
    menu_content = self.extract_html_content(menu_url)
    
    # Parse navigation links
    navigation_urls = self.parse_navigation_links(menu_content)
    
    return navigation_urls
```

### Phase 5: Data Validation and Quality Assurance
**Objective**: Ensure completeness and accuracy of extracted data

**5.1 Content Validation**
```python
def validate_extraction(self, extracted_data):
    """Validate extracted content for completeness"""
    validation_report = {
        'total_items': len(extracted_data),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'greek_text_items': 0,
        'empty_content': 0,
        'dk_coverage': {}
    }
    
    for item in extracted_data:
        if item['status'] == 'success':
            validation_report['successful_extractions'] += 1
            if item.get('greek_text'):
                validation_report['greek_text_items'] += 1
        else:
            validation_report['failed_extractions'] += 1
    
    return validation_report
```

**5.2 DK Completeness Check**
```python
def check_dk_completeness(self, extracted_data):
    """Verify we have complete DK coverage"""
    dk_found = set()
    
    for item in extracted_data:
        if 'Presocratics' in item['url']:
            # Extract DK reference from URL
            dk_ref = self.extract_dk_reference(item['url'])
            if dk_ref:
                dk_found.add(dk_ref)
    
    # Compare against expected DK numbering
    expected_dk = self.generate_expected_dk_list()
    missing_dk = expected_dk - dk_found
    
    return {
        'found': sorted(list(dk_found)),
        'missing': sorted(list(missing_dk)),
        'coverage_percentage': len(dk_found) / len(expected_dk) * 100
    }
```

## Implementation Timeline

### Week 1: Setup and Discovery
- [ ] Set up development environment
- [ ] Implement basic URL discovery
- [ ] Test extraction methods on sample URLs
- [ ] Analyze site structure and response patterns

### Week 2: Core Extraction
- [ ] Implement comprehensive crawling system
- [ ] Extract Presocratics collection (primary target)
- [ ] Validate Greek text extraction accuracy
- [ ] Implement data storage systems

### Week 3: Extended Collections
- [ ] Extract Socratics collection
- [ ] Extract Laertius collection
- [ ] Extract Sextus collection
- [ ] Cross-validate content completeness

### Week 4: Quality Assurance and Output
- [ ] Data validation and completeness checks
- [ ] Generate multiple output formats
- [ ] Create documentation and metadata
- [ ] Final quality assurance review

## Output Deliverables

### Primary Outputs
1. **Complete JSON Dataset** - Structured data with all metadata
2. **Greek Text Corpus** - Clean UTF-8 text file of all fragments
3. **TEI-XML Edition** - Scholarly markup following TEI standards
4. **CSV Database** - Tabular format for analysis
5. **RDF/Turtle** - Semantic web format with linked data

### Metadata Schema
```json
{
  "dk_reference": "22-B,30",
  "chapter": 22,
  "type": "B",
  "number": 30,
  "philosopher": "Heraclitus",
  "greek_text": "κόσμον τόνδε...",
  "source_url": "http://ancientsource.daphnet.iliesi.cnr.it/texts/Presocratics/22-B,30",
  "extraction_date": "2025-08-18",
  "content_length": 156,
  "has_apparatus": true,
  "translation_available": true
}
```

## Risk Mitigation

### Technical Risks
- **Server blocks**: Implement respectful crawling with delays
- **Content changes**: Version control and regular validation
- **Encoding issues**: Robust Unicode handling
- **Incomplete extractions**: Multiple validation layers

### Legal and Ethical Considerations
- **Respect robots.txt**: Check and follow site policies
- **Academic use**: Ensure extraction is for scholarly purposes
- **Attribution**: Properly cite ILIESI-CNR as source
- **Rate limiting**: Implement conservative request patterns

## Success Metrics
- **Completeness**: >95% of available DK fragments extracted
- **Accuracy**: >99% correct Greek text preservation
- **Usability**: Multiple output formats for different use cases
- **Documentation**: Complete metadata for all fragments
- **Reproducibility**: Extraction process can be repeated/updated

## Conclusion
This plan provides a comprehensive approach to extracting the complete Daphnet Presocratics database while respecting the host institution and maintaining academic standards. The multi-format output ensures the extracted data will be useful for computational analysis, traditional scholarship, and digital humanities research.