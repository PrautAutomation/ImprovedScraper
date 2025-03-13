# Advanced Asynchronous Web Scraper

A high-performance, modular web scraper with fault tolerance, resource management, and comprehensive content analysis capabilities.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Overview

This enterprise-grade web scraper is designed for large-scale data collection with a focus on reliability, performance, and content analysis. It employs asynchronous programming patterns to efficiently crawl websites while respecting server limitations and providing detailed analytics on the collected data.

## Key Features

- **High Performance**: Asynchronous architecture for concurrent scraping
- **Fault Tolerance**: Circuit breaker pattern, automatic retries, and error handling
- **Resource Management**: Memory monitoring and adaptive resource allocation
- **Content Analysis**: Real-time content processing and reporting
- **Configurable**: Extensive JSON-based configuration options
- **Polite Crawling**: Rate limiting, robots.txt compliance, and sitemap integration
- **JavaScript Support**: Optional rendering of JavaScript-heavy pages
- **Proxy Support**: Rotation of proxy servers for distributed scraping
- **Storage Management**: Efficient content storage with deduplication
- **Comprehensive Reporting**: Detailed logs and Word document reports

## Architecture

The scraper is built with a modular architecture that separates concerns and allows for easy extension:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ AsyncWebScraper │────▶│   PageFetcher   │────▶│Content Processor│
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   URLManager    │     │   RateLimiter   │     │ ContentAnalyzer │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CircuitBreaker │     │  MemoryMonitor  │     │   FileStorage   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Installation

### Requirements

- Python 3.8 or higher
- Required packages:
  - aiohttp
  - beautifulsoup4
  - psutil
  - python-docx

### Optional Dependencies

- Playwright (for JavaScript rendering)
- Transformers (for AI-based content processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-company/advanced-web-scraper.git
   cd advanced-web-scraper
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv scraper_env
   source scraper_env/bin/activate  # On Windows: scraper_env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install optional dependencies:
   ```bash
   # For JavaScript rendering
   pip install playwright
   playwright install chromium
   
   # For AI-based content processing
   pip install transformers torch
   ```

## Usage

### Basic Usage

```bash
python improvedScraper.py --url https://example.com --output ./scraped_data
```

### Using a Configuration File

```bash
python improvedScraper.py --config config.json
```

### Example Configuration

```json
{
  "root_url": "https://example.com",
  "output_dir": "scraped_data",
  "max_retries": 5,
  "delay": 2.0,
  "max_workers": 10,
  "timeout": 30,
  "max_depth": 3,
  "user_agent": "Your Company Web Scraper 1.0",
  "excluded_patterns": [".pdf", ".zip", "#", "mailto:", "javascript:"],
  "memory_limit_mb": 2000,
  "enable_javascript": true,
  "max_file_size_mb": 50,
  "content_type_filters": ["text/html", "application/xhtml+xml"],
  "extract_metadata": true,
  "follow_redirects": true,
  "verify_ssl": true
}
```

## Output

The scraper generates a structured output directory:

```
output/
├── config.json                # Saved configuration
├── content_analysis.docx      # Detailed content analysis report
├── scraper.log                # Comprehensive log file
├── content/                   # Scraped content organized by domain
│   └── example.com/
│       └── 20250313_[hash].json
├── metadata/                  # Metadata about scraped URLs
│   └── metadata.json
└── logs/                      # Additional log files
```

## Advanced Features

### JavaScript Rendering

Enable JavaScript rendering to scrape single-page applications and JavaScript-heavy websites:

```json
{
  "enable_javascript": true
}
```

### Proxy Rotation

Configure multiple proxies for distributed scraping:

```json
{
  "proxies": [
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080"
  ]
}
```

### Content Analysis

The scraper generates a detailed Word document report with:

- Content statistics
- URL structure analysis
- Content categorization
- Sample content from each category

### AI-Based Content Processing

When the transformers library is available, the scraper can use AI models to process and enhance the scraped content.

## Performance Considerations

- **Memory Usage**: Monitor the `memory_limit_mb` setting to prevent excessive memory consumption
- **Concurrency**: Adjust `max_workers` based on your system capabilities and target server limitations
- **Rate Limiting**: Set appropriate `delay` values to avoid overloading target servers

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [aiohttp](https://docs.aiohttp.org/)
- HTML parsing by [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)
- Document generation with [python-docx](https://python-docx.readthedocs.io/)
- Optional AI processing with [Hugging Face Transformers](https://huggingface.co/transformers/)
