# Pokročilý Asynchronní Web Scraper

Vysoce výkonný, modulární web scraper s odolností proti chybám, správou zdrojů a komplexními možnostmi analýzy obsahu.

![Licence](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Přehled

Tento podnikový web scraper je navržen pro rozsáhlý sběr dat se zaměřením na spolehlivost, výkon a analýzu obsahu. Využívá asynchronní programovací vzory pro efektivní procházení webových stránek s respektováním omezení serverů a poskytuje detailní analytiku shromážděných dat.

## Klíčové funkce

- **Vysoký výkon**: Asynchronní architektura pro souběžné scrapování
- **Odolnost proti chybám**: Vzor circuit breaker, automatické opakování a zpracování chyb
- **Správa zdrojů**: Monitorování paměti a adaptivní alokace zdrojů
- **Analýza obsahu**: Zpracování a reportování obsahu v reálném čase
- **Konfigurovatelnost**: Rozsáhlé možnosti konfigurace založené na JSON
- **Ohleduplné procházení**: Omezení rychlosti, dodržování robots.txt a integrace sitemap
- **Podpora JavaScriptu**: Volitelné vykreslování stránek s vysokým obsahem JavaScriptu
- **Podpora proxy**: Rotace proxy serverů pro distribuované scrapování
- **Správa úložiště**: Efektivní ukládání obsahu s deduplikací
- **Komplexní reportování**: Detailní logy a reporty ve formátu Word

## Architektura

Scraper je postaven na modulární architektuře, která odděluje jednotlivé funkce a umožňuje snadné rozšíření:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  AsyncWebScraper │────▶│   PageFetcher   │────▶│  ContentProcessor│
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

## Instalace

### Požadavky

- Python 3.8 nebo vyšší
- Požadované balíčky:
  - aiohttp
  - beautifulsoup4
  - psutil
  - python-docx

### Volitelné závislosti

- Playwright (pro vykreslování JavaScriptu)
- Transformers (pro zpracování obsahu pomocí AI)

### Nastavení

1. Klonujte repozitář:
   ```bash
   git clone https://github.com/your-company/advanced-web-scraper.git
   cd advanced-web-scraper
   ```

2. Vytvořte a aktivujte virtuální prostředí:
   ```bash
   python -m venv scraper_env
   source scraper_env/bin/activate  # Na Windows: scraper_env\Scripts\activate
   ```

3. Nainstalujte závislosti:
   ```bash
   pip install -r requirements.txt
   ```

4. Nainstalujte volitelné závislosti:
   ```bash
   # Pro vykreslování JavaScriptu
   pip install playwright
   playwright install chromium
   
   # Pro zpracování obsahu pomocí AI
   pip install transformers torch
   ```

## Použití

### Základní použití

```bash
python improvedScraper.py --url https://example.com --output ./scraped_data
```

### Použití konfiguračního souboru

```bash
python improvedScraper.py --config config.json
```

### Příklad konfigurace

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

## Výstup

Scraper generuje strukturovaný výstupní adresář:

```
output/
├── config.json                # Uložená konfigurace
├── content_analysis.docx      # Detailní report analýzy obsahu
├── scraper.log                # Komplexní log soubor
├── content/                   # Scrapovaný obsah organizovaný podle domén
│   └── example.com/
│       └── 20250313_[hash].json
├── metadata/                  # Metadata o scrapovaných URL
│   └── metadata.json
└── logs/                      # Další log soubory
```

## Pokročilé funkce

### Vykreslování JavaScriptu

Povolte vykreslování JavaScriptu pro scrapování single-page aplikací a webů s vysokým obsahem JavaScriptu:

```json
{
  "enable_javascript": true
}
```

### Rotace proxy

Nakonfigurujte více proxy pro distribuované scrapování:

```json
{
  "proxies": [
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080"
  ]
}
```

### Analýza obsahu

Scraper generuje detailní report ve formátu Word s:

- Statistikami obsahu
- Analýzou struktury URL
- Kategorizací obsahu
- Ukázkami obsahu z každé kategorie

### Zpracování obsahu pomocí AI

Když je dostupná knihovna transformers, scraper může využívat AI modely pro zpracování a vylepšení scrapovaného obsahu.

## Výkonnostní doporučení

- **Využití paměti**: Sledujte nastavení `memory_limit_mb` pro prevenci nadměrné spotřeby paměti
- **Souběžnost**: Upravte `max_workers` podle možností vašeho systému a omezení cílového serveru
- **Omezení rychlosti**: Nastavte vhodné hodnoty `delay` pro prevenci přetížení cílových serverů

## Přispívání

1. Forkněte repozitář
2. Vytvořte větev pro vaši funkci (`git checkout -b feature/amazing-feature`)
3. Commitněte vaše změny (`git commit -m 'Přidána úžasná funkce'`)
4. Pushněte do větve (`git push origin feature/amazing-feature`)
5. Otevřete Pull Request

## Licence

Tento projekt je licencován pod licencí MIT - podrobnosti viz soubor LICENSE.

## Poděkování

- Vytvořeno pomocí [aiohttp](https://docs.aiohttp.org/)
- Parsování HTML pomocí [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)
- Generování dokumentů pomocí [python-docx](https://python-docx.readthedocs.io/)
- Volitelné zpracování AI pomocí [Hugging Face Transformers](https://huggingface.co/transformers/)
