# Ultimate Crypto Analyzer

A professional-grade cryptocurrency analysis system that combines advanced technical analysis, volume profiling, and AI-powered insights to generate institutional-quality trading reports.

## Core Components

### 🎯 **Ultimate Crypto Analyzer** (`ultimate_crypto_analyzer.py`)
- Advanced technical analysis with 20+ indicators
- Multi-timeframe confluence (4H, 1D, 1W)
- Dynamic support/resistance levels
- Pattern recognition and trend analysis

### 📊 **Volume Profile Analyzer** (`volume_profile_analyzer.py`)
- Professional volume profiling 
- Value Area calculations (70% volume concentration)
- Point of Control (POC) identification
- Volume-based trading signals

### 🤖 **LLM Trading Analyst** (`llm_trading_analyst.py`)
- AI-powered analysis using Claude Sonnet 4
- Professional trading insights
- Risk assessment and recommendations
- Institutional-grade reporting

### 📄 **Report Generation**
- **Complete Prompt Generator**: Creates comprehensive analysis prompts
- **Rich Report Generator**: Produces HTML and Markdown reports
- Full data inclusion pipeline ensuring 100% analysis coverage

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp config.env.template config.env
# Edit config.env with your API keys
```

### 3. Run Analysis
```python
# Basic analysis
python ultimate_crypto_analyzer.py

# Generate AI analysis
python llm_trading_analyst.py

# Create rich reports
python complete_rich_report_generator.py
```

## Workflow

1. **Data Collection**: Fetches OHLCV data from Binance API
2. **Technical Analysis**: Runs comprehensive technical indicators
3. **Volume Profiling**: Analyzes volume distribution and key levels
4. **Prompt Generation**: Creates complete analysis prompts
5. **AI Analysis**: Processes prompts with Claude Sonnet 4
6. **Report Generation**: Produces institutional-grade HTML/Markdown reports

## Configuration

### Required API Keys
- **Anthropic API**: For Claude Sonnet 4 LLM analysis
- **Binance API**: For market data (optional, uses public endpoints)

### Environment Variables
```bash
ANTHROPIC_API_KEY=your_anthropic_key_here
BINANCE_API_KEY=your_binance_key_here  # Optional
BINANCE_SECRET_KEY=your_binance_secret_here  # Optional
```

## Output Structure

```
output/
├── ultimate_analysis/
│   ├── raw_data/          # Raw OHLCV data
│   ├── processed_data/    # Technical analysis results
│   └── llm_prompts/       # Generated analysis prompts
├── llm_responses/         # AI analysis responses
└── rich_reports/          # Final HTML/Markdown reports
```

## Features

- **Comprehensive Technical Analysis**: 20+ indicators across multiple timeframes
- **Professional Volume Profiling**: Value Area, POC, and distribution analysis
- **AI-Powered Insights**: Claude Sonnet 4 integration for professional analysis
- **Rich Report Generation**: Institutional-quality HTML and Markdown reports
- **Complete Data Pipeline**: Ensures 100% analysis data inclusion
- **Multi-Timeframe Confluence**: 4H, 1D, and 1W timeframe analysis

## License

MIT License - see LICENSE file for details.