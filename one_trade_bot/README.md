# 🚀 Crypto Trading Bot - Multi-Pair Intelligence System
# 🎯 Philosophy: One Good Trade Per Day - Enhanced with Multi-Pair Intelligence

A professional-grade cryptocurrency trading system that maintains the disciplined approach of **"One Good Trade Per Day"** while leveraging advanced multi-pair scanning and dynamic target upgrading to maximize opportunity detection.

## 🌟 Enhanced Features (NEW!)

- **🔍 Multi-Pair Scanning**: Automatically discovers and analyzes 40+ liquid Kraken pairs
- **📊 Transparency Dashboard**: Complete audit trail of every trading decision  
- **🔄 Dynamic Target Upgrading**: Switches to better opportunities during the day
- **💻 Live Market Data**: Real-time Kraken price feeds (~$110k BTC vs simulated data)
- **📈 Professional Tracking**: SQLite database with comprehensive trade journaling

## Strategy Overview

Enhanced 5-filter system with multi-pair intelligence:

1. **Market Regime Filter** - Eliminates choppy markets across all pairs
2. **Setup Scanner** - Finds pullback patterns in 40+ liquid pairs  
3. **Confluence Checker** - Picks THE ONE best trade from entire universe
4. **Risk Check** - Final safety validation with real market data
5. **Dynamic Upgrading** - Switches to better targets if found during monitoring

## 📈 Enhanced Performance Targets
- **Opportunity Detection**: 40+ pairs vs 1 pair (4000%+ improvement)
- **Scan Speed**: 23 seconds for complete multi-pair analysis
- **Decision Quality**: Always picks best opportunity across entire universe
- **Transparency**: Complete audit trail of every decision
- **Win Rate**: 60-65% (maintained with better target selection)
- **Risk:Reward**: 2:1 minimum (enhanced with live market data)
- **Risk per Trade**: 1% of account maximum (disciplined approach maintained)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to set:
- Watchlist symbols
- Exchange settings  
- Risk parameters
- Filter thresholds

## 🚀 Quick Start

### Enhanced System (Recommended)
```bash
# Run complete enhanced system with multi-pair scanning
python main_simple.py

# Test all enhanced features
python test_complete_system.py

# Check transparency dashboard
python -c "from core.transparency_dashboard import print_transparency_report; print_transparency_report()"
```

### Configuration
Edit `config.yaml` for enhanced features:
```yaml
paper_trading:
  use_live_market_data: true  # Enable live Kraken data

execution:
  enable_dynamic_upgrading: true  # Allow target switching
  rescan_interval_hours: 1        # Hourly rescans
  upgrade_threshold_points: 10    # Switch threshold
```

## Key Principles

1. **ONE trade maximum per day**
2. **NO trading if no setup meets ALL filters**
3. **1% risk per trade - NO exceptions**
4. **Limit orders only - NO market orders**
5. **Record every decision for analysis**

## Directory Structure

```
one_trade_bot/
├── main.py              # Daily scanner entry point
├── config.yaml          # All configuration settings
├── requirements.txt     # Dependencies
├── filters/             # The 5 filter system
│   ├── regime_filter.py
│   ├── setup_scanner.py  
│   ├── confluence_checker.py
│   ├── risk_check.py
│   └── execution_engine.py
├── core/                # Core functionality
│   ├── data_provider.py
│   ├── position_manager.py
│   └── indicators.py
├── utils/               # Utilities
│   ├── logging.py
│   └── database.py
├── backtest/            # Historical testing
│   └── backtester.py
└── tests/               # Unit tests
    └── test_filters.py
```

## Safety Features

- Position size limits (max 10% of account)
- Liquidity checks before execution
- Spread validation
- News event avoidance
- Automatic stop-loss placement
- Emergency position closure

---
**Remember**: The hardest part isn't the code - it's the discipline to execute only ONE trade when everything aligns, and ZERO trades when they don't.