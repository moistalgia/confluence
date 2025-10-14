# One Good Trade Per Day Bot
# 🎯 Philosophy: Quality Over Quantity

A disciplined trading bot that executes exactly ONE high-conviction trade per day, or sits on hands if no quality setup exists.

## Strategy Overview

This bot implements a 5-filter system designed for 60-70% win rate with 2:1 minimum risk:reward ratio:

1. **Market Regime Filter** - Eliminates 50% of bad trading days
2. **Setup Scanner** - Finds pullback-to-support in uptrend patterns  
3. **Confluence Checker** - Picks THE ONE best trade from candidates
4. **Final Risk Check** - Veto power for safety violations
5. **Entry Execution** - Disciplined limit orders with proper stops

## Target Performance
- **Trades**: 3-5 per week maximum
- **Win Rate**: 60-65%
- **Risk:Reward**: 2:1 minimum
- **Risk per Trade**: 1% of account maximum
- **Expected Return**: 30-60% annually

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

## Usage

```bash
# Run daily scan (once per day only)
python main.py

# Paper trading mode
python main.py --paper

# Backtest strategy
python backtest.py --start 2024-01-01 --end 2024-06-30
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