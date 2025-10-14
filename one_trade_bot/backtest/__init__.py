"""
One Trade Bot - Backtest Package
🎯 Historical and Real-time Testing Capabilities

Components:
- Historical Backtester: Validate 5-filter system on past data
- Paper Trading Simulator: Real-time testing with live market data
- Performance Analysis: Comprehensive metrics and reporting

Testing the complete "One Good Trade Per Day" philosophy
"""

from .backtester import HistoricalBacktester, BacktestResults, BacktestTrade, run_historical_backtest
from .paper_trader import PaperTradingSimulator, PaperAccount, PaperTrade, run_paper_trading_simulation

__all__ = [
    # Historical Backtesting
    'HistoricalBacktester',
    'BacktestResults', 
    'BacktestTrade',
    'run_historical_backtest',
    
    # Paper Trading
    'PaperTradingSimulator',
    'PaperAccount',
    'PaperTrade', 
    'run_paper_trading_simulation'
]