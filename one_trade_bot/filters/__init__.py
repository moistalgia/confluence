"""
One Trade Bot - Filters Package
🎯 5-Filter System for Quality-Over-Quantity Trading

Filter Pipeline:
1. Market Regime Filter - Eliminates 50% of bad trading days
2. Setup Scanner - Identifies pullback-to-support patterns  
3. Confluence Checker - Validates multi-timeframe alignment
4. Risk Check Filter - Final safety validation with veto power
5. Entry Execution - Disciplined limit order placement

Each filter has veto power. All 5 must pass for THE ONE TRADE.
"""

from .regime_filter import MarketRegimeFilter, filter_watchlist_by_regime
from .setup_scanner import SetupScanner, scan_watchlist_for_setups
from .confluence_checker import ConfluenceChecker, check_setup_confluence
from .risk_check import RiskCheckFilter, perform_final_risk_validation
from .entry_execution import EntryExecution, execute_entry_workflow

__all__ = [
    # Filter Classes
    'MarketRegimeFilter',
    'SetupScanner', 
    'ConfluenceChecker',
    'RiskCheckFilter',
    'EntryExecution',
    
    # Workflow Functions
    'filter_watchlist_by_regime',
    'scan_watchlist_for_setups',
    'check_setup_confluence', 
    'perform_final_risk_validation',
    'execute_entry_workflow'
]

# Filter execution order and descriptions
FILTER_PIPELINE = [
    {
        'name': 'Market Regime Filter',
        'class': MarketRegimeFilter,
        'function': filter_watchlist_by_regime,
        'purpose': 'Eliminate unfavorable market conditions',
        'expected_filter_rate': 0.5,  # Filters out ~50% of days
        'veto_power': True
    },
    {
        'name': 'Setup Scanner',
        'class': SetupScanner,
        'function': scan_watchlist_for_setups,
        'purpose': 'Identify pullback-to-support patterns',
        'expected_filter_rate': 0.8,  # Filters out ~80% of remaining
        'veto_power': True
    },
    {
        'name': 'Confluence Checker', 
        'class': ConfluenceChecker,
        'function': check_setup_confluence,
        'purpose': 'Validate multi-timeframe alignment',
        'expected_filter_rate': 0.6,  # Filters out ~60% of setups
        'veto_power': True
    },
    {
        'name': 'Risk Check Filter',
        'class': RiskCheckFilter, 
        'function': perform_final_risk_validation,
        'purpose': 'Final safety validation',
        'expected_filter_rate': 0.3,  # Filters out ~30% for safety
        'veto_power': True
    },
    {
        'name': 'Entry Execution',
        'class': EntryExecution,
        'function': execute_entry_workflow,
        'purpose': 'Disciplined trade execution',
        'expected_filter_rate': 0.0,  # No filtering, just execution
        'veto_power': False  # Execution phase, not filtering
    }
]