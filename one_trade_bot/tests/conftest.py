"""
Test Configuration
🎯 Shared test configuration and fixtures

Provides mock data and configuration for all test modules
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import tempfile
import os

# Test configuration
TEST_CONFIG = {
    'data_provider': {
        'exchange': 'binance',
        'timeout': 10000,
        'rateLimit': 1200,
        'sandbox': True
    },
    'account': {
        'balance': 10000.0
    },
    'watchlist': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    'regime_filter': {
        'min_trend_distance': 0.02,
        'max_atr_pct': 0.08,
        'min_volume_ratio': 0.5,
        'max_bb_squeeze': 0.15,
        'min_checks_pass': 3
    },
    'setup_scanner': {
        'min_trend_bars': 10,
        'trend_angle_min': 0.01,
        'pullback_depth_min': 0.02,
        'pullback_depth_max': 0.08,
        'max_pullback_bars': 15
    },
    'confluence_checker': {
        'timeframes': ['1h', '4h', '1d'],
        'min_confluence_count': 2,
        'sma_alignment_buffer': 0.01
    },
    'risk_check': {
        'max_account_risk': 0.01,
        'max_portfolio_risk': 0.05,
        'max_correlation': 0.7
    },
    'entry_execution': {
        'max_entry_wait_minutes': 60,
        'profit_target_ratio': 2.0,
        'account_risk_pct': 0.01
    },
    'position_manager': {
        'max_account_risk': 0.01,
        'profit_target_ratio': 2.0,
        'trailing_stop_enabled': True
    },
    'trade_logger': {
        'trades_file': 'test_data/trades.json',
        'daily_log_file': 'test_data/daily_logs.json'
    }
}

@pytest.fixture
def test_config():
    """Provide test configuration"""
    return TEST_CONFIG.copy()

@pytest.fixture
def temp_dir():
    """Provide temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture  
def mock_ohlcv_data():
    """Generate mock OHLCV data for testing"""
    def _generate_data(days: int = 30, symbol: str = 'BTC/USDT', trend: str = 'up') -> pd.DataFrame:
        """
        Generate realistic OHLCV data
        
        Args:
            days: Number of days of data
            symbol: Trading pair
            trend: 'up', 'down', or 'sideways'
        """
        dates = pd.date_range(
            start=datetime.utcnow() - timedelta(days=days),
            end=datetime.utcnow(),
            freq='1H'
        )
        
        # Base price
        base_price = 50000 if 'BTC' in symbol else 3000
        
        # Generate price movement based on trend
        if trend == 'up':
            trend_factor = np.linspace(0.95, 1.05, len(dates))
        elif trend == 'down':
            trend_factor = np.linspace(1.05, 0.95, len(dates))  
        else:  # sideways
            trend_factor = np.ones(len(dates)) + np.random.normal(0, 0.02, len(dates))
        
        # Add realistic noise
        noise = np.random.normal(0, 0.01, len(dates))
        price_multiplier = trend_factor + noise
        
        # Generate OHLCV
        close_prices = base_price * price_multiplier
        
        # Generate realistic OHLC from close prices
        highs = close_prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
        lows = close_prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0]
        
        # Generate volume (higher volume on trend moves)
        base_volume = 1000000
        volume_multiplier = 1 + np.abs(np.diff(np.append([close_prices[0]], close_prices))) * 10
        volumes = base_volume * volume_multiplier
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes
        })
        
        return df.set_index('timestamp')
    
    return _generate_data

@pytest.fixture
def mock_data_provider(mock_ohlcv_data):
    """Mock DataProvider for testing"""
    class MockDataProvider:
        def __init__(self, config: Dict = None):
            self.config = config or {}
            self._data_cache = {}
        
        def get_ohlcv(self, symbol: str, timeframe: str = '1h', days: int = 30) -> pd.DataFrame:
            """Return mock OHLCV data"""
            cache_key = f"{symbol}_{timeframe}_{days}"
            
            if cache_key not in self._data_cache:
                # Generate different trends for different symbols
                trend = 'up' if 'BTC' in symbol else 'sideways' if 'ETH' in symbol else 'down'
                self._data_cache[cache_key] = mock_ohlcv_data(days, symbol, trend)
            
            return self._data_cache[cache_key]
        
        def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
            """Mock validation - always passes"""
            return len(df) > 0
    
    return MockDataProvider

@pytest.fixture
def sample_workflow_result():
    """Sample workflow result for testing"""
    return {
        'workflow_status': 'TRADE_READY',
        'message': 'THE ONE TRADE: BTC/USDT LONG',
        'the_one_trade': 'BTC/USDT',
        'trade_direction': 'LONG',
        'start_time': '2024-01-01T09:00:00',
        'end_time': '2024-01-01T09:05:00',
        'duration_seconds': 300,
        'filter_results': {
            'regime': {
                'tradeable_symbols': ['BTC/USDT', 'ETH/USDT'],
                'filtered_symbols': ['SOL/USDT'],
                'summary': {
                    'total_checked': 3,
                    'tradeable_count': 2,
                    'filter_rate': 0.33
                }
            },
            'setup': {
                'setup_symbols': ['BTC/USDT'],
                'no_setup_symbols': ['ETH/USDT'],
                'summary': {
                    'total_scanned': 2,
                    'setups_count': 1
                }
            },
            'confluence': {
                'approved_symbols': ['BTC/USDT'],
                'rejected_symbols': [],
                'summary': {
                    'total_checked': 1,
                    'approved_count': 1,
                    'approval_rate': 1.0
                }
            },
            'risk': {
                'final_approved_symbols': ['BTC/USDT'],
                'risk_rejected_symbols': [],
                'summary': {
                    'total_checked': 1,
                    'final_approved_count': 1
                }
            }
        },
        'entry_plan': {
            'symbol': 'BTC/USDT',
            'trade_direction': 'LONG',
            'plan_ready': True,
            'current_price': 50000.0,
            'entry_price': 49950.0,
            'stop_loss_price': 49000.0,
            'profit_target_price': 51900.0,
            'position_size': 0.2,
            'position_value_usd': 9990.0,
            'risk_amount': 100.0,
            'potential_profit': 390.0,
            'risk_reward_ratio': 2.05,
            'stop_distance_pct': 0.019,
            'account_risk_pct': 0.01
        }
    }

@pytest.fixture
def sample_trades():
    """Sample trade records for testing"""
    return [
        {
            'trade_id': 'TRADE_20240101_090000',
            'timestamp': '2024-01-01T09:00:00',
            'symbol': 'BTC/USDT',
            'direction': 'LONG',
            'entry_price': 50000.0,
            'exit_price': 51000.0,
            'position_size': 0.2,
            'net_pnl': 200.0,
            'pnl_percentage': 0.02,
            'risk_amount': 100.0,
            'reward_risk_ratio': 2.0,
            'exit_reason': 'PROFIT_TARGET'
        },
        {
            'trade_id': 'TRADE_20240102_090000', 
            'timestamp': '2024-01-02T09:00:00',
            'symbol': 'ETH/USDT',
            'direction': 'SHORT',
            'entry_price': 3000.0,
            'exit_price': 2950.0,
            'position_size': 3.0,
            'net_pnl': 150.0,
            'pnl_percentage': 0.0167,
            'risk_amount': 90.0,
            'reward_risk_ratio': 1.67,
            'exit_reason': 'PROFIT_TARGET'
        },
        {
            'trade_id': 'TRADE_20240103_090000',
            'timestamp': '2024-01-03T09:00:00', 
            'symbol': 'BTC/USDT',
            'direction': 'LONG',
            'entry_price': 51000.0,
            'exit_price': 50500.0,
            'position_size': 0.19,
            'net_pnl': -95.0,
            'pnl_percentage': -0.0098,
            'risk_amount': 100.0,
            'reward_risk_ratio': -0.95,
            'exit_reason': 'STOP_LOSS'
        }
    ]

# Test utilities
def create_test_position(symbol: str = 'BTC/USDT', direction: str = 'LONG') -> Dict:
    """Create test position data"""
    return {
        'symbol': symbol,
        'direction': direction,
        'entry_price': 50000.0,
        'position_size': 0.2,
        'stop_loss_price': 49000.0 if direction == 'LONG' else 51000.0,
        'profit_target_price': 52000.0 if direction == 'LONG' else 48000.0,
        'risk_amount': 100.0,
        'account_balance_at_entry': 10000.0,
        'entry_time': datetime.utcnow().isoformat(),
        'last_update_time': datetime.utcnow().isoformat(),
        'status': 'OPEN',
        'current_price': 50000.0,
        'unrealized_pnl': 0.0
    }

def assert_filter_result_structure(result: Dict, filter_name: str) -> None:
    """Assert that filter result has expected structure"""
    required_keys = ['symbol', 'tradeable', 'filter_name']
    
    for key in required_keys:
        assert key in result, f"Filter {filter_name} result missing key: {key}"
    
    assert result['filter_name'] == filter_name
    assert isinstance(result['tradeable'], bool)

def assert_workflow_result_structure(result: Dict) -> None:
    """Assert that workflow result has expected structure"""
    required_keys = [
        'workflow_status', 'message', 'start_time', 'end_time', 
        'duration_seconds', 'filter_results', 'daily_decision'
    ]
    
    for key in required_keys:
        assert key in result, f"Workflow result missing key: {key}"
    
    assert result['workflow_status'] in [
        'TRADE_READY', 'NO_TRADEABLE_REGIME', 'NO_SETUPS_FOUND',
        'NO_CONFLUENCE', 'RISK_REJECTED', 'POSITION_ALREADY_OPEN',
        'EXECUTION_FAILED', 'WORKFLOW_ERROR'
    ]