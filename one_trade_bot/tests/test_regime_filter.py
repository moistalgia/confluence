"""
Test Suite for Market Regime Filter
🎯 Tests actual MarketRegimeFilter implementation

Tests all methods and conditions of the regime filter
"""

import unittest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.regime_filter import MarketRegimeFilter
from core.data_provider import TechnicalIndicators

class TestMarketRegimeFilter(unittest.TestCase):
    """Test cases for MarketRegimeFilter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'min_trend_distance': 0.02,  # 2% from SMA
            'max_atr_pct': 0.08,        # 8% ATR/price
            'min_volume_ratio': 0.5,     # 50% of average
            'max_bb_squeeze': 0.15,      # BB squeeze limit
            'min_checks_pass': 3         # 3 of 4 checks
        }
        
        self.regime_filter = MarketRegimeFilter(self.config)
        self.mock_data_provider = Mock()
        
    def _create_mock_data(self, days=30, trend='up', volatility='normal', volume='normal'):
        """Create realistic mock OHLCV data"""
        dates = pd.date_range(
            start=datetime.utcnow() - timedelta(days=days),
            end=datetime.utcnow(),
            freq='1D'
        )
        
        base_price = 50000
        
        # Generate price trend
        if trend == 'up':
            price_multiplier = np.linspace(0.9, 1.1, len(dates))
        elif trend == 'down':
            price_multiplier = np.linspace(1.1, 0.9, len(dates))
        else:  # sideways
            price_multiplier = np.ones(len(dates))
        
        # Add volatility
        if volatility == 'high':
            noise = np.random.normal(0, 0.05, len(dates))  # 5% daily moves
        elif volatility == 'low':
            noise = np.random.normal(0, 0.005, len(dates))  # 0.5% daily moves
        else:  # normal
            noise = np.random.normal(0, 0.02, len(dates))  # 2% daily moves
        
        close_prices = base_price * (price_multiplier + noise)
        
        # Generate OHLC
        highs = close_prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        lows = close_prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0]
        
        # Generate volume
        base_volume = 1000000
        if volume == 'high':
            volume_multiplier = np.random.normal(2.0, 0.5, len(dates))
        elif volume == 'low':
            volume_multiplier = np.random.normal(0.3, 0.1, len(dates))
        else:  # normal
            volume_multiplier = np.random.normal(1.0, 0.3, len(dates))
        
        volumes = base_volume * np.abs(volume_multiplier)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes
        })
        
        return df.set_index('timestamp')
    
    def test_initialization(self):
        """Test MarketRegimeFilter initialization"""
        self.assertIsNotNone(self.regime_filter)
        self.assertEqual(self.regime_filter.min_trend_distance, 0.02)
        self.assertEqual(self.regime_filter.max_atr_pct, 0.08)
        self.assertEqual(self.regime_filter.min_volume_ratio, 0.5)
        self.assertEqual(self.regime_filter.max_bb_squeeze, 0.15)
        self.assertEqual(self.regime_filter.min_checks_pass, 3)
    
    def test_check_regime_sufficient_data(self):
        """Test check_regime with sufficient data"""
        # Create data with 30 days (> 25 required)
        df = self._create_mock_data(days=30, trend='up')
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.regime_filter.check_regime('BTC/USDT', self.mock_data_provider)
        
        # Verify result structure
        self.assertIn('symbol', result)
        self.assertIn('tradeable', result)
        self.assertIn('score', result)
        self.assertIn('max_score', result)
        self.assertIn('regime', result)
        self.assertIn('failed_reasons', result)
        self.assertIn('checks', result)
        self.assertIn('filter_name', result)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertEqual(result['max_score'], 4)
        self.assertEqual(result['filter_name'], 'MarketRegimeFilter')
        self.assertIsInstance(result['tradeable'], bool)
        self.assertIsInstance(result['score'], int)
    
    def test_check_regime_insufficient_data(self):
        """Test check_regime with insufficient data"""
        # Create data with only 10 days (< 25 required)
        df = self._create_mock_data(days=10)
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.regime_filter.check_regime('BTC/USDT', self.mock_data_provider)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertFalse(result['tradeable'])
        self.assertEqual(result['score'], 0)
        self.assertIn('Insufficient historical data', result['failed_reasons'])
    
    def test_check_regime_data_provider_error(self):
        """Test check_regime handles data provider errors"""
        self.mock_data_provider.get_ohlcv.side_effect = Exception("API Error")
        
        result = self.regime_filter.check_regime('BTC/USDT', self.mock_data_provider)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertFalse(result['tradeable'])
        self.assertEqual(result['score'], 0)
        self.assertTrue(any('Analysis error: API Error' in reason for reason in result['failed_reasons']))
    
    def test_run_regime_checks_all_pass(self):
        """Test _run_regime_checks when all checks pass"""
        # Set up values that should pass all checks
        price = 50000
        sma_20 = 48000  # Price is 4.2% above SMA (> 2% required)
        atr = 2000      # ATR is 4% of price (< 8% allowed)
        volume_today = 1500000
        volume_avg = 1000000  # Today's volume is 1.5x average (> 0.5 required)
        squeeze_ratio = 0.2   # Above 0.15 limit
        
        checks = self.regime_filter._run_regime_checks(
            price, sma_20, atr, volume_today, volume_avg, squeeze_ratio
        )
        
        self.assertIn('trending', checks)
        self.assertIn('not_too_volatile', checks)
        self.assertIn('has_volume', checks)
        self.assertIn('not_in_squeeze', checks)
        
        self.assertTrue(checks['trending'])
        self.assertTrue(checks['not_too_volatile'])
        self.assertTrue(checks['has_volume'])
        self.assertTrue(checks['not_in_squeeze'])
    
    def test_run_regime_checks_trending_fail(self):
        """Test _run_regime_checks when trending check fails"""
        price = 50000
        sma_20 = 49900  # Price is only 0.2% above SMA (< 2% required)
        atr = 2000
        volume_today = 1500000
        volume_avg = 1000000
        squeeze_ratio = 0.2
        
        checks = self.regime_filter._run_regime_checks(
            price, sma_20, atr, volume_today, volume_avg, squeeze_ratio
        )
        
        self.assertFalse(checks['trending'])
        self.assertTrue(checks['not_too_volatile'])
        self.assertTrue(checks['has_volume'])
        self.assertTrue(checks['not_in_squeeze'])
    
    def test_run_regime_checks_volatility_fail(self):
        """Test _run_regime_checks when volatility check fails"""
        price = 50000
        sma_20 = 48000
        atr = 5000      # ATR is 10% of price (> 8% allowed)
        volume_today = 1500000
        volume_avg = 1000000
        squeeze_ratio = 0.2
        
        checks = self.regime_filter._run_regime_checks(
            price, sma_20, atr, volume_today, volume_avg, squeeze_ratio
        )
        
        self.assertTrue(checks['trending'])
        self.assertFalse(checks['not_too_volatile'])
        self.assertTrue(checks['has_volume'])
        self.assertTrue(checks['not_in_squeeze'])
    
    def test_run_regime_checks_volume_fail(self):
        """Test _run_regime_checks when volume check fails"""
        price = 50000
        sma_20 = 48000
        atr = 2000
        volume_today = 400000
        volume_avg = 1000000  # Today's volume is only 0.4x average (< 0.5 required)
        squeeze_ratio = 0.2
        
        checks = self.regime_filter._run_regime_checks(
            price, sma_20, atr, volume_today, volume_avg, squeeze_ratio
        )
        
        self.assertTrue(checks['trending'])
        self.assertTrue(checks['not_too_volatile'])
        self.assertFalse(checks['has_volume'])
        self.assertTrue(checks['not_in_squeeze'])
    
    def test_run_regime_checks_squeeze_fail(self):
        """Test _run_regime_checks when squeeze check fails"""
        price = 50000
        sma_20 = 48000
        atr = 2000
        volume_today = 1500000
        volume_avg = 1000000
        squeeze_ratio = 0.1   # Below 0.15 limit (in squeeze)
        
        checks = self.regime_filter._run_regime_checks(
            price, sma_20, atr, volume_today, volume_avg, squeeze_ratio
        )
        
        self.assertTrue(checks['trending'])
        self.assertTrue(checks['not_too_volatile'])
        self.assertTrue(checks['has_volume'])
        self.assertFalse(checks['not_in_squeeze'])
    
    def test_create_result_structure(self):
        """Test _create_result creates proper structure"""
        result = self.regime_filter._create_result(
            symbol='BTC/USDT',
            tradeable=True,
            score=3,
            reasons=[],
            regime='TRENDING',
            checks={'trending': True, 'not_too_volatile': True}
        )
        
        expected_keys = ['symbol', 'tradeable', 'score', 'max_score', 'regime', 'failed_reasons', 'checks', 'filter_name']
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertTrue(result['tradeable'])
        self.assertEqual(result['score'], 3)
        self.assertEqual(result['max_score'], 4)
        self.assertEqual(result['regime'], 'TRENDING')
        self.assertEqual(result['filter_name'], 'MarketRegimeFilter')
    
    def test_check_regime_trending_market(self):
        """Test check_regime with trending market data"""
        # Create strong uptrend data
        df = self._create_mock_data(days=30, trend='up', volatility='normal', volume='normal')
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.regime_filter.check_regime('BTC/USDT', self.mock_data_provider)
        
        # Trending market should likely pass regime checks
        self.assertGreater(result['score'], 0)
        if result['tradeable']:
            self.assertEqual(result['regime'], 'TRENDING')
    
    def test_check_regime_sideways_market(self):
        """Test check_regime with sideways market data"""
        # Create sideways market data
        df = self._create_mock_data(days=30, trend='sideways', volatility='low', volume='normal')
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.regime_filter.check_regime('BTC/USDT', self.mock_data_provider)
        
        # Sideways market should likely fail trending check
        if not result['tradeable']:
            self.assertIn('trending', [check for check, passed in result['checks'].items() if not passed])

if __name__ == '__main__':
    unittest.main()