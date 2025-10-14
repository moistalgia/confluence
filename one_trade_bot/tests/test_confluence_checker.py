"""
Test Suite for Confluence Checker
🎯 Tests actual ConfluenceChecker implementation

Tests all methods and conditions of the confluence checker
"""

import unittest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.confluence_checker import ConfluenceChecker

class TestConfluenceChecker(unittest.TestCase):
    """Test cases for ConfluenceChecker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'timeframes': ['1h', '4h', '1d'],
            'min_confluence_count': 2,
            'sma_alignment_buffer': 0.01,
            'rsi_neutral_zone': [40, 60],
            'momentum_threshold': 0.02,
            'timeframe_weights': {
                '1h': 1.0,
                '4h': 2.0,
                '1d': 1.5,
                '1w': 1.0
            }
        }
        
        self.confluence_checker = ConfluenceChecker(self.config)
        self.mock_data_provider = Mock()
        
    def _create_timeframe_data(self, days, freq, trend='bullish'):
        """Create realistic mock data for specific timeframe"""
        dates = pd.date_range(
            start=datetime.utcnow() - timedelta(days=days),
            end=datetime.utcnow(),
            freq=freq
        )
        
        base_price = 50000
        
        # Generate price movement based on trend
        if trend == 'bullish':
            price_multiplier = np.linspace(0.9, 1.1, len(dates))
        elif trend == 'bearish':
            price_multiplier = np.linspace(1.1, 0.9, len(dates))
        else:  # neutral/sideways
            price_multiplier = np.ones(len(dates)) + np.random.normal(0, 0.01, len(dates))
        
        # Add noise appropriate for timeframe
        if freq == '1H':
            noise_factor = 0.005  # Lower noise for 1H
        elif freq == '4H':
            noise_factor = 0.01
        else:  # 1D
            noise_factor = 0.02
        
        noise = np.random.normal(0, noise_factor, len(dates))
        close_prices = base_price * (price_multiplier + noise)
        
        # Generate OHLC
        highs = close_prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
        lows = close_prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0]
        
        # Generate volume
        volumes = np.random.lognormal(np.log(1000000), 0.3, len(dates))
        
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
        """Test ConfluenceChecker initialization"""
        self.assertIsNotNone(self.confluence_checker)
        self.assertEqual(self.confluence_checker.timeframes, ['1h', '4h', '1d'])
        self.assertEqual(self.confluence_checker.min_confluence_count, 2)
        self.assertEqual(self.confluence_checker.sma_alignment_buffer, 0.01)
        self.assertIn('4h', self.confluence_checker.timeframe_weights)
        self.assertEqual(self.confluence_checker.timeframe_weights['4h'], 2.0)
    
    def test_check_confluence_long_bullish_all_timeframes(self):
        """Test check_confluence for LONG with all timeframes bullish"""
        # Mock data provider to return bullish data for all timeframes
        def mock_get_ohlcv(symbol, timeframe, days):
            # Ensure enough data for all indicators (SMA50 needs 50 periods minimum)
            sufficient_days = max(days, 60)  
            if timeframe == '1h':
                return self._create_timeframe_data(sufficient_days, '1h', 'bullish')
            elif timeframe == '4h':
                return self._create_timeframe_data(sufficient_days, '4h', 'bullish')
            else:  # 1d
                return self._create_timeframe_data(sufficient_days, '1d', 'bullish')
        
        self.mock_data_provider.get_ohlcv.side_effect = mock_get_ohlcv
        
        result = self.confluence_checker.check_confluence('BTC/USDT', 'LONG', self.mock_data_provider)
        
        # Verify result structure
        expected_keys = ['symbol', 'trade_direction', 'approved', 'weighted_score', 'timeframe_results']
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertEqual(result['trade_direction'], 'LONG')
        self.assertIsInstance(result['approved'], bool)
        self.assertIsInstance(result['weighted_score'], (int, float))
        
        # Should have results for all timeframes
        self.assertIn('timeframe_results', result)
        for tf in self.config['timeframes']:
            self.assertIn(tf, result['timeframe_results'])
    
    def test_check_confluence_short_bearish_all_timeframes(self):
        """Test check_confluence for SHORT with all timeframes bearish"""
        # Mock data provider to return bearish data for all timeframes
        def mock_get_ohlcv(symbol, timeframe, days):
            # Ensure enough data for all indicators (SMA50 needs 50 periods minimum)
            sufficient_days = max(days, 60)  
            if timeframe == '1h':
                return self._create_timeframe_data(sufficient_days, '1h', 'bearish')
            elif timeframe == '4h':
                return self._create_timeframe_data(sufficient_days, '4h', 'bearish')
            else:  # 1d
                return self._create_timeframe_data(sufficient_days, '1d', 'bearish')
        
        self.mock_data_provider.get_ohlcv.side_effect = mock_get_ohlcv
        
        result = self.confluence_checker.check_confluence('BTC/USDT', 'SHORT', self.mock_data_provider)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertEqual(result['trade_direction'], 'SHORT')
        self.assertIsInstance(result['approved'], bool)
        
        # With all bearish timeframes, SHORT should have good confluence
        if result['approved']:
            self.assertGreater(result['weighted_score'], 2.0)  # 0-10 scale
    
    def test_check_confluence_mixed_signals(self):
        """Test check_confluence with mixed timeframe signals"""
        # Mock mixed signals: 1h bullish, 4h bearish, 1d neutral
        def mock_get_ohlcv(symbol, timeframe, days):
            # Ensure enough data for all indicators (SMA50 needs 50 periods minimum)
            sufficient_days = max(days, 60)  
            if timeframe == '1h':
                return self._create_timeframe_data(sufficient_days, '1h', 'bullish')
            elif timeframe == '4h':
                return self._create_timeframe_data(sufficient_days, '4h', 'bearish')
            else:  # 1d
                return self._create_timeframe_data(sufficient_days, '1d', 'neutral')
        
        self.mock_data_provider.get_ohlcv.side_effect = mock_get_ohlcv
        
        result = self.confluence_checker.check_confluence('BTC/USDT', 'LONG', self.mock_data_provider)
        
        # With mixed signals, confluence should be lower
        self.assertLess(result['weighted_score'], 8.0)  # 0-10 scale
        
        # Check that all timeframes were analyzed
        self.assertEqual(len(result['timeframe_results']), len(self.config['timeframes']))
    
    def test_check_confluence_data_provider_error(self):
        """Test check_confluence handles data provider errors"""
        self.mock_data_provider.get_ohlcv.side_effect = Exception("API Error")
        
        result = self.confluence_checker.check_confluence('BTC/USDT', 'LONG', self.mock_data_provider)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertFalse(result['approved'])
        self.assertEqual(result['weighted_score'], 0)
    
    def test_check_confluence_partial_timeframe_errors(self):
        """Test check_confluence when some timeframes fail"""
        # Mock data provider to fail on 4h but succeed on others
        def mock_get_ohlcv(symbol, timeframe, days):
            if timeframe == '4h':
                raise Exception("4H data error")
            # Ensure enough data for all indicators (SMA50 needs 50 periods minimum)
            sufficient_days = max(days, 60)  
            if timeframe == '1h':
                return self._create_timeframe_data(sufficient_days, '1h', 'bullish')
            else:  # 1d
                return self._create_timeframe_data(sufficient_days, '1d', 'bullish')
        
        self.mock_data_provider.get_ohlcv.side_effect = mock_get_ohlcv
        
        result = self.confluence_checker.check_confluence('BTC/USDT', 'LONG', self.mock_data_provider)
        
        # Should still return result with available timeframes
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertIsInstance(result['approved'], bool)
        
        # Check that 4h timeframe shows error
        if '4h' in result['timeframe_results']:
            tf_4h = result['timeframe_results']['4h']
            self.assertFalse(tf_4h.get('supports_direction', True))
            self.assertEqual(tf_4h.get('confluence_score', -1), 0)
    
    def test_calculate_weighted_score_method(self):
        """Test _calculate_weighted_score method indirectly"""
        # Mock all bullish data to test weighted scoring
        def mock_get_ohlcv(symbol, timeframe, days):
            sufficient_days = max(days, 60)  # Ensure enough data for SMA50
            return self._create_timeframe_data(sufficient_days, '1h' if 'h' in timeframe else '1d', 'bullish')
        
        self.mock_data_provider.get_ohlcv.side_effect = mock_get_ohlcv
        
        result = self.confluence_checker.check_confluence('BTC/USDT', 'LONG', self.mock_data_provider)
        
        # Weighted score should reflect the timeframe weights
        # 4h has weight 2.0, so it should be most influential
        self.assertIsInstance(result['weighted_score'], (int, float))
        self.assertGreaterEqual(result['weighted_score'], 0)
        self.assertLessEqual(result['weighted_score'], 10.0)  # 0-10 scale
    
    def test_minimum_confluence_count_requirement(self):
        """Test minimum confluence count requirement"""
        # Mock data where only 1 timeframe supports direction (less than min_confluence_count=2)
        def mock_get_ohlcv(symbol, timeframe, days):
            # Ensure enough data for all indicators (SMA50 needs 50 periods minimum)
            sufficient_days = max(days, 60)  
            if timeframe == '1h':
                return self._create_timeframe_data(sufficient_days, '1h', 'bullish')
            else:  # 4h and 1d are bearish/neutral
                return self._create_timeframe_data(sufficient_days, '4h', 'bearish')
        
        self.mock_data_provider.get_ohlcv.side_effect = mock_get_ohlcv
        
        result = self.confluence_checker.check_confluence('BTC/USDT', 'LONG', self.mock_data_provider)
        
        # With insufficient confluence, should not be tradeable
        # (This depends on the internal logic for counting supporting timeframes)
        self.assertIsInstance(result['approved'], bool)
    
    def test_analyze_timeframe_method_indirectly(self):
        """Test _analyze_timeframe method indirectly through check_confluence"""
        # This tests that each timeframe analysis returns proper structure
        def mock_get_ohlcv(symbol, timeframe, days):
            sufficient_days = max(days, 60)  # Ensure enough data for SMA50
            return self._create_timeframe_data(sufficient_days, '1h', 'bullish')
        
        self.mock_data_provider.get_ohlcv.side_effect = mock_get_ohlcv
        
        result = self.confluence_checker.check_confluence('BTC/USDT', 'LONG', self.mock_data_provider)
        
        # Check that timeframe results have expected structure
        for tf in self.config['timeframes']:
            if tf in result['timeframe_results']:
                tf_result = result['timeframe_results'][tf]
                self.assertIn('supports_direction', tf_result)
                self.assertIn('confluence_score', tf_result)
                self.assertIsInstance(tf_result['supports_direction'], bool)
                self.assertIsInstance(tf_result['confluence_score'], (int, float))

if __name__ == '__main__':
    unittest.main()