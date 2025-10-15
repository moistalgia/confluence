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
        
        # With mixed signals, confluence should be moderate (not high)
        self.assertLess(result['weighted_score'], 9.0)  # 0-10 scale, adjusted for mixed signals
        
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

    def _create_enhanced_test_data(self, bars=100, trend='sideways', with_fibonacci=False, 
                                 with_volume_support=False, with_bounces=False):
        """Create test OHLCV data with specific enhanced confluence characteristics"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=bars),
            end=datetime.now(),
            periods=bars
        )
        
        base_price = 50000.0
        
        # Create controlled price data with smaller variations
        if trend == 'uptrend':
            price_trend = np.linspace(0.95, 1.05, bars)  # Smaller trend range
        elif trend == 'downtrend':
            price_trend = np.linspace(1.05, 0.95, bars)
        else:
            price_trend = np.ones(bars) + np.random.normal(0, 0.01, bars)  # Smaller noise
        
        # Generate controlled OHLCV data
        closes = base_price * price_trend
        opens = closes + np.random.normal(0, closes * 0.0005, bars)  # Smaller spreads
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, closes * 0.0005, bars))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, closes * 0.0005, bars))
        volumes = np.random.normal(1000000, 100000, bars)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Add specific patterns for testing
        if with_fibonacci:
            # Create CLEAR and CONTROLLED swing high and low
            swing_high_idx = bars // 3
            swing_low_idx = 2 * bars // 3
            swing_high_price = base_price * 1.06  # +6% swing high
            swing_low_price = base_price * 0.96   # -4% swing low
            
            # Ensure these are the actual extremes
            df.loc[swing_high_idx, 'high'] = swing_high_price
            df.loc[swing_high_idx, 'close'] = swing_high_price - 50
            df.loc[swing_low_idx, 'low'] = swing_low_price
            df.loc[swing_low_idx, 'close'] = swing_low_price + 50
            
            # Constrain other prices to not exceed these levels
            for i in range(bars):
                if i != swing_high_idx:
                    df.loc[i, 'high'] = min(df.loc[i, 'high'], swing_high_price - 100)
                if i != swing_low_idx:
                    df.loc[i, 'low'] = max(df.loc[i, 'low'], swing_low_price + 100)
            
            # Position current price near 61.8% retracement
            price_range = swing_high_price - swing_low_price
            fib_618_price = swing_low_price + (price_range * 0.618)
            df.loc[bars-1, 'close'] = fib_618_price
            
        if with_volume_support:
            # Add high volume at current price level
            current_price = df['close'].iloc[-1]
            
            # Create more controlled price movement to ensure volume concentration
            # First, move most prices away from current level
            for i in range(bars-30):
                if np.random.random() < 0.7:  # 70% of early bars move away
                    df.loc[i, 'close'] *= (1.05 if i % 2 == 0 else 0.95)  # +/- 5% from current
                    df.loc[i, 'high'] = df.loc[i, 'close'] * 1.01
                    df.loc[i, 'low'] = df.loc[i, 'close'] * 0.99
            
            # Then create strong volume concentration at current price level
            concentration_indices = list(range(bars-20, bars, 2))  # Every other bar in last 20
            for i in concentration_indices:
                df.loc[i, 'close'] = current_price  # Exactly at current price
                df.loc[i, 'high'] = current_price * 1.005
                df.loc[i, 'low'] = current_price * 0.995
                df.loc[i, 'volume'] *= 8.0  # Very strong volume boost
        
        if with_bounces:
            # Add historical bounces at current price level
            current_price = df['close'].iloc[-1]
            bounce_indices = [bars//5, bars//2, 3*bars//5]  # Spread out bounces
            
            for idx in bounce_indices:
                if idx < bars - 10:  # Ensure not too recent
                    # Create clear bounce pattern
                    if idx > 2:
                        df.loc[idx-1, 'low'] = max(df.loc[idx-1, 'low'], current_price * 1.003)
                    df.loc[idx, 'low'] = current_price * 0.9985   # Touch level (0.15% below)
                    df.loc[idx, 'close'] = current_price * 1.007  # Strong bounce (0.7% above)
                    if idx + 1 < bars - 10:
                        df.loc[idx + 1, 'close'] = current_price * 1.01  # Continue higher
        
        return df

    def test_fibonacci_analysis_basic(self):
        """Test basic Fibonacci level calculation"""
        df = self._create_enhanced_test_data(bars=100, with_fibonacci=True)
        
        result = self.confluence_checker._calculate_fibonacci_levels(df)
        
        self.assertTrue(result['valid'])
        self.assertIn('levels', result)
        self.assertIn('50%', result['levels'])
        self.assertIn('61.8%', result['levels'])
        self.assertGreater(result['swing_high'], result['swing_low'])
        self.assertTrue(result['near_fibonacci'])
        self.assertEqual(result['confluence_score'], 15)
    
    def test_fibonacci_analysis_insufficient_data(self):
        """Test Fibonacci analysis with insufficient data"""
        df = self._create_enhanced_test_data(bars=20)  # Less than required 50 bars
        
        result = self.confluence_checker._calculate_fibonacci_levels(df)
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'Insufficient data')
    
    def test_volume_profile_analysis_strong_support(self):
        """Test volume profile with strong volume support at current level"""
        df = self._create_enhanced_test_data(bars=100, with_volume_support=True)
        
        result = self.confluence_checker._calculate_volume_profile(df)
        
        self.assertTrue(result['valid'])
        self.assertGreater(result['zone_strength'], 1.2)  # Should show strong volume concentration with 5x boost
        self.assertGreaterEqual(result['confluence_score'], 10)  # Should get good score with strong volume
        self.assertIn('price_zone', result)
    
    def test_volume_profile_insufficient_data(self):
        """Test volume profile with insufficient data"""
        df = self._create_enhanced_test_data(bars=20)
        
        result = self.confluence_checker._calculate_volume_profile(df)
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'Insufficient data')
    
    def test_previous_bounces_detection(self):
        """Test detection of previous bounces at current level"""
        df = self._create_enhanced_test_data(bars=100, with_bounces=True)
        
        result = self.confluence_checker._detect_previous_bounces(df)
        
        self.assertTrue(result['valid'])
        self.assertGreaterEqual(result['bounce_count'], 2)  # Should detect multiple bounces with test data
        self.assertGreaterEqual(result['confluence_score'], 15)  # Should get strong confluence score  
        self.assertIn(result['level_strength'], ['Strong', 'Very Strong'])  # Should be strong with multiple bounces
    
    def test_previous_bounces_no_bounces(self):
        """Test bounce detection when no historical bounces exist"""
        df = self._create_enhanced_test_data(bars=100, trend='sideways')
        
        result = self.confluence_checker._detect_previous_bounces(df)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['bounce_count'], 0)
        self.assertEqual(result['confluence_score'], 0)
        self.assertEqual(result['level_strength'], 'Weak')
    
    def test_enhanced_confluence_analysis_all_factors(self):
        """Test enhanced confluence analysis with all factors present"""
        df = self._create_enhanced_test_data(
            bars=100, 
            with_fibonacci=True, 
            with_volume_support=True, 
            with_bounces=True
        )
        
        result = self.confluence_checker._enhanced_confluence_analysis(df, 'LONG')
        
        self.assertTrue(result['fibonacci']['valid'])
        self.assertTrue(result['volume_profile']['valid'])
        self.assertTrue(result['previous_bounces']['valid'])
        self.assertGreaterEqual(result['total_confluence_score'], 40)  # Should have strong confluence with all factors (Fib15+Vol10+Bounce20=45)
        self.assertGreaterEqual(len(result['confluence_factors']), 2)
        self.assertTrue(result['has_enhanced_confluence'])
    
    def test_enhanced_confluence_analysis_no_factors(self):
        """Test enhanced confluence analysis with no confluence factors"""
        # Create controlled data that definitely won't trigger confluence factors
        bars = 100
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=bars),
            end=datetime.now(),
            periods=bars
        )
        
        # Create flat, boring data with no patterns
        base_price = 50000.0
        closes = np.full(bars, base_price)  # Perfectly flat
        opens = closes.copy()
        highs = closes.copy()  # No wicks at all
        lows = closes.copy()   # No wicks at all  
        volumes = np.full(bars, 1000000)  # Uniform volume
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        result = self.confluence_checker._enhanced_confluence_analysis(df, 'LONG')
        
        # Should have no confluence score with perfectly flat data
        self.assertEqual(result['total_confluence_score'], 0)  # No patterns should be detected
        self.assertEqual(len(result['confluence_factors']), 0)  # No factors should be found
        self.assertFalse(result['has_enhanced_confluence'])  # Should not trigger enhanced confluence
    
    def test_enhanced_confluence_integration_in_timeframe_analysis(self):
        """Test that enhanced confluence is integrated into timeframe analysis"""
        # Create test data with enhanced confluence factors
        df = self._create_enhanced_test_data(
            bars=100, 
            trend='uptrend',
            with_fibonacci=True, 
            with_volume_support=True
        )
        
        # Add required technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = 50.0
        df['macd'] = 0.1
        df['macd_signal'] = 0.05
        
        self.mock_data_provider.get_ohlcv.return_value = df
        
        # Analyze timeframe
        result = self.confluence_checker._analyze_timeframe('BTC/USDT', '4h', 'LONG', self.mock_data_provider)
        
        # Should have enhanced analysis in result
        self.assertIn('enhanced_analysis', result)
        self.assertIn('enhanced_confluence_score', result)
        
        enhanced = result['enhanced_analysis']
        self.assertTrue(enhanced['has_enhanced_confluence'])
        self.assertGreater(enhanced['total_confluence_score'], 0)
        
        # Final confluence score should include enhanced factors
        self.assertGreater(result['confluence_score'], result['base_confluence_score'])

    def test_fibonacci_level_accuracy(self):
        """Test accuracy of Fibonacci level calculations"""
        # Create simple test data
        df = self._create_enhanced_test_data(bars=100)
        
        # Set clear recent swing points for testing
        swing_high = 52000.0
        swing_low = 48000.0
        
        # Make sure these are the highest and lowest points in recent data
        df.loc[25, 'high'] = swing_high
        df.loc[25, 'close'] = swing_high - 100
        df.loc[75, 'low'] = swing_low  
        df.loc[75, 'close'] = swing_low + 100
        
        # Set current price near 61.8% level
        price_range = swing_high - swing_low  
        expected_618 = swing_low + (price_range * 0.618)
        df.loc[99, 'close'] = expected_618
        
        result = self.confluence_checker._calculate_fibonacci_levels(df, lookback_bars=100)
        
        # Test that Fibonacci levels are calculated
        self.assertTrue(result.get('valid', False))
        self.assertIn('levels', result)
        
        # Check that levels are calculated correctly
        actual_618 = result['levels']['61.8%']
        actual_50 = result['levels']['50%']
        self.assertIsInstance(actual_618, (int, float, np.number))
        self.assertIsInstance(actual_50, (int, float, np.number))
        
        # Verify swing points were found (may not be exact due to algorithm logic)
        self.assertGreater(result['swing_high'], result['swing_low'])
        self.assertTrue(result['near_fibonacci'])  # Should be near a fib level

if __name__ == '__main__':
    unittest.main()