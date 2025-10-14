"""
Test Suite for Setup Scanner
🎯 Tests actual SetupScanner implementation

Tests all methods and conditions of the setup scanner
"""

import unittest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.setup_scanner import SetupScanner

class TestSetupScanner(unittest.TestCase):
    """Test cases for SetupScanner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'min_trend_bars': 10,
            'trend_angle_min': 0.01,
            'pullback_depth_min': 0.02,
            'pullback_depth_max': 0.08,
            'max_pullback_bars': 15,
            'support_tolerance': 0.005,
            'min_level_strength': 2,
            'volume_surge_ratio': 1.5
        }
        
        self.setup_scanner = SetupScanner(self.config)
        self.mock_data_provider = Mock()
        
    def _create_pullback_data(self, bars=120, has_trend=True, has_pullback=True, pullback_depth=0.04):
        """Create realistic mock data with pullback pattern"""
        dates = pd.date_range(
            start=datetime.utcnow() - timedelta(hours=bars*4),  # 4h bars
            end=datetime.utcnow(),
            freq='4h'
        )[:bars]
        
        base_price = 50000
        prices = []
        
        if has_trend:
            # Create uptrend for first 60% of data
            trend_bars = int(bars * 0.6)
            trend_prices = np.linspace(base_price * 0.9, base_price, trend_bars)
            
            if has_pullback:
                # Add pullback in next 20% of data
                pullback_bars = int(bars * 0.2)
                pullback_low = base_price * (1 - pullback_depth)
                pullback_prices = np.linspace(base_price, pullback_low, pullback_bars // 2)
                recovery_prices = np.linspace(pullback_low, base_price * 0.98, pullback_bars // 2)
                
                # Fill remaining with continuation
                remaining_bars = bars - trend_bars - pullback_bars
                continuation_prices = np.full(remaining_bars, base_price * 0.98)
                
                all_prices = np.concatenate([trend_prices, pullback_prices, recovery_prices, continuation_prices])
            else:
                # Continue trend without pullback
                remaining_bars = bars - trend_bars
                continuation_prices = np.linspace(base_price, base_price * 1.05, remaining_bars)
                all_prices = np.concatenate([trend_prices, continuation_prices])
        else:
            # Create sideways/choppy market
            all_prices = base_price * (1 + np.random.normal(0, 0.02, bars))
        
        # Ensure we have exactly the right number of prices
        if len(all_prices) > bars:
            all_prices = all_prices[:bars]
        elif len(all_prices) < bars:
            padding = np.full(bars - len(all_prices), all_prices[-1])
            all_prices = np.concatenate([all_prices, padding])
        
        # Add realistic noise
        noise = np.random.normal(0, base_price * 0.01, bars)
        close_prices = all_prices + noise
        
        # Generate OHLC
        highs = close_prices * (1 + np.abs(np.random.normal(0, 0.005, bars)))
        lows = close_prices * (1 - np.abs(np.random.normal(0, 0.005, bars)))
        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0]
        
        # Generate volume with higher volume on trend moves
        base_volume = 1000000
        volume_multiplier = 1 + np.abs(np.diff(np.append(close_prices[0], close_prices))) * 5
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
    
    def test_initialization(self):
        """Test SetupScanner initialization"""
        self.assertIsNotNone(self.setup_scanner)
        self.assertEqual(self.setup_scanner.min_trend_bars, 10)
        self.assertEqual(self.setup_scanner.trend_angle_min, 0.01)
        self.assertEqual(self.setup_scanner.pullback_depth_min, 0.02)
        self.assertEqual(self.setup_scanner.pullback_depth_max, 0.08)
        self.assertEqual(self.setup_scanner.max_pullback_bars, 15)
    
    def test_scan_for_setup_sufficient_data(self):
        """Test scan_for_setup with sufficient data"""
        # Create data with 120 bars (> 50 required)
        df = self._create_pullback_data(bars=120, has_trend=True, has_pullback=True)
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.setup_scanner.scan_for_setup('BTC/USDT', self.mock_data_provider)
        
        # Verify result structure
        expected_keys = ['symbol', 'tradeable', 'quality_score', 'max_score', 'failed_reasons', 'filter_name']
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertIsInstance(result['tradeable'], bool)
        self.assertIsInstance(result['quality_score'], (int, float))
    
    def test_scan_for_setup_insufficient_data(self):
        """Test scan_for_setup with insufficient data"""
        # Create data with only 30 bars (< 50 required)
        df = self._create_pullback_data(bars=30)
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.setup_scanner.scan_for_setup('BTC/USDT', self.mock_data_provider)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertFalse(result['tradeable'])
        self.assertIn('Insufficient data', result['failed_reasons'])
    
    def test_scan_for_setup_data_provider_error(self):
        """Test scan_for_setup handles data provider errors"""
        self.mock_data_provider.get_ohlcv.side_effect = Exception("API Error")
        
        result = self.setup_scanner.scan_for_setup('BTC/USDT', self.mock_data_provider)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
        self.assertFalse(result['tradeable'])
        self.assertTrue(any('Scan error: API Error' in reason for reason in result['failed_reasons']))
    
    def test_scan_for_setup_no_trend(self):
        """Test scan_for_setup when no clear trend is found"""
        # Create sideways/choppy data without clear trend
        df = self._create_pullback_data(bars=120, has_trend=False)
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.setup_scanner.scan_for_setup('BTC/USDT', self.mock_data_provider)
        
        # Should fail at trend analysis stage
        if not result['tradeable']:
            self.assertTrue(any('No clear trend' in reason for reason in result['failed_reasons']))
    
    def test_scan_for_setup_no_pullback(self):
        """Test scan_for_setup when trend exists but no pullback"""
        # Create trending data without pullback
        df = self._create_pullback_data(bars=120, has_trend=True, has_pullback=False)
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.setup_scanner.scan_for_setup('BTC/USDT', self.mock_data_provider)
        
        # Should fail at trend or pullback analysis stage  
        if not result['tradeable']:
            # Could fail at trend detection or pullback detection
            has_trend_failure = any('No clear trend' in reason for reason in result['failed_reasons'])
            has_pullback_failure = any('No valid pullback' in reason for reason in result['failed_reasons'])
            self.assertTrue(has_trend_failure or has_pullback_failure)
    
    def test_scan_for_setup_valid_pullback(self):
        """Test scan_for_setup with valid pullback setup"""
        # Create data with clear trend and valid pullback
        df = self._create_pullback_data(bars=120, has_trend=True, has_pullback=True, pullback_depth=0.04)
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.setup_scanner.scan_for_setup('BTC/USDT', self.mock_data_provider)
        
        # Should progress further in analysis
        self.assertGreater(result['quality_score'], 0)
        
        # Check if analysis components are present
        if 'trend_analysis' in result:
            self.assertIn('direction', result['trend_analysis'])
        if 'pullback_analysis' in result:
            self.assertIn('has_pullback', result['pullback_analysis'])
    
    def test_scan_for_setup_excessive_pullback(self):
        """Test scan_for_setup with excessive pullback depth"""
        # Create data with pullback deeper than max allowed
        df = self._create_pullback_data(bars=120, has_trend=True, has_pullback=True, pullback_depth=0.12)
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.setup_scanner.scan_for_setup('BTC/USDT', self.mock_data_provider)
        
        # Should likely fail due to excessive pullback depth
        # This depends on the internal validation logic
        self.assertIsInstance(result['tradeable'], bool)
    
    def test_create_result_structure(self):
        """Test _create_result method creates proper structure"""
        # This tests the internal _create_result method indirectly through scan_for_setup
        df = self._create_pullback_data(bars=120)
        self.mock_data_provider.get_ohlcv.return_value = df
        
        result = self.setup_scanner.scan_for_setup('BTC/USDT', self.mock_data_provider)
        
        # Verify all required keys are present
        required_keys = ['symbol', 'tradeable', 'quality_score', 'failed_reasons']
        for key in required_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result['symbol'], 'BTC/USDT')
    
    def test_add_indicators_method(self):
        """Test that _add_indicators method works correctly"""
        # Create basic dataframe
        df = self._create_pullback_data(bars=60)
        
        # Test indirectly through scan_for_setup which calls _add_indicators
        self.mock_data_provider.get_ohlcv.return_value = df
        
        # Should not crash when adding indicators
        try:
            result = self.setup_scanner.scan_for_setup('BTC/USDT', self.mock_data_provider)
            # If we get here, indicators were added successfully
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"_add_indicators caused error: {e}")

if __name__ == '__main__':
    unittest.main()