"""
Test Suite for Workflow Engine
🎯 Tests actual DailyWorkflowEngine implementation

Tests workflow orchestration and filter integration
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.workflow_engine import DailyWorkflowEngine

class TestDailyWorkflowEngine(unittest.TestCase):
    """Test cases for DailyWorkflowEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'watchlist': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            'account': {'balance': 10000.0},
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
            }
        }
        
        self.mock_data_provider = Mock()
        self.workflow_engine = DailyWorkflowEngine(self.config, self.mock_data_provider)
    
    def test_initialization(self):
        """Test DailyWorkflowEngine initialization"""
        self.assertIsNotNone(self.workflow_engine)
        self.assertEqual(self.workflow_engine.watchlist, ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
        self.assertEqual(self.workflow_engine.account_config['balance'], 10000.0)
        self.assertIn('regime', self.workflow_engine.filter_configs)
        self.assertIn('setup', self.workflow_engine.filter_configs)
        self.assertIn('confluence', self.workflow_engine.filter_configs)
        self.assertIn('risk', self.workflow_engine.filter_configs)
        self.assertIn('entry', self.workflow_engine.filter_configs)
    
    def test_run_daily_workflow_with_open_positions(self):
        """Test run_daily_workflow when positions already exist"""
        # Mock existing open positions
        open_positions = [
            {'symbol': 'BTC/USDT', 'status': 'OPEN', 'entry_time': '2024-01-01T09:00:00'}
        ]
        
        result = self.workflow_engine.run_daily_workflow(open_positions)
        
        # Should skip workflow due to existing position
        self.assertEqual(result['workflow_status'], 'POSITION_ALREADY_OPEN')
        self.assertIn('Daily workflow skipped', result['message'])
        self.assertIn('start_time', result)
        self.assertIn('end_time', result)
    
    @patch('core.workflow_engine.filter_watchlist_by_regime')
    def test_run_daily_workflow_no_tradeable_regime(self, mock_regime_filter):
        """Test run_daily_workflow when regime filter finds no tradeable symbols"""
        # Mock regime filter to return no tradeable symbols
        mock_regime_filter.return_value = {
            'tradeable_symbols': [],
            'filtered_symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            'regime_results': {},
            'summary': {
                'total_checked': 3,
                'tradeable_count': 0,
                'filter_rate': 1.0
            }
        }
        
        result = self.workflow_engine.run_daily_workflow()
        
        # Should stop at regime filter
        self.assertEqual(result['workflow_status'], 'NO_TRADEABLE_REGIME')
        self.assertIn('Market regime unfavorable', result['message'])
        
        # Verify regime filter was called with correct parameters
        mock_regime_filter.assert_called_once()
        call_args = mock_regime_filter.call_args
        self.assertEqual(call_args[0][0], self.config['watchlist'])  # watchlist
        self.assertEqual(call_args[0][1], self.mock_data_provider)   # data_provider
    
    @patch('core.workflow_engine.scan_watchlist_for_setups')  
    @patch('core.workflow_engine.filter_watchlist_by_regime')
    def test_run_daily_workflow_no_setups_found(self, mock_regime_filter, mock_setup_scanner):
        """Test run_daily_workflow when setup scanner finds no setups"""
        # Mock regime filter success
        mock_regime_filter.return_value = {
            'tradeable_symbols': ['BTC/USDT', 'ETH/USDT'],
            'filtered_symbols': ['SOL/USDT'],
            'regime_results': {},
            'summary': {'tradeable_count': 2, 'total_checked': 3}
        }
        
        # Mock setup scanner to find no setups
        mock_setup_scanner.return_value = {
            'setup_symbols': [],
            'no_setup_symbols': ['BTC/USDT', 'ETH/USDT'],
            'results': {},
            'summary': {'setups_count': 0, 'total_scanned': 2}
        }
        
        result = self.workflow_engine.run_daily_workflow()
        
        # Should stop at setup scanner
        self.assertEqual(result['workflow_status'], 'NO_SETUPS_FOUND')
        self.assertIn('No pullback setups identified', result['message'])
        
        # Verify both filters were called
        mock_regime_filter.assert_called_once()
        mock_setup_scanner.assert_called_once()
    
    @patch('core.workflow_engine.check_setup_confluence')
    @patch('core.workflow_engine.scan_watchlist_for_setups')  
    @patch('core.workflow_engine.filter_watchlist_by_regime')
    def test_run_daily_workflow_no_confluence(self, mock_regime_filter, mock_setup_scanner, mock_confluence_checker):
        """Test run_daily_workflow when confluence checker rejects all setups"""
        # Mock regime filter success
        mock_regime_filter.return_value = {
            'tradeable_symbols': ['BTC/USDT'],
            'filtered_symbols': ['ETH/USDT', 'SOL/USDT'],
            'regime_results': {},
            'summary': {'tradeable_count': 1, 'total_checked': 3}
        }
        
        # Mock setup scanner success
        mock_setup_scanner.return_value = {
            'setup_symbols': ['BTC/USDT'],
            'no_setup_symbols': [],
            'results': {},
            'summary': {'setups_count': 1, 'total_scanned': 1}
        }
        
        # Mock confluence checker rejection
        mock_confluence_checker.return_value = {
            'approved_symbols': [],
            'rejected_symbols': ['BTC/USDT'],
            'confluence_results': {},
            'summary': {'approved_count': 0, 'total_checked': 1}
        }
        
        result = self.workflow_engine.run_daily_workflow()
        
        # Should stop at confluence checker
        self.assertEqual(result['workflow_status'], 'NO_CONFLUENCE')
        self.assertIn('Multi-timeframe analysis rejected', result['message'])
        
        # Verify all three filters were called
        mock_regime_filter.assert_called_once()
        mock_setup_scanner.assert_called_once()
        mock_confluence_checker.assert_called_once()
    
    def test_create_workflow_result_structure(self):
        """Test _create_workflow_result method creates proper structure"""
        # Test indirectly through run_daily_workflow with open positions
        open_positions = [{'symbol': 'BTC/USDT', 'status': 'OPEN'}]
        
        result = self.workflow_engine.run_daily_workflow(open_positions)
        
        # Verify result structure
        expected_keys = ['workflow_status', 'message', 'start_time', 'end_time', 'duration_seconds']
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertIsInstance(result['duration_seconds'], (int, float))
        self.assertGreater(result['duration_seconds'], 0)
    
    def test_workflow_timing_tracking(self):
        """Test that workflow tracks execution timing"""
        # Use the simple case with open positions to test timing
        open_positions = [{'symbol': 'BTC/USDT'}]
        
        start_time = datetime.utcnow()
        result = self.workflow_engine.run_daily_workflow(open_positions)
        end_time = datetime.utcnow()
        
        # Verify timing information
        self.assertIn('start_time', result)
        self.assertIn('end_time', result)
        self.assertIn('duration_seconds', result)
        
        # Duration should be reasonable
        self.assertGreater(result['duration_seconds'], 0)
        self.assertLess(result['duration_seconds'], 60)  # Should complete within 60 seconds
        
        # Times should be in reasonable range
        result_start = datetime.fromisoformat(result['start_time'].replace('Z', '+00:00'))
        result_end = datetime.fromisoformat(result['end_time'].replace('Z', '+00:00'))
        
        self.assertGreaterEqual(result_start, start_time - pd.Timedelta(seconds=1))
        self.assertLessEqual(result_end, end_time + pd.Timedelta(seconds=1))
    
    def test_workflow_configuration_access(self):
        """Test that workflow properly accesses configuration sections"""
        # Test that filter configs are properly extracted
        self.assertEqual(
            self.workflow_engine.filter_configs['regime']['min_trend_distance'],
            0.02
        )
        self.assertEqual(
            self.workflow_engine.filter_configs['setup']['min_trend_bars'],
            10
        )
        self.assertEqual(
            self.workflow_engine.filter_configs['confluence']['min_confluence_count'],
            2
        )
        self.assertEqual(
            self.workflow_engine.filter_configs['risk']['max_account_risk'],
            0.01
        )
        self.assertEqual(
            self.workflow_engine.filter_configs['entry']['profit_target_ratio'],
            2.0
        )

if __name__ == '__main__':
    unittest.main()