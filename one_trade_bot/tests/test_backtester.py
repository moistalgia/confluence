"""
Comprehensive tests for the HistoricalBacktester class and related components.

Tests cover:
- BacktestTrade and BacktestResults dataclass validation  
- HistoricalBacktester initialization and configuration
- Backtest execution with realistic scenarios
- Trade simulation and outcome modeling
- Performance metrics calculation
- Filter effectiveness tracking
- Slippage and commission modeling
- Results serialization and logging
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from dataclasses import asdict
import numpy as np
import json
import tempfile
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.backtester import HistoricalBacktester, BacktestTrade, BacktestResults
from core.workflow_engine import DailyWorkflowEngine
from core.data_provider import DataProvider


class TestBacktestTrade(unittest.TestCase):
    """Test BacktestTrade dataclass functionality"""
    
    def setUp(self):
        """Set up test trade data"""
        self.sample_trade = BacktestTrade(
            trade_id=1,
            date='2024-01-15',
            symbol='BTC-USDT',
            direction='LONG',
            entry_price=42500.0,
            entry_time='2024-01-15 09:00:00',
            position_size=0.1,
            exit_price=44000.0,
            exit_time='2024-01-17 14:30:00',
            exit_reason='PROFIT_TARGET',
            gross_pnl=150.0,
            net_pnl=140.0,
            pnl_pct=0.035,
            risk_amount=425.0,
            regime_score=3,
            setup_quality=4,
            confluence_score=2,
            risk_score=1,
            market_volatility=0.025,
            entry_delay_minutes=45
        )
    
    def test_trade_creation(self):
        """Test BacktestTrade object creation"""
        trade = self.sample_trade
        
        self.assertEqual(trade.trade_id, 1)
        self.assertEqual(trade.symbol, 'BTC-USDT')
        self.assertEqual(trade.direction, 'LONG')
        self.assertEqual(trade.entry_price, 42500.0)
        self.assertEqual(trade.net_pnl, 140.0)
        self.assertEqual(trade.exit_reason, 'PROFIT_TARGET')
    
    def test_trade_to_dict(self):
        """Test BacktestTrade serialization to dictionary"""
        trade_dict = self.sample_trade.to_dict()
        
        self.assertIsInstance(trade_dict, dict)
        self.assertEqual(trade_dict['symbol'], 'BTC-USDT')
        self.assertEqual(trade_dict['net_pnl'], 140.0)
        self.assertEqual(trade_dict['pnl_pct'], 0.035)
        
        # Verify all required fields are present
        required_fields = [
            'trade_id', 'date', 'symbol', 'direction', 'entry_price',
            'exit_price', 'net_pnl', 'pnl_pct', 'risk_amount'
        ]
        for field in required_fields:
            self.assertIn(field, trade_dict)
    
    def test_losing_trade(self):
        """Test BacktestTrade for losing trade"""
        losing_trade = BacktestTrade(
            trade_id=2,
            date='2024-01-16',
            symbol='ETH-USDT',
            direction='SHORT',
            entry_price=2500.0,
            entry_time='2024-01-16 10:15:00',
            position_size=2.0,
            exit_price=2600.0,
            exit_time='2024-01-16 15:45:00',
            exit_reason='STOP_LOSS',
            gross_pnl=-200.0,
            net_pnl=-210.0,
            pnl_pct=-0.04,
            risk_amount=100.0,
            regime_score=2,
            setup_quality=3,
            confluence_score=1,
            risk_score=1,
            market_volatility=0.03,
            entry_delay_minutes=15
        )
        
        self.assertTrue(losing_trade.net_pnl < 0)
        self.assertEqual(losing_trade.exit_reason, 'STOP_LOSS')
        self.assertEqual(losing_trade.direction, 'SHORT')


class TestBacktestResults(unittest.TestCase):
    """Test BacktestResults dataclass functionality"""
    
    def setUp(self):
        """Set up test results data"""
        self.sample_trades = [
            BacktestTrade(
                trade_id=1, date='2024-01-15', symbol='BTC-USDT', direction='LONG',
                entry_price=42500.0, entry_time='2024-01-15 09:00:00', position_size=0.1,
                exit_price=44000.0, exit_time='2024-01-17 14:30:00', exit_reason='PROFIT_TARGET',
                gross_pnl=150.0, net_pnl=140.0, pnl_pct=0.035, risk_amount=425.0,
                regime_score=3, setup_quality=4, confluence_score=2, risk_score=1,
                market_volatility=0.025, entry_delay_minutes=45
            ),
            BacktestTrade(
                trade_id=2, date='2024-01-20', symbol='ETH-USDT', direction='SHORT',
                entry_price=2500.0, entry_time='2024-01-20 11:00:00', position_size=2.0,
                exit_price=2600.0, exit_time='2024-01-20 16:30:00', exit_reason='STOP_LOSS',
                gross_pnl=-200.0, net_pnl=-210.0, pnl_pct=-0.04, risk_amount=100.0,
                regime_score=2, setup_quality=2, confluence_score=1, risk_score=1,
                market_volatility=0.03, entry_delay_minutes=30
            )
        ]
        
        self.equity_curve = [
            {'date': '2024-01-01', 'capital': 10000, 'drawdown': 0.0},
            {'date': '2024-01-15', 'capital': 10140, 'drawdown': 0.0},
            {'date': '2024-01-20', 'capital': 9930, 'drawdown': 0.021}
        ]
    
    def test_results_creation(self):
        """Test BacktestResults object creation"""
        results = BacktestResults(
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_capital=10000.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            win_rate=0.5,
            total_return=-70.0,
            total_return_pct=-0.007,
            max_drawdown=210.0,
            max_drawdown_pct=0.021,
            sharpe_ratio=0.15,
            calmar_ratio=-0.33,
            avg_risk_per_trade=262.5,
            max_risk_per_trade=425.0,
            avg_win=140.0,
            avg_loss=-210.0,
            profit_factor=0.67,
            regime_filter_rate=0.85,
            setup_filter_rate=0.75,
            confluence_filter_rate=0.90,
            risk_filter_rate=0.95,
            avg_trades_per_month=2.0,
            longest_dry_spell_days=15,
            equity_curve=self.equity_curve,
            trades=self.sample_trades
        )
        
        self.assertEqual(results.total_trades, 2)
        self.assertEqual(results.win_rate, 0.5)
        self.assertEqual(results.total_return, -70.0)
        self.assertEqual(len(results.trades), 2)
        self.assertEqual(len(results.equity_curve), 3)


class TestHistoricalBacktester(unittest.TestCase):
    """Test HistoricalBacktester class functionality"""
    
    def setUp(self):
        """Set up test configuration and mocks"""
        self.test_config = {
            'account': {'balance': 10000.0},
            'risk': {'max_risk_per_trade': 0.01},
            'watchlist': ['BTC-USDT', 'ETH-USDT', 'ADA-USDT'],
            'daily_scan_time': '09:00',
            'backtest': {
                'slippage_bps': 5,
                'commission_pct': 0.001,
                'max_entry_delay_hours': 2
            }
        }
        
        # Create mock data provider
        self.mock_data_provider = Mock(spec=DataProvider)
        
        # Mock OHLCV data
        self.mock_ohlcv_data = {
            'open': [42000, 42500, 43000],
            'high': [42800, 43200, 44100],
            'low': [41900, 42300, 42800],
            'close': [42500, 43000, 44000],
            'volume': [1000000, 1200000, 950000]
        }
        
        self.mock_data_provider.get_ohlcv.return_value = self.mock_ohlcv_data
        
    def test_backtester_initialization(self):
        """Test HistoricalBacktester initialization"""
        backtester = HistoricalBacktester(self.test_config)
        
        self.assertEqual(backtester.config, self.test_config)
        self.assertEqual(backtester.initial_capital, 10000.0)
        self.assertEqual(backtester.slippage_bps, 5)
        self.assertEqual(backtester.commission_pct, 0.001)
        self.assertEqual(backtester.max_entry_delay, 2)
        self.assertIsNone(backtester.results)
    
    def test_trading_days_generation(self):
        """Test _generate_trading_days method"""
        backtester = HistoricalBacktester(self.test_config)
        
        trading_days = backtester._generate_trading_days('2024-01-01', '2024-01-05')
        
        expected_days = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        self.assertEqual(trading_days, expected_days)
        self.assertEqual(len(trading_days), 5)
    
    def test_trading_days_single_day(self):
        """Test trading days generation for single day"""
        backtester = HistoricalBacktester(self.test_config)
        
        trading_days = backtester._generate_trading_days('2024-01-15', '2024-01-15')
        
        self.assertEqual(trading_days, ['2024-01-15'])
        self.assertEqual(len(trading_days), 1)
    
    @patch('backtest.backtester.DailyWorkflowEngine')
    def test_daily_workflow_simulation_no_trade(self, mock_engine_class):
        """Test daily workflow simulation when no trade is executed"""
        # Setup mock workflow engine
        mock_engine = Mock()
        mock_engine.run_daily_workflow.return_value = {
            'workflow_status': 'NO_SETUP',
            'filter_results': {
                'regime': {'summary': {'total_checked': 3, 'tradeable_count': 2}},
                'setup': {'summary': {'total_scanned': 3, 'setups_count': 0}}
            }
        }
        mock_engine_class.return_value = mock_engine
        
        backtester = HistoricalBacktester(self.test_config)
        backtester.data_provider = self.mock_data_provider
        
        filter_stats = {
            'regime_total': 0, 'regime_passed': 0,
            'setup_total': 0, 'setup_passed': 0,
            'confluence_total': 0, 'confluence_passed': 0,
            'risk_total': 0, 'risk_passed': 0
        }
        
        result = backtester._simulate_daily_workflow(
            '2024-01-15', ['BTC-USDT'], 10000.0, filter_stats
        )
        
        self.assertFalse(result['trade_executed'])
        self.assertIsNone(result['trade'])
        self.assertEqual(result['workflow_result']['workflow_status'], 'NO_SETUP')
        
        # Verify filter stats were updated
        self.assertEqual(filter_stats['regime_total'], 3)
        self.assertEqual(filter_stats['regime_passed'], 2)
    
    @patch('backtest.backtester.DailyWorkflowEngine')
    def test_daily_workflow_simulation_with_trade(self, mock_engine_class):
        """Test daily workflow simulation when trade is executed"""
        # Setup mock workflow engine
        mock_engine = Mock()
        mock_engine.run_daily_workflow.return_value = {
            'workflow_status': 'TRADE_READY',
            'the_one_trade': 'BTC-USDT',
            'trade_direction': 'LONG',
            'entry_plan': {
                'entry_price': 42500.0,
                'position_size': 0.1,
                'stop_loss_price': 41000.0,
                'profit_target_price': 45000.0,
                'risk_amount': 150.0
            },
            'filter_results': {
                'regime': {'summary': {'total_checked': 3, 'tradeable_count': 3}},
                'setup': {'summary': {'total_scanned': 3, 'setups_count': 1}},
                'confluence': {'summary': {'total_checked': 1, 'approved_count': 1}},
                'risk': {'summary': {'total_checked': 1, 'final_approved_count': 1}}
            }
        }
        mock_engine_class.return_value = mock_engine
        
        backtester = HistoricalBacktester(self.test_config)
        backtester.data_provider = self.mock_data_provider
        
        filter_stats = {
            'regime_total': 0, 'regime_passed': 0,
            'setup_total': 0, 'setup_passed': 0,
            'confluence_total': 0, 'confluence_passed': 0,
            'risk_total': 0, 'risk_passed': 0
        }
        
        with patch.object(backtester, '_simulate_trade_execution') as mock_trade_exec:
            mock_trade = BacktestTrade(
                trade_id=1, date='2024-01-15', symbol='BTC-USDT', direction='LONG',
                entry_price=42525.0, entry_time='2024-01-15 09:30:00', position_size=0.1,
                exit_price=44000.0, exit_time='2024-01-17 14:00:00', exit_reason='PROFIT_TARGET',
                gross_pnl=147.5, net_pnl=138.5, pnl_pct=0.0325, risk_amount=150.0,
                regime_score=3, setup_quality=1, confluence_score=1, risk_score=1,
                market_volatility=0.025, entry_delay_minutes=30
            )
            mock_trade_exec.return_value = mock_trade
            
            result = backtester._simulate_daily_workflow(
                '2024-01-15', ['BTC-USDT'], 10000.0, filter_stats
            )
            
            self.assertTrue(result['trade_executed'])
            self.assertEqual(result['trade'].symbol, 'BTC-USDT')
            self.assertEqual(result['trade'].direction, 'LONG')
            self.assertEqual(result['workflow_result']['workflow_status'], 'TRADE_READY')
    
    def test_trade_execution_simulation_long(self):
        """Test trade execution simulation for LONG position"""
        backtester = HistoricalBacktester(self.test_config)
        
        workflow_result = {
            'the_one_trade': 'BTC-USDT',
            'trade_direction': 'LONG',
            'entry_plan': {
                'entry_price': 42500.0,
                'position_size': 0.1,
                'stop_loss_price': 41000.0,
                'profit_target_price': 45000.0,
                'risk_amount': 150.0
            },
            'filter_results': {
                'regime': {'summary': {'tradeable_count': 3}},
                'setup': {'summary': {'setups_count': 1}},
                'confluence': {'summary': {'approved_count': 1}},
                'risk': {'summary': {'final_approved_count': 1}}
            }
        }
        
        with patch.object(backtester, '_simulate_trade_outcome') as mock_outcome:
            mock_outcome.return_value = {
                'exit_price': 44000.0,
                'exit_time': '2024-01-17 14:00:00',
                'exit_reason': 'PROFIT_TARGET'
            }
            
            trade = backtester._simulate_trade_execution(workflow_result, '2024-01-15', 1)
            
            # Verify trade properties
            self.assertEqual(trade.symbol, 'BTC-USDT')
            self.assertEqual(trade.direction, 'LONG')
            self.assertEqual(trade.trade_id, 1)
            self.assertEqual(trade.date, '2024-01-15')
            self.assertGreater(trade.entry_price, 42500.0)  # Should have slippage
            self.assertEqual(trade.exit_price, 44000.0)
            self.assertGreater(trade.net_pnl, 0)  # Should be profitable
    
    def test_trade_execution_simulation_short(self):
        """Test trade execution simulation for SHORT position"""
        backtester = HistoricalBacktester(self.test_config)
        
        workflow_result = {
            'the_one_trade': 'ETH-USDT',
            'trade_direction': 'SHORT',
            'entry_plan': {
                'entry_price': 2500.0,
                'position_size': 2.0,
                'stop_loss_price': 2600.0,
                'profit_target_price': 2300.0,
                'risk_amount': 200.0
            },
            'filter_results': {
                'regime': {'summary': {'tradeable_count': 2}},
                'setup': {'summary': {'setups_count': 1}},
                'confluence': {'summary': {'approved_count': 1}},
                'risk': {'summary': {'final_approved_count': 1}}
            }
        }
        
        with patch.object(backtester, '_simulate_trade_outcome') as mock_outcome:
            mock_outcome.return_value = {
                'exit_price': 2350.0,
                'exit_time': '2024-01-16 11:30:00',
                'exit_reason': 'PROFIT_TARGET'
            }
            
            trade = backtester._simulate_trade_execution(workflow_result, '2024-01-15', 2)
            
            # Verify trade properties
            self.assertEqual(trade.symbol, 'ETH-USDT')
            self.assertEqual(trade.direction, 'SHORT')
            self.assertEqual(trade.trade_id, 2)
            self.assertLess(trade.entry_price, 2500.0)  # Should have slippage (get less for short)
            self.assertEqual(trade.exit_price, 2350.0)
            self.assertGreater(trade.net_pnl, 0)  # Should be profitable
    
    def test_trade_outcome_simulation_winner(self):
        """Test trade outcome simulation for winning trade"""
        backtester = HistoricalBacktester(self.test_config)
        
        with patch('numpy.random.random', return_value=0.3):  # Force win (< 0.65)
            with patch('numpy.random.uniform') as mock_uniform:
                mock_uniform.side_effect = [0.9, 2]  # target_hit_ratio, hold_days
                
                outcome = backtester._simulate_trade_outcome(
                    'BTC-USDT', 'LONG', 42500.0, 41000.0, 45000.0, '2024-01-15'
                )
                
                self.assertEqual(outcome['exit_reason'], 'PROFIT_TARGET')
                self.assertGreater(outcome['exit_price'], 42500.0)  # Should be above entry
                self.assertIn('2024-01-', outcome['exit_time'])
    
    def test_trade_outcome_simulation_loser(self):
        """Test trade outcome simulation for losing trade"""
        backtester = HistoricalBacktester(self.test_config)
        
        with patch('numpy.random.random', return_value=0.8):  # Force loss (> 0.65)
            with patch('numpy.random.uniform') as mock_uniform:
                mock_uniform.side_effect = [1.02, 1]  # stop_hit_ratio, hold_days
                with patch('numpy.random.randint', return_value=1):
                    
                    outcome = backtester._simulate_trade_outcome(
                        'BTC-USDT', 'LONG', 42500.0, 41000.0, 45000.0, '2024-01-15'
                    )
                    
                    self.assertEqual(outcome['exit_reason'], 'STOP_LOSS')
                    self.assertLess(outcome['exit_price'], 42500.0)  # Should be below entry
                    self.assertIn('2024-01-', outcome['exit_time'])
    
    def test_filter_stats_update(self):
        """Test filter statistics updating"""
        backtester = HistoricalBacktester(self.test_config)
        
        workflow_result = {
            'filter_results': {
                'regime': {'summary': {'total_checked': 5, 'tradeable_count': 3}},
                'setup': {'summary': {'total_scanned': 3, 'setups_count': 1}},
                'confluence': {'summary': {'total_checked': 1, 'approved_count': 1}},
                'risk': {'summary': {'total_checked': 1, 'final_approved_count': 1}}
            }
        }
        
        filter_stats = {
            'regime_total': 10, 'regime_passed': 6,
            'setup_total': 8, 'setup_passed': 2,
            'confluence_total': 3, 'confluence_passed': 1,
            'risk_total': 2, 'risk_passed': 1
        }
        
        backtester._update_filter_stats(workflow_result, filter_stats)
        
        # Verify stats were updated correctly
        self.assertEqual(filter_stats['regime_total'], 15)
        self.assertEqual(filter_stats['regime_passed'], 9)
        self.assertEqual(filter_stats['setup_total'], 11)
        self.assertEqual(filter_stats['setup_passed'], 3)
        self.assertEqual(filter_stats['confluence_total'], 4)
        self.assertEqual(filter_stats['confluence_passed'], 2)
        self.assertEqual(filter_stats['risk_total'], 3)
        self.assertEqual(filter_stats['risk_passed'], 2)
    
    def test_drawdown_calculation(self):
        """Test current drawdown calculation"""
        backtester = HistoricalBacktester(self.test_config)
        
        equity_curve = [
            {'date': '2024-01-01', 'capital': 10000},
            {'date': '2024-01-02', 'capital': 10500},
            {'date': '2024-01-03', 'capital': 9800}
        ]
        
        # Current capital below peak
        drawdown = backtester._calculate_current_drawdown(equity_curve, 9500)
        expected_drawdown = (10500 - 9500) / 10500  # Peak was 10500
        self.assertAlmostEqual(drawdown, expected_drawdown, places=4)
        
        # Current capital at new peak
        drawdown = backtester._calculate_current_drawdown(equity_curve, 11000)
        self.assertEqual(drawdown, 0.0)
    
    def test_final_results_calculation(self):
        """Test comprehensive final results calculation"""
        backtester = HistoricalBacktester(self.test_config)
        
        # Create sample trades
        trades = [
            BacktestTrade(
                trade_id=1, date='2024-01-15', symbol='BTC-USDT', direction='LONG',
                entry_price=42500.0, entry_time='2024-01-15 09:00:00', position_size=0.1,
                exit_price=44000.0, exit_time='2024-01-17 14:30:00', exit_reason='PROFIT_TARGET',
                gross_pnl=150.0, net_pnl=140.0, pnl_pct=0.035, risk_amount=150.0,
                regime_score=3, setup_quality=4, confluence_score=2, risk_score=1,
                market_volatility=0.025, entry_delay_minutes=45
            ),
            BacktestTrade(
                trade_id=2, date='2024-01-20', symbol='ETH-USDT', direction='SHORT',
                entry_price=2500.0, entry_time='2024-01-20 11:00:00', position_size=2.0,
                exit_price=2600.0, exit_time='2024-01-20 16:30:00', exit_reason='STOP_LOSS',
                gross_pnl=-200.0, net_pnl=-210.0, pnl_pct=-0.04, risk_amount=100.0,
                regime_score=2, setup_quality=2, confluence_score=1, risk_score=1,
                market_volatility=0.03, entry_delay_minutes=30
            )
        ]
        
        equity_curve = [
            {'date': '2024-01-01', 'capital': 10000},
            {'date': '2024-01-15', 'capital': 10140},
            {'date': '2024-01-20', 'capital': 9930}
        ]
        
        filter_stats = {
            'regime_total': 100, 'regime_passed': 20,
            'setup_total': 20, 'setup_passed': 5,
            'confluence_total': 5, 'confluence_passed': 2,
            'risk_total': 2, 'risk_passed': 2
        }
        
        results = backtester._calculate_final_results(
            '2024-01-01', '2024-01-31', 10000.0, 9930.0,
            trades, equity_curve, filter_stats
        )
        
        # Verify basic metrics
        self.assertEqual(results.total_trades, 2)
        self.assertEqual(results.winning_trades, 1)
        self.assertEqual(results.losing_trades, 1)
        self.assertEqual(results.win_rate, 0.5)
        self.assertEqual(results.total_return, -70.0)
        self.assertAlmostEqual(results.total_return_pct, -0.007, places=4)
        
        # Verify filter rates
        self.assertAlmostEqual(results.regime_filter_rate, 0.8, places=2)
        self.assertAlmostEqual(results.setup_filter_rate, 0.75, places=2)
        self.assertAlmostEqual(results.confluence_filter_rate, 0.6, places=2)
        self.assertAlmostEqual(results.risk_filter_rate, 0.0, places=2)  # All passed
    
    @patch('backtest.backtester.logger')
    def test_backtest_summary_logging(self, mock_logger):
        """Test backtest summary logging"""
        backtester = HistoricalBacktester(self.test_config)
        
        # Create mock results
        backtester.results = BacktestResults(
            start_date='2024-01-01', end_date='2024-01-31', initial_capital=10000.0,
            total_trades=5, winning_trades=3, losing_trades=2, win_rate=0.6,
            total_return=250.0, total_return_pct=0.025, max_drawdown=150.0, max_drawdown_pct=0.015,
            sharpe_ratio=1.2, calmar_ratio=1.67, avg_risk_per_trade=150.0, max_risk_per_trade=200.0,
            avg_win=120.0, avg_loss=-80.0, profit_factor=1.5,
            regime_filter_rate=0.85, setup_filter_rate=0.75, confluence_filter_rate=0.90, risk_filter_rate=0.95,
            avg_trades_per_month=5.0, longest_dry_spell_days=7,
            equity_curve=[], trades=[]
        )
        
        backtester._log_backtest_summary()
        
        # Verify logging was called
        self.assertTrue(mock_logger.info.called)
        # Check that key metrics were logged
        logged_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        logged_text = ' '.join(logged_messages)
        
        self.assertIn('BACKTEST RESULTS', logged_text)
        self.assertIn('Total Trades: 5', logged_text)
        self.assertIn('Win Rate: 60.0%', logged_text)
        self.assertIn('Sharpe Ratio: 1.20', logged_text)
    
    def test_results_serialization(self):
        """Test saving backtest results to file"""
        backtester = HistoricalBacktester(self.test_config)
        
        # Create mock results
        trades = [BacktestTrade(
            trade_id=1, date='2024-01-15', symbol='BTC-USDT', direction='LONG',
            entry_price=42500.0, entry_time='2024-01-15 09:00:00', position_size=0.1,
            exit_price=44000.0, exit_time='2024-01-17 14:30:00', exit_reason='PROFIT_TARGET',
            gross_pnl=150.0, net_pnl=140.0, pnl_pct=0.035, risk_amount=150.0,
            regime_score=3, setup_quality=4, confluence_score=2, risk_score=1,
            market_volatility=0.025, entry_delay_minutes=45
        )]
        
        backtester.results = BacktestResults(
            start_date='2024-01-01', end_date='2024-01-31', initial_capital=10000.0,
            total_trades=1, winning_trades=1, losing_trades=0, win_rate=1.0,
            total_return=140.0, total_return_pct=0.014, max_drawdown=0.0, max_drawdown_pct=0.0,
            sharpe_ratio=2.5, calmar_ratio=0.0, avg_risk_per_trade=150.0, max_risk_per_trade=150.0,
            avg_win=140.0, avg_loss=0.0, profit_factor=float('inf'),
            regime_filter_rate=0.85, setup_filter_rate=0.75, confluence_filter_rate=0.90, risk_filter_rate=0.95,
            avg_trades_per_month=1.0, longest_dry_spell_days=0,
            equity_curve=[], trades=trades
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            backtester.save_results(temp_filename)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(temp_filename))
            
            with open(temp_filename, 'r') as f:
                loaded_data = json.load(f)
            
            # Verify key fields are present
            self.assertEqual(loaded_data['total_trades'], 1)
            self.assertEqual(loaded_data['win_rate'], 1.0)
            self.assertEqual(len(loaded_data['trades']), 1)
            self.assertEqual(loaded_data['trades'][0]['symbol'], 'BTC-USDT')
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


if __name__ == '__main__':
    unittest.main()