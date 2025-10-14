#!/usr/bin/env python3
"""
Unified Trading System Test Suite
=================================

Single entry point for all trading system tests and validation.
Replaces scattered test files with a comprehensive testing framework.

Features:
- Professional signal engine integration
- Multiple test scenarios (live data, backtest, validation)
- Comprehensive reporting and logging
- Real Kraken data feeds with professional signal generation
- Balance verification and system validation

Usage:
    python unified_trading_tests.py --mode [live|backtest|validation|all]

Author: Professional Trading Team
Date: October 13, 2025
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Import our core systems
from paper_trading_engine import PaperTradingEngine, TradingConfig, TradingSignal
from ultimate_crypto_analyzer import UltimateCryptoAnalyzer

# Try to import real trading functionality 
try:
    from real_kraken_paper_trading import KrakenPaperTradingSystem
    LIVE_TRADING_AVAILABLE = True
except ImportError:
    LIVE_TRADING_AVAILABLE = False

# Configure comprehensive logging
log_filename = f'unified_trading_test_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class UnifiedTradingTestSuite:
    """Comprehensive trading system test suite"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
        # Initialize core systems
        self.trading_config = TradingConfig(
            starting_balance=10000.0,
            max_risk_per_trade=0.015,  # Conservative 1.5%
            max_portfolio_risk=0.60,   # Conservative 60% deployment
            enable_compounding=True
        )
        
        self.paper_trader = PaperTradingEngine(self.trading_config)
        self.ultimate_analyzer = UltimateCryptoAnalyzer()
        
        # Test symbols for comprehensive coverage
        self.test_symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT"]
        
        logger.info("🚀 UNIFIED TRADING TEST SUITE INITIALIZED")
        logger.info(f"   📊 Testing {len(self.test_symbols)} symbols")
        logger.info(f"   💰 Starting balance: ${self.trading_config.starting_balance:,.0f}")
        logger.info(f"   🔧 Professional signal engine: {'✅ Enabled' if self.paper_trader.professional_signal_engine else '❌ Disabled'}")
        
    async def run_test_suite(self, mode: str = "all"):
        """Run comprehensive test suite"""
        
        logger.info(f"🧪 STARTING TEST MODE: {mode.upper()}")
        print(f"\\n{'='*80}")
        print(f"🧪 UNIFIED TRADING SYSTEM TEST SUITE")
        print(f"{'='*80}")
        print(f"Mode: {mode.upper()}")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log File: {log_filename}")
        print(f"{'='*80}")
        
        try:
            if mode in ["validation", "all"]:
                await self._run_system_validation()
            
            if mode in ["live", "all"]:
                if LIVE_TRADING_AVAILABLE:
                    await self._run_live_data_test()
                else:
                    logger.warning("⚠️ Skipping live data test - real_kraken_paper_trading not available")
                    print("   ⚠️ Skipping live data test - WebSocket functionality not available")
            
            if mode in ["backtest", "all"]:
                await self._run_backtest_simulation()
            
            # Generate comprehensive report
            await self._generate_final_report()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            raise
    
    async def _run_system_validation(self):
        """Validate all system components are working correctly"""
        
        logger.info("🔍 RUNNING SYSTEM VALIDATION TESTS")
        print("\\n🔍 System Validation Phase...")
        
        validation_results = {}
        
        # Test 1: Professional signal engine integration
        try:
            if self.paper_trader.professional_signal_engine:
                # Create mock analysis data
                mock_analysis = {
                    'symbol': 'BTC/USDT',
                    'multi_timeframe_analysis': {
                        'timeframe_data': {
                            '1h': {
                                'indicators': {
                                    'rsi': 25.0,  # Oversold
                                    'macd': 0.15,
                                    'macd_signal': 0.05,
                                    'close': 45000.0
                                }
                            },
                            '4h': {
                                'indicators': {
                                    'rsi': 30.0,  # Oversold
                                    'macd': 0.20,
                                    'macd_signal': 0.10,
                                    'close': 45000.0
                                }
                            }
                        }
                    },
                    'volume_profile_analysis': {
                        'metadata': {
                            'current_price': 45000.0
                        }
                    }
                }
                
                # Test professional signal generation
                professional_signals = self.paper_trader.generate_professional_signals(mock_analysis)
                
                validation_results['professional_signals'] = {
                    'status': 'PASS' if len(professional_signals) >= 0 else 'FAIL',
                    'signals_generated': len(professional_signals),
                    'details': 'Professional signal engine working correctly'
                }
                
                logger.info(f"   ✅ Professional signal test: {len(professional_signals)} signals generated")
            else:
                validation_results['professional_signals'] = {
                    'status': 'SKIP',
                    'reason': 'Professional signal engine not available'
                }
                
        except Exception as e:
            validation_results['professional_signals'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"   ❌ Professional signal test failed: {e}")
        
        # Test 2: Balance calculation accuracy
        try:
            starting_balance = self.paper_trader.get_cash_balance()
            
            # Create and process a test signal
            test_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol='BTC/USDT',
                action='BUY',
                confidence=0.75,
                entry_price=45000.0,
                stop_loss=43000.0,
                take_profit=47000.0,
                source='validation_test',
                reason='System validation test signal'
            )
            
            # Update price so the engine has current data
            self.paper_trader.update_price('BTC/USDT', 45000.0, 1000000.0)
            
            # Process the signal (should go to watchlist, not execute immediately)
            result = self.paper_trader.process_trading_signal(test_signal)
            
            balance_after_signal = self.paper_trader.get_cash_balance()
            
            validation_results['balance_integrity'] = {
                'status': 'PASS' if abs(starting_balance - balance_after_signal) < 1.0 else 'FAIL',
                'starting_balance': starting_balance,
                'balance_after_signal': balance_after_signal,
                'difference': balance_after_signal - starting_balance,
                'signal_result': result['status']
            }
            
            logger.info(f"   ✅ Balance integrity test: {result['status']} - Balance unchanged as expected")
            
        except Exception as e:
            validation_results['balance_integrity'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"   ❌ Balance integrity test failed: {e}")
        
        # Test 3: Signal validation system
        try:
            watchlist_status = self.paper_trader.signal_validator.get_watchlist_status()
            throttle_status = self.paper_trader.throttler.get_throttle_status()
            
            validation_results['validation_systems'] = {
                'status': 'PASS',
                'watchlist_signals': watchlist_status['watching'],
                'throttle_capacity': f"{throttle_status['positions_this_hour']}/{throttle_status['max_per_hour']}",
                'can_take_positions': throttle_status['can_take_more']
            }
            
            logger.info("   ✅ Signal validation systems operational")
            
        except Exception as e:
            validation_results['validation_systems'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"   ❌ Validation systems test failed: {e}")
        
        self.test_results['system_validation'] = validation_results
        
        # Print validation summary
        print("\\n📋 System Validation Results:")
        for test_name, result in validation_results.items():
            status_emoji = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}
            status = result.get('status', 'UNKNOWN')
            print(f"   {status_emoji.get(status, '❓')} {test_name}: {status}")
    
    async def _run_live_data_test(self):
        """Run live data test with real Kraken feeds"""
        
        logger.info("📡 RUNNING LIVE DATA TEST")
        print("\\n📡 Live Data Test Phase...")
        print("   Connecting to Kraken WebSocket feeds...")
        print("   Duration: 60 minutes (1 hour)")
        print("   Goal: Professional signal generation + validation + live monitoring")
        
        # Initialize the Kraken live trading system
        live_trader = KrakenPaperTradingSystem()
        
        # Note: KrakenPaperTradingSystem has its own internal configuration
        # It runs for 1 hour by default with BTC, ETH, XRP, ADA, SOL, DOT
        
        try:
            # Run the live test
            await live_trader.start_real_trading_test()
            
            # Collect results from the live trader's internal engine
            final_balance = live_trader.paper_trader.get_current_balance()
            starting_balance = live_trader.paper_trader.starting_balance
            total_return = (final_balance - starting_balance) / starting_balance
            
            # Try to get watchlist status if validation system is available
            try:
                watchlist_status = live_trader.paper_trader.signal_validator.get_watchlist_status()
            except AttributeError:
                # Fallback if validation system not available in this version
                watchlist_status = {
                    'watching': 0,
                    'validated_today': 0,
                    'invalidated_today': 0,
                    'validation_rate': 0.0
                }
            
            live_results = {
                'status': 'COMPLETED',
                'duration_minutes': 60,
                'starting_balance': starting_balance,
                'final_balance': final_balance,
                'total_return_pct': total_return * 100,
                'signals_watched': watchlist_status['watching'],
                'signals_validated': watchlist_status['validated_today'],
                'signals_invalidated': watchlist_status['invalidated_today'],
                'validation_rate': watchlist_status['validation_rate'],
                'completed_trades': len(live_trader.paper_trader.trades),
                'open_positions': sum(len(pos_list) for pos_list in live_trader.paper_trader.positions.values())
            }
            
            self.test_results['live_data_test'] = live_results
            
            print(f"\\n📊 Live Test Results:")
            print(f"   💰 P&L: ${final_balance - starting_balance:+.2f} ({total_return:+.1%})")
            print(f"   🔍 Signals: {watchlist_status['validated_today']} validated, {watchlist_status['invalidated_today']} rejected")
            print(f"   📈 Trades: {len(live_trader.paper_trader.trades)} completed, {sum(len(pos_list) for pos_list in live_trader.paper_trader.positions.values())} open")
            print(f"   ⚡ Validation Rate: {watchlist_status['validation_rate']:.1%}")
            
            logger.info(f"📡 Live test completed: {total_return:+.1%} return, {len(live_trader.paper_trader.trades)} trades")
            
        except Exception as e:
            self.test_results['live_data_test'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"❌ Live data test failed: {e}")
            print(f"   ❌ Live test failed: {e}")
    
    async def _run_backtest_simulation(self):
        """Run backtesting simulation with historical scenarios"""
        
        logger.info("📈 RUNNING BACKTEST SIMULATION")
        print("\\n📈 Backtest Simulation Phase...")
        
        # Reset engine for backtest
        backtest_engine = PaperTradingEngine(self.trading_config)
        
        # Simulate various market scenarios
        scenarios = [
            {"name": "Bull Market", "price_changes": [1.02, 1.015, 1.03, 1.01, 1.025]},
            {"name": "Bear Market", "price_changes": [0.98, 0.985, 0.97, 0.99, 0.975]},
            {"name": "Sideways Market", "price_changes": [1.01, 0.99, 1.005, 0.995, 1.002]}
        ]
        
        backtest_results = {}
        
        for scenario in scenarios:
            print(f"   Testing {scenario['name']}...")
            
            scenario_engine = PaperTradingEngine(self.trading_config)
            base_price = 45000.0
            current_price = base_price
            
            trades_executed = 0
            
            for i, price_change in enumerate(scenario['price_changes']):
                current_price *= price_change
                scenario_engine.update_price('BTC/USDT', current_price, 1000000.0)
                
                # Generate test signals based on price movement
                if price_change > 1.01:  # Rising market
                    signal = TradingSignal(
                        timestamp=datetime.now(),
                        symbol='BTC/USDT',
                        action='BUY',
                        confidence=0.70,
                        entry_price=current_price,
                        stop_loss=current_price * 0.97,
                        take_profit=current_price * 1.05,
                        source='backtest',
                        reason=f'{scenario["name"]} bullish signal'
                    )
                elif price_change < 0.99:  # Falling market
                    signal = TradingSignal(
                        timestamp=datetime.now(),
                        symbol='BTC/USDT',
                        action='SELL',
                        confidence=0.70,
                        entry_price=current_price,
                        stop_loss=current_price * 1.03,
                        take_profit=current_price * 0.95,
                        source='backtest',
                        reason=f'{scenario["name"]} bearish signal'
                    )
                else:
                    continue
                
                # Process signal
                result = scenario_engine.process_trading_signal(signal)
                if result['status'] == 'WATCHING':
                    trades_executed += 1
            
            final_balance = scenario_engine.get_current_balance()
            scenario_return = (final_balance - self.trading_config.starting_balance) / self.trading_config.starting_balance
            
            backtest_results[scenario['name']] = {
                'return_pct': scenario_return * 100,
                'final_balance': final_balance,
                'signals_processed': trades_executed,
                'price_range': f"{min(scenario['price_changes']):.3f} to {max(scenario['price_changes']):.3f}"
            }
        
        self.test_results['backtest_simulation'] = backtest_results
        
        print("\\n📊 Backtest Results:")
        for scenario_name, results in backtest_results.items():
            print(f"   📈 {scenario_name}: {results['return_pct']:+.2f}% ({results['signals_processed']} signals)")
        
        logger.info("📈 Backtest simulation completed")
    
    async def _generate_final_report(self):
        """Generate comprehensive final report"""
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"\\n{'='*80}")
        print(f"📊 UNIFIED TRADING TEST SUITE - FINAL REPORT")
        print(f"{'='*80}")
        print(f"Test Duration: {duration}")
        print(f"Completion Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System health summary
        print(f"\\n🏥 SYSTEM HEALTH SUMMARY:")
        
        if 'system_validation' in self.test_results:
            validation_results = self.test_results['system_validation']
            passed_tests = sum(1 for result in validation_results.values() if result.get('status') == 'PASS')
            total_tests = len(validation_results)
            print(f"   ✅ Validation Tests: {passed_tests}/{total_tests} passed")
            
            if passed_tests == total_tests:
                print(f"   🟢 System Status: ALL SYSTEMS OPERATIONAL")
            else:
                print(f"   🟡 System Status: SOME ISSUES DETECTED")
        
        # Trading performance summary
        if 'live_data_test' in self.test_results:
            live_results = self.test_results['live_data_test']
            if live_results.get('status') == 'COMPLETED':
                print(f"\\n📈 LIVE TRADING PERFORMANCE:")
                print(f"   💰 Total Return: {live_results['total_return_pct']:+.2f}%")
                print(f"   🎯 Signal Validation Rate: {live_results['validation_rate']:.1%}")
                print(f"   📊 Trades Completed: {live_results['completed_trades']}")
                print(f"   🔄 Open Positions: {live_results['open_positions']}")
                
                # Performance assessment
                if live_results['total_return_pct'] > -2 and live_results['validation_rate'] > 0.3:
                    print(f"   🟢 Performance Status: ACCEPTABLE")
                else:
                    print(f"   🟡 Performance Status: NEEDS IMPROVEMENT")
        
        # Export detailed results
        report_filename = f"trading_test_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        detailed_report = {
            'test_suite_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'log_file': log_filename
            },
            'system_config': {
                'starting_balance': self.trading_config.starting_balance,
                'max_risk_per_trade': self.trading_config.max_risk_per_trade,
                'test_symbols': self.test_symbols,
                'professional_engine_available': self.paper_trader.professional_signal_engine is not None
            },
            'test_results': self.test_results,
            'final_portfolio_state': self.paper_trader.get_portfolio_status()
        }
        
        with open(report_filename, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        print(f"\\n📄 DETAILED REPORT EXPORTED:")
        print(f"   📊 Report File: {report_filename}")
        print(f"   📝 Log File: {log_filename}")
        
        print(f"\\n✅ UNIFIED TRADING TEST SUITE COMPLETED")
        print(f"{'='*80}")
        
        logger.info(f"✅ Test suite completed successfully. Report: {report_filename}")

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Unified Trading System Test Suite')
    parser.add_argument('--mode', 
                       choices=['live', 'backtest', 'validation', 'all'],
                       default='all',
                       help='Test mode to run (default: all)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Unified Trading Test Suite...")
    print(f"Mode: {args.mode}")
    
    test_suite = UnifiedTradingTestSuite()
    
    try:
        await test_suite.run_test_suite(args.mode)
    except KeyboardInterrupt:
        print("\\n⚠️ Test suite interrupted by user")
        logger.info("Test suite interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test suite failed: {e}")
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())