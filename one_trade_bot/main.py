#!/usr/bin/env python3
"""
One Good Trade Per Day Bot - Main Runner
🎯 Daily execution script for the complete 5-filter trading system

Modes:
- daily: Run single daily scan and execute if trade found
- continuous: Run continuous paper trading simulation  
- backtest: Run historical backtest on specified period
- status: Check current positions and account status

Usage:
    python main.py --mode daily                    # Single daily scan
    python main.py --mode continuous              # Continuous paper trading
    python main.py --mode backtest --days 365     # Historical backtest
    python main.py --mode status                  # Current status check
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import yaml

# Add the bot directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_provider import DataProvider
from core.workflow_engine import DailyWorkflowEngine
from core.position_manager import PositionManager
from backtest import PaperTradingSimulator, run_historical_backtest
from utils import TradeLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/one_trade_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OneTradeBotRunner:
    """
    Main orchestrator for One Good Trade Per Day bot
    Handles different execution modes and system coordination
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the bot runner
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize core components
        self.data_provider = DataProvider(self.config.get('data_provider', {}))
        self.position_manager = PositionManager(self.config.get('position_manager', {}))
        self.trade_logger = TradeLogger(self.config.get('trade_logger', {}))
        
        logger.info("🎯 ONE GOOD TRADE PER DAY BOT - Initialized")
        logger.info(f"   Config: {config_path}")
        logger.info(f"   Account Balance: ${self.config.get('account', {}).get('balance', 0):,.0f}")
        
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config from {self.config_path}: {str(e)}")
            sys.exit(1)
    
    async def run_daily_mode(self) -> Dict:
        """
        Execute single daily scan and trade if signal found
        
        Returns:
            Daily execution results
        """
        logger.info("🌅 DAILY MODE: Running single daily scan")
        logger.info("=" * 60)
        
        try:
            # Create workflow engine
            engine = DailyWorkflowEngine(self.config, self.data_provider)
            
            # Get current positions
            current_positions = []
            if self.position_manager.current_position:
                current_positions = [self.position_manager.current_position.to_dict()]
            
            # Run daily workflow
            workflow_result = engine.run_daily_workflow(current_positions)
            
            # Log daily scan
            scan_date = datetime.utcnow().strftime('%Y-%m-%d')
            self.trade_logger.log_daily_scan(scan_date, workflow_result)
            
            # Process workflow result
            execution_result = await self._process_workflow_result(workflow_result)
            
            # Final summary
            self._log_daily_summary(workflow_result, execution_result)
            
            return {
                'mode': 'daily',
                'scan_date': scan_date,
                'workflow_result': workflow_result,
                'execution_result': execution_result,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"🚨 Daily mode failed: {str(e)}")
            return {
                'mode': 'daily',
                'success': False,
                'error': str(e)
            }
    
    async def run_continuous_mode(self) -> None:
        """
        Run continuous paper trading simulation
        """
        logger.info("🔄 CONTINUOUS MODE: Starting paper trading simulation")
        logger.info("=" * 60)
        
        try:
            # Create paper trading simulator
            simulator = PaperTradingSimulator(self.config)
            
            # Start continuous simulation
            await simulator.start_simulation()
            
        except KeyboardInterrupt:
            logger.info("🛑 Continuous mode stopped by user")
        except Exception as e:
            logger.error(f"🚨 Continuous mode failed: {str(e)}")
    
    def run_backtest_mode(self, days: int = 365, save_results: bool = True) -> Dict:
        """
        Run historical backtest
        
        Args:
            days: Number of days to backtest
            save_results: Whether to save results to file
            
        Returns:
            Backtest results
        """
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        logger.info(f"📈 BACKTEST MODE: Testing {days} days ({start_date} to {end_date})")
        logger.info("=" * 60)
        
        try:
            # Prepare save file
            save_file = None
            if save_results:
                os.makedirs('backtest/results', exist_ok=True)
                save_file = f'backtest/results/backtest_{start_date}_to_{end_date}.json'
            
            # Run backtest
            results = run_historical_backtest(
                config_path=self.config_path,
                start_date=start_date,
                end_date=end_date,
                save_file=save_file
            )
            
            # Log key results
            logger.info("📊 BACKTEST COMPLETE:")
            logger.info(f"   Total Return: {results.total_return_pct:+.1%}")
            logger.info(f"   Win Rate: {results.win_rate:.1%}")
            logger.info(f"   Max Drawdown: {results.max_drawdown_pct:.1%}")
            logger.info(f"   Total Trades: {results.total_trades}")
            logger.info(f"   Avg Trades/Month: {results.avg_trades_per_month:.1f}")
            
            return {
                'mode': 'backtest',
                'success': True,
                'results': results,
                'save_file': save_file
            }
            
        except Exception as e:
            logger.error(f"🚨 Backtest failed: {str(e)}")
            return {
                'mode': 'backtest',
                'success': False,
                'error': str(e)
            }
    
    def run_status_mode(self) -> Dict:
        """
        Check current system status
        
        Returns:
            Status information
        """
        logger.info("📊 STATUS MODE: Checking current system status")
        logger.info("=" * 60)
        
        try:
            # Get account/position status
            position_summary = self.position_manager.get_position_summary()
            
            # Get recent performance  
            performance_30d = self.trade_logger.get_performance_summary(days=30)
            performance_all = self.trade_logger.get_performance_summary(days=0)
            
            # Log status
            logger.info("💰 ACCOUNT STATUS:")
            logger.info(f"   Balance: ${self.config.get('account', {}).get('balance', 0):,.0f}")
            
            if position_summary['has_position']:
                logger.info(f"\n📊 CURRENT POSITION:")
                logger.info(f"   {position_summary['summary']}")
                logger.info(f"   Unrealized P&L: ${position_summary['unrealized_pnl']:+,.0f}")
                logger.info(f"   Status: {position_summary['status']}")
            else:
                logger.info(f"\n📊 POSITION: None (ready for new trade)")
            
            if performance_30d['total_trades'] > 0:
                logger.info(f"\n📈 PERFORMANCE (30 DAYS):")
                logger.info(f"   Total Trades: {performance_30d['total_trades']}")
                logger.info(f"   Win Rate: {performance_30d['win_rate']:.1%}")
                logger.info(f"   Total P&L: ${performance_30d['total_pnl']:+,.0f}")
                logger.info(f"   Avg Win: ${performance_30d['avg_win']:+,.0f}")
                logger.info(f"   Avg Loss: ${performance_30d['avg_loss']:+,.0f}")
            
            if performance_all['total_trades'] > 0:
                logger.info(f"\n📈 PERFORMANCE (ALL TIME):")
                logger.info(f"   Total Trades: {performance_all['total_trades']}")
                logger.info(f"   Win Rate: {performance_all['win_rate']:.1%}")
                logger.info(f"   Total P&L: ${performance_all['total_pnl']:+,.0f}")
            
            return {
                'mode': 'status',
                'success': True,
                'position': position_summary,
                'performance_30d': performance_30d,
                'performance_all': performance_all
            }
            
        except Exception as e:
            logger.error(f"🚨 Status check failed: {str(e)}")
            return {
                'mode': 'status',
                'success': False,
                'error': str(e)
            }
    
    async def _process_workflow_result(self, workflow_result: Dict) -> Dict:
        """
        Process workflow result and execute trades if needed
        
        Returns:
            Execution result
        """
        workflow_status = workflow_result.get('workflow_status')
        
        if workflow_status == 'TRADE_READY':
            return await self._execute_trade_signal(workflow_result)
        elif workflow_status == 'POSITION_ALREADY_OPEN':
            return await self._manage_existing_position()
        else:
            # No trade today
            return {
                'action': 'NO_TRADE',
                'reason': workflow_result.get('message', 'No trade signal'),
                'success': True
            }
    
    async def _execute_trade_signal(self, workflow_result: Dict) -> Dict:
        """
        Execute trade signal from workflow
        
        Returns:
            Execution result
        """
        symbol = workflow_result.get('the_one_trade')
        direction = workflow_result.get('trade_direction')
        entry_plan = workflow_result.get('entry_plan', {})
        
        logger.info(f"🎯 EXECUTING TRADE SIGNAL: {symbol} {direction}")
        
        try:
            # Start trade logging
            trade_id = self.trade_logger.start_trade(
                symbol=symbol,
                direction=direction,
                entry_plan=entry_plan,
                workflow_result=workflow_result,
                account_balance=self.config.get('account', {}).get('balance', 0)
            )
            
            # In daily mode, we simulate immediate execution for demonstration
            # In production, this would place actual orders and monitor fills
            
            # Simulate order placement and fill (for demo)
            entry_price = entry_plan.get('entry_price')
            position_size = entry_plan.get('position_size')
            risk_amount = entry_plan.get('risk_amount')
            stop_price = entry_plan.get('stop_loss_price')
            
            # Open position in position manager
            position_result = self.position_manager.open_position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                position_size=position_size,
                risk_amount=risk_amount,
                account_balance=self.config.get('account', {}).get('balance', 0)
            )
            
            if position_result['success']:
                # Update trade logger with actual entry
                self.trade_logger.update_trade_entry(
                    trade_id=trade_id,
                    actual_entry_price=entry_price,
                    entry_time=datetime.utcnow().isoformat(),
                    entry_method='LIMIT'
                )
                
                logger.info(f"✅ TRADE EXECUTED: {trade_id}")
                
                return {
                    'action': 'TRADE_EXECUTED',
                    'trade_id': trade_id,
                    'position': position_result['position'],
                    'success': True
                }
            else:
                logger.error(f"❌ Trade execution failed: {position_result['error']}")
                return {
                    'action': 'EXECUTION_FAILED',
                    'error': position_result['error'],
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"🚨 Trade execution error: {str(e)}")
            return {
                'action': 'EXECUTION_ERROR',
                'error': str(e),
                'success': False
            }
    
    async def _manage_existing_position(self) -> Dict:
        """
        Manage existing open position
        
        Returns:
            Management result
        """
        if not self.position_manager.current_position:
            return {
                'action': 'NO_POSITION',
                'success': True
            }
        
        position = self.position_manager.current_position
        
        try:
            # Get current market price
            df = self.data_provider.get_ohlcv(position.symbol, '1h', days=1)
            current_price = df['close'].iloc[-1]
            
            # Update position
            update_result = self.position_manager.update_position(current_price, self.data_provider)
            
            if update_result['success']:
                exit_signal = update_result.get('exit_signal')
                
                if exit_signal:
                    # Close position
                    close_result = self.position_manager.close_position(
                        exit_signal['exit_price'], 
                        exit_signal['type']
                    )
                    
                    if close_result['success']:
                        # Log trade close
                        # Note: In a full implementation, we'd find the trade_id from the position
                        logger.info(f"🔒 Position closed: {exit_signal['reason']}")
                        logger.info(f"   P&L: ${close_result['realized_pnl']:+,.0f}")
                        
                        return {
                            'action': 'POSITION_CLOSED',
                            'exit_reason': exit_signal['type'],
                            'pnl': close_result['realized_pnl'],
                            'success': True
                        }
                else:
                    # Position still open, just monitoring
                    unrealized_pnl = update_result.get('unrealized_pnl', 0)
                    logger.info(f"📊 Position monitoring: {position.symbol} "
                               f"P&L: ${unrealized_pnl:+,.0f}")
                    
                    return {
                        'action': 'POSITION_MONITORED',
                        'unrealized_pnl': unrealized_pnl,
                        'success': True
                    }
            else:
                logger.error(f"❌ Position update failed: {update_result['error']}")
                return {
                    'action': 'UPDATE_FAILED',
                    'error': update_result['error'],
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"🚨 Position management error: {str(e)}")
            return {
                'action': 'MANAGEMENT_ERROR',
                'error': str(e),
                'success': False
            }
    
    def _log_daily_summary(self, workflow_result: Dict, execution_result: Dict) -> None:
        """Log daily execution summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 DAILY EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        workflow_status = workflow_result.get('workflow_status')
        execution_action = execution_result.get('action', 'UNKNOWN')
        
        logger.info(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info(f"Workflow Status: {workflow_status}")
        logger.info(f"Execution Action: {execution_action}")
        
        if workflow_status == 'TRADE_READY':
            symbol = workflow_result.get('the_one_trade', 'Unknown')
            direction = workflow_result.get('trade_direction', 'Unknown')
            logger.info(f"Trade Signal: {symbol} {direction}")
            
            if execution_result['success']:
                logger.info("✅ Execution: SUCCESS")
            else:
                logger.info("❌ Execution: FAILED")
                logger.info(f"   Error: {execution_result.get('error', 'Unknown')}")
        else:
            logger.info(f"Decision: {workflow_result.get('message', 'No details')}")
        
        # Position status
        position_summary = self.position_manager.get_position_summary()
        if position_summary['has_position']:
            logger.info(f"Current Position: {position_summary['summary']}")
        else:
            logger.info("Current Position: None")
        
        logger.info("=" * 60)


async def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='One Good Trade Per Day Bot - Main Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode daily                    # Single daily scan
  python main.py --mode continuous              # Continuous paper trading  
  python main.py --mode backtest --days 365     # 1-year backtest
  python main.py --mode status                  # Check current status
  python main.py --config custom.yaml --mode daily  # Use custom config
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['daily', 'continuous', 'backtest', 'status'],
        required=True,
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days for backtest mode (default: 365)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save backtest results to file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Initialize bot runner
        bot = OneTradeBotRunner(args.config)
        
        # Execute based on mode
        if args.mode == 'daily':
            result = await bot.run_daily_mode()
            
        elif args.mode == 'continuous':
            await bot.run_continuous_mode()
            result = {'mode': 'continuous', 'success': True}
            
        elif args.mode == 'backtest':
            result = bot.run_backtest_mode(
                days=args.days,
                save_results=not args.no_save
            )
            
        elif args.mode == 'status':
            result = bot.run_status_mode()
        
        # Exit with appropriate code
        if result.get('success', False):
            logger.info("🎯 Bot execution completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Bot execution failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"🚨 Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())