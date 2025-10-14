"""
Paper Trading Simulator
🎯 Real-time simulation with live market data

Features:
- Live market data integration
- Real-time daily workflow execution
- Position tracking with realistic fills
- Performance monitoring and reporting
- Risk management validation
- Daily execution logs

Simulates actual trading without real money risk
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import time
import asyncio
from dataclasses import dataclass, asdict

from core.data_provider import DataProvider
from core.workflow_engine import DailyWorkflowEngine
from core.position_manager import PositionManager, Position
# from utils.trade_logger import TradeLogger  # Optional import - will be available when needed

logger = logging.getLogger(__name__)

@dataclass
class PaperTrade:
    """
    Paper trading trade record
    """
    trade_id: str
    timestamp: str
    symbol: str
    direction: str
    
    # Order details
    order_type: str  # 'LIMIT', 'MARKET', 'STOP'
    order_price: float
    filled_price: float
    quantity: float
    
    # Execution details
    order_time: str
    fill_time: Optional[str]
    fill_status: str  # 'PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED'
    
    # P&L tracking (for closed trades)
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    realized_pnl: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PaperTrade':
        return cls(**data)


class PaperAccount:
    """
    Simulated trading account for paper trading
    """
    
    def __init__(self, initial_balance: float, account_file: str = 'data/paper_account.json'):
        """
        Initialize paper trading account
        
        Args:
            initial_balance: Starting balance
            account_file: File to persist account state
        """
        self.account_file = account_file
        
        # Load existing account or create new
        if os.path.exists(account_file):
            self._load_account()
        else:
            self.balance = initial_balance
            self.equity = initial_balance
            self.open_orders = []
            self.trade_history = []
            self.daily_pnl = 0.0
            self.total_pnl = 0.0
            self.created_at = datetime.utcnow().isoformat()
            self._save_account()
        
        logger.info(f"📊 Paper Account initialized: ${self.balance:,.0f} balance")
    
    def get_account_summary(self) -> Dict:
        """Get current account summary"""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'open_orders': len(self.open_orders),
            'total_trades': len(self.trade_history),
            'created_at': self.created_at
        }
    
    def place_order(self, symbol: str, direction: str, order_type: str, 
                   price: float, quantity: float) -> str:
        """
        Place paper trading order
        
        Returns:
            Order ID
        """
        order_id = f"PAPER_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self.open_orders)}"
        
        order = PaperTrade(
            trade_id=order_id,
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            direction=direction,
            order_type=order_type,
            order_price=price,
            filled_price=0.0,
            quantity=quantity,
            order_time=datetime.utcnow().isoformat(),
            fill_time=None,
            fill_status='PENDING'
        )
        
        self.open_orders.append(order)
        self._save_account()
        
        logger.info(f"📋 Paper Order Placed: {order_id} - {direction} {quantity} {symbol} @ ${price:,.4f}")
        
        return order_id
    
    def check_fills(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check if any pending orders should be filled
        
        Returns:
            List of filled order IDs
        """
        filled_orders = []
        
        for order in self.open_orders[:]:  # Copy list to avoid modification during iteration
            if order.fill_status != 'PENDING':
                continue
            
            current_price = current_prices.get(order.symbol)
            if current_price is None:
                continue
            
            should_fill = False
            fill_price = order.order_price
            
            # Check fill conditions based on order type
            if order.order_type == 'MARKET':
                should_fill = True
                fill_price = current_price
                
            elif order.order_type == 'LIMIT':
                if order.direction == 'LONG' and current_price <= order.order_price:
                    should_fill = True
                    fill_price = order.order_price  # Limit orders get filled at limit price or better
                elif order.direction == 'SHORT' and current_price >= order.order_price:
                    should_fill = True  
                    fill_price = order.order_price
            
            if should_fill:
                order.filled_price = fill_price
                order.fill_time = datetime.utcnow().isoformat()
                order.fill_status = 'FILLED'
                
                # Update account balance (for entry orders)
                position_value = order.quantity * fill_price
                # Note: In real implementation, we'd track separate cash/positions
                # For simplicity, we'll update on exit only
                
                filled_orders.append(order.trade_id)
                self.trade_history.append(order)
                
                logger.info(f"✅ Order Filled: {order.trade_id} - {order.direction} {order.quantity} "
                           f"{order.symbol} @ ${fill_price:,.4f}")
        
        # Remove filled orders from open orders
        self.open_orders = [o for o in self.open_orders if o.fill_status == 'PENDING']
        
        if filled_orders:
            self._save_account()
        
        return filled_orders
    
    def close_position(self, position: Position, exit_price: float, exit_reason: str) -> float:
        """
        Close position and update account
        
        Returns:
            Realized P&L
        """
        # Calculate P&L
        if position.direction == 'LONG':
            pnl = (exit_price - position.entry_price) * position.position_size
        else:
            pnl = (position.entry_price - exit_price) * position.position_size
        
        # Update account
        self.balance += pnl
        self.equity = self.balance  # Simplified
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        # Log the close
        logger.info(f"🔒 Position Closed: {position.symbol} {position.direction} "
                   f"P&L: ${pnl:+,.0f} (Reason: {exit_reason})")
        
        self._save_account()
        
        return pnl
    
    def _load_account(self) -> None:
        """Load account state from file"""
        try:
            with open(self.account_file, 'r') as f:
                data = json.load(f)
                
            self.balance = data['balance']
            self.equity = data['equity']  
            self.daily_pnl = data['daily_pnl']
            self.total_pnl = data['total_pnl']
            self.created_at = data['created_at']
            
            # Load orders and trades
            self.open_orders = [PaperTrade.from_dict(o) for o in data.get('open_orders', [])]
            self.trade_history = [PaperTrade.from_dict(t) for t in data.get('trade_history', [])]
            
        except Exception as e:
            logger.error(f"Failed to load paper account: {str(e)}")
            raise
    
    def _save_account(self) -> None:
        """Save account state to file"""
        try:
            os.makedirs(os.path.dirname(self.account_file), exist_ok=True)
            
            data = {
                'balance': self.balance,
                'equity': self.equity,
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'created_at': self.created_at,
                'open_orders': [o.to_dict() for o in self.open_orders],
                'trade_history': [t.to_dict() for t in self.trade_history],
                'last_updated': datetime.utcnow().isoformat()
            }
            
            with open(self.account_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save paper account: {str(e)}")


class PaperTradingSimulator:
    """
    Complete paper trading simulation system
    Runs daily workflow and manages positions with live data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize paper trading simulator
        
        Args:
            config: Complete bot configuration
        """
        self.config = config
        self.paper_config = config.get('paper_trading', {})
        
        # Initialize components
        self.data_provider = DataProvider(config.get('data_provider', {}))
        self.paper_account = PaperAccount(
            config.get('account', {}).get('balance', 10000),
            self.paper_config.get('account_file', 'data/paper_account.json')
        )
        self.position_manager = PositionManager(config.get('position_manager', {}))
        
        # Simulation settings
        self.daily_scan_time = self.paper_config.get('daily_scan_time', '09:00')  # UTC
        self.position_check_interval = self.paper_config.get('check_interval_minutes', 15)  # Minutes
        self.max_order_wait_hours = self.paper_config.get('max_order_wait_hours', 4)  # Hours
        
        # State tracking
        self.last_scan_date = None
        self.current_order_id = None
        self.simulation_running = False
        
        logger.info("🎭 Paper Trading Simulator initialized")
        logger.info(f"   Account Balance: ${self.paper_account.balance:,.0f}")
        logger.info(f"   Daily Scan Time: {self.daily_scan_time} UTC")
        logger.info(f"   Position Check Interval: {self.position_check_interval} minutes")
    
    async def start_simulation(self) -> None:
        """
        Start continuous paper trading simulation
        """
        logger.info("🚀 STARTING PAPER TRADING SIMULATION")
        logger.info("=" * 60)
        
        self.simulation_running = True
        
        try:
            while self.simulation_running:
                
                current_time = datetime.utcnow()
                current_date = current_time.strftime('%Y-%m-%d')
                
                # Check if it's time for daily scan
                if self._should_run_daily_scan(current_time):
                    await self._execute_daily_scan(current_date)
                    self.last_scan_date = current_date
                
                # Update existing position (if any)
                await self._update_positions()
                
                # Check for order fills
                await self._check_order_fills()
                
                # Log periodic status
                if current_time.minute % 30 == 0 and current_time.second < 10:  # Every 30 minutes
                    self._log_status()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Paper trading simulation stopped by user")
        except Exception as e:
            logger.error(f"🚨 Paper trading simulation error: {str(e)}")
        finally:
            self.simulation_running = False
            logger.info("📊 Paper trading simulation ended")
    
    def stop_simulation(self) -> None:
        """Stop paper trading simulation"""
        self.simulation_running = False
        logger.info("🛑 Stopping paper trading simulation...")
    
    def _should_run_daily_scan(self, current_time: datetime) -> bool:
        """
        Check if it's time to run daily scan
        """
        current_date = current_time.strftime('%Y-%m-%d')
        scan_hour, scan_minute = map(int, self.daily_scan_time.split(':'))
        
        # Check if we haven't scanned today and it's past scan time
        if (self.last_scan_date != current_date and 
            current_time.hour >= scan_hour and current_time.minute >= scan_minute):
            return True
        
        return False
    
    async def _execute_daily_scan(self, scan_date: str) -> None:
        """
        Execute daily workflow scan
        """
        logger.info(f"🔍 EXECUTING DAILY SCAN: {scan_date}")
        
        try:
            # Update account balance in config
            workflow_config = self.config.copy()
            workflow_config['account']['balance'] = self.paper_account.balance
            
            # Create workflow engine
            engine = DailyWorkflowEngine(workflow_config, self.data_provider)
            
            # Get current positions
            current_positions = []
            if self.position_manager.current_position:
                current_positions = [self.position_manager.current_position.to_dict()]
            
            # Run daily workflow
            workflow_result = engine.run_daily_workflow(current_positions)
            
            # Process workflow result
            if workflow_result['workflow_status'] == 'TRADE_READY':
                await self._process_trade_signal(workflow_result)
            else:
                logger.info(f"📋 Daily Decision: {workflow_result['message']}")
                
        except Exception as e:
            logger.error(f"🚨 Daily scan failed: {str(e)}")
    
    async def _process_trade_signal(self, workflow_result: Dict) -> None:
        """
        Process trade signal from daily workflow
        """
        symbol = workflow_result['the_one_trade']
        direction = workflow_result['trade_direction'] 
        entry_plan = workflow_result.get('entry_plan', {})
        
        logger.info(f"🎯 TRADE SIGNAL: {symbol} {direction}")
        
        # Check if we can open new position
        can_open, reason = self.position_manager.can_open_new_position()
        if not can_open:
            logger.warning(f"❌ Cannot open position: {reason}")
            return
        
        # Place entry order
        entry_price = entry_plan['entry_price']
        position_size = entry_plan['position_size']
        
        order_id = self.paper_account.place_order(
            symbol, direction, 'LIMIT', entry_price, position_size
        )
        
        self.current_order_id = order_id
        
        logger.info(f"📋 Entry order placed: {order_id}")
        logger.info(f"   Target: {direction} {position_size:.2f} {symbol} @ ${entry_price:,.4f}")
        logger.info(f"   Risk: ${entry_plan.get('risk_amount', 0):,.0f}")
    
    async def _update_positions(self) -> None:
        """
        Update existing positions with current market data
        """
        if not self.position_manager.current_position:
            return
        
        position = self.position_manager.current_position
        
        try:
            # Get current price
            df = self.data_provider.get_ohlcv(position.symbol, '1m', days=1)
            current_price = df['close'].iloc[-1]
            
            # Update position
            update_result = self.position_manager.update_position(current_price, self.data_provider)
            
            # Check for exit signal
            exit_signal = update_result.get('exit_signal')
            if exit_signal:
                logger.info(f"🚨 EXIT SIGNAL: {exit_signal['reason']}")
                
                # Close position in paper account
                exit_price = exit_signal['exit_price']
                pnl = self.paper_account.close_position(position, exit_price, exit_signal['type'])
                
                # Close position in position manager
                self.position_manager.close_position(exit_price, exit_signal['type'])
                
                logger.info(f"🔒 Position closed - P&L: ${pnl:+,.0f}")
                
        except Exception as e:
            logger.error(f"🚨 Position update failed: {str(e)}")
    
    async def _check_order_fills(self) -> None:
        """
        Check for order fills and update positions
        """
        if not self.paper_account.open_orders:
            return
        
        try:
            # Get current prices for all symbols with open orders
            symbols = list(set(o.symbol for o in self.paper_account.open_orders))
            current_prices = {}
            
            for symbol in symbols:
                df = self.data_provider.get_ohlcv(symbol, '1m', days=1)
                current_prices[symbol] = df['close'].iloc[-1]
            
            # Check fills
            filled_orders = self.paper_account.check_fills(current_prices)
            
            # Process filled entry orders
            for order_id in filled_orders:
                if order_id == self.current_order_id:
                    await self._process_order_fill(order_id)
                    
        except Exception as e:
            logger.error(f"🚨 Order fill check failed: {str(e)}")
    
    async def _process_order_fill(self, order_id: str) -> None:
        """
        Process filled entry order
        """
        # Find the filled order
        filled_order = None
        for trade in self.paper_account.trade_history:
            if trade.trade_id == order_id and trade.fill_status == 'FILLED':
                filled_order = trade
                break
        
        if not filled_order:
            logger.error(f"Could not find filled order: {order_id}")
            return
        
        logger.info(f"✅ ENTRY FILLED: {order_id}")
        
        # Open position in position manager  
        position_result = self.position_manager.open_position(
            symbol=filled_order.symbol,
            direction=filled_order.direction,
            entry_price=filled_order.filled_price,
            stop_price=0.0,  # Will be calculated by position manager
            position_size=filled_order.quantity,
            risk_amount=self.paper_account.balance * 0.01,  # 1% risk
            account_balance=self.paper_account.balance
        )
        
        if position_result['success']:
            logger.info(f"📊 Position opened in manager: {filled_order.symbol}")
            self.current_order_id = None  # Clear pending order
        else:
            logger.error(f"Failed to open position: {position_result['error']}")
    
    def _log_status(self) -> None:
        """Log current simulation status"""
        account_summary = self.paper_account.get_account_summary()
        position_summary = self.position_manager.get_position_summary()
        
        logger.info(f"📊 STATUS UPDATE:")
        logger.info(f"   Balance: ${account_summary['balance']:,.0f} "
                   f"(P&L: ${account_summary['total_pnl']:+,.0f})")
        
        if position_summary['has_position']:
            logger.info(f"   Position: {position_summary['summary']}")
            logger.info(f"   Unrealized P&L: ${position_summary['unrealized_pnl']:+,.0f}")
        else:
            logger.info(f"   Position: None")
        
        if self.paper_account.open_orders:
            logger.info(f"   Pending Orders: {len(self.paper_account.open_orders)}")
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary
        """
        account_summary = self.paper_account.get_account_summary()
        position_summary = self.position_manager.get_position_summary()
        
        # Calculate some basic metrics
        total_trades = len([t for t in self.paper_account.trade_history if t.exit_price is not None])
        
        return {
            'account': account_summary,
            'position': position_summary,
            'total_trades': total_trades,
            'simulation_running': self.simulation_running,
            'last_scan_date': self.last_scan_date,
            'pending_orders': len(self.paper_account.open_orders)
        }


async def run_paper_trading_simulation(config_path: str = 'config.yaml') -> None:
    """
    Convenience function to run paper trading simulation
    
    Args:
        config_path: Path to configuration file
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and start simulator
    simulator = PaperTradingSimulator(config)
    
    try:
        await simulator.start_simulation()
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
    finally:
        simulator.stop_simulation()


# CLI interface for paper trading
if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser(description='One Trade Bot Paper Trading')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paper_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run simulation
    asyncio.run(run_paper_trading_simulation(args.config))