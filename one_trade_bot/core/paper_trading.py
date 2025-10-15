"""
Paper Trading System

Simulates real trading with live market data for system validation and performance testing.
Provides comprehensive order simulation, position tracking, and performance analytics.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class PaperOrder:
    """Represents a paper trading order"""
    id: str
    symbol: str
    direction: TradeDirection
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    fees_paid: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass  
class PaperPosition:
    """Represents an open paper trading position"""
    symbol: str
    direction: TradeDirection
    entry_price: float
    quantity: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    fees_paid: float = 0.0
    
    def update_current_price(self, price: float):
        """Update current price and calculate unrealized PnL"""
        self.current_price = price
        if self.direction == TradeDirection.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

@dataclass
class PaperTrade:
    """Represents a completed paper trade"""
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    fees_paid: float
    duration: timedelta
    setup_quality: Optional[int] = None
    confluence_score: Optional[float] = None
    
    @property
    def return_pct(self) -> float:
        """Calculate percentage return"""
        if self.direction == TradeDirection.LONG:
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100

class PaperTradingEngine:
    """
    Main paper trading engine that simulates order execution and position management
    """
    
    def __init__(self, config: Dict, data_provider):
        self.config = config
        self.data_provider = data_provider
        
        # Trading parameters
        self.initial_balance = config.get('initial_balance', 10000.0)
        self.current_balance = self.initial_balance
        self.position_size_pct = config.get('position_size_pct', 0.02)  # 2% risk per trade
        self.trading_fees_pct = config.get('trading_fees', 0.001)  # 0.1% fees
        self.slippage_pct = config.get('slippage', 0.0005)  # 0.05% slippage
        
        # State tracking
        self.orders: Dict[str, PaperOrder] = {}
        self.positions: Dict[str, PaperPosition] = {}
        self.completed_trades: List[PaperTrade] = []
        self.order_counter = 0
        
        # Performance tracking
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        logger.info(f"Paper Trading Engine initialized with ${self.initial_balance:,.2f}")

    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"PAPER_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self.order_counter:04d}"

    def place_order(self, symbol: str, direction: TradeDirection, order_type: OrderType,
                   quantity: float, price: Optional[float] = None, 
                   stop_price: Optional[float] = None,
                   setup_quality: Optional[int] = None,
                   confluence_score: Optional[float] = None) -> str:
        """
        Place a paper trading order
        
        Returns:
            Order ID string
        """
        order_id = self.generate_order_id()
        
        order = PaperOrder(
            id=order_id,
            symbol=symbol,
            direction=direction,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        self.orders[order_id] = order
        
        # Store setup metadata for trade analysis
        if hasattr(order, 'metadata'):
            order.metadata = {}
        else:
            order.metadata = {}
        order.metadata['setup_quality'] = setup_quality
        order.metadata['confluence_score'] = confluence_score
        
        logger.info(f"Order placed: {order.symbol} {order.direction.value} {order.order_type.value} "
                   f"qty={order.quantity} price={order.price}")
        
        return order_id

    def process_market_data(self, symbol: str, current_price: float, timestamp: datetime):
        """
        Process incoming market data and execute pending orders
        """
        # Update positions with current prices
        if symbol in self.positions:
            self.positions[symbol].update_current_price(current_price)
        
        # Check pending orders for this symbol
        symbol_orders = [order for order in self.orders.values() 
                        if order.symbol == symbol and order.status == OrderStatus.PENDING]
        
        for order in symbol_orders:
            self._try_execute_order(order, current_price, timestamp)
        
        # Check stop loss and take profit levels
        self._check_exit_conditions(symbol, current_price, timestamp)
        
        # Update equity curve
        self._update_performance_metrics(timestamp)

    def _try_execute_order(self, order: PaperOrder, current_price: float, timestamp: datetime):
        """Attempt to execute a pending order"""
        should_fill = False
        fill_price = current_price
        
        if order.order_type == OrderType.MARKET:
            should_fill = True
            # Apply slippage
            if order.direction == TradeDirection.LONG:
                fill_price = current_price * (1 + self.slippage_pct)
            else:
                fill_price = current_price * (1 - self.slippage_pct)
                
        elif order.order_type == OrderType.LIMIT:
            if order.direction == TradeDirection.LONG and current_price <= order.price:
                should_fill = True
                fill_price = order.price
            elif order.direction == TradeDirection.SHORT and current_price >= order.price:
                should_fill = True
                fill_price = order.price
                
        elif order.order_type == OrderType.STOP_LOSS:
            if order.direction == TradeDirection.LONG and current_price <= order.stop_price:
                should_fill = True
                fill_price = current_price * (1 - self.slippage_pct)  # Slippage on stop loss
            elif order.direction == TradeDirection.SHORT and current_price >= order.stop_price:
                should_fill = True
                fill_price = current_price * (1 + self.slippage_pct)
        
        if should_fill:
            self._execute_order(order, fill_price, timestamp)

    def _execute_order(self, order: PaperOrder, fill_price: float, timestamp: datetime):
        """Execute an order at the given price"""
        order.status = OrderStatus.FILLED
        order.filled_at = timestamp
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        
        # Calculate fees
        trade_value = order.quantity * fill_price
        order.fees_paid = trade_value * self.trading_fees_pct
        
        if order.order_type in [OrderType.MARKET, OrderType.LIMIT]:
            # Check if we're opening or closing a position
            existing_position = self.positions.get(order.symbol)
            
            if existing_position is None:
                # Opening a new position
                position = PaperPosition(
                    symbol=order.symbol,
                    direction=order.direction,
                    entry_price=fill_price,
                    quantity=order.quantity,
                    entry_time=timestamp,
                    current_price=fill_price,
                    fees_paid=order.fees_paid
                )
                self.positions[order.symbol] = position
                self.current_balance -= order.fees_paid
                
                logger.info(f"Position opened: {order.symbol} {order.direction.value} "
                           f"@ ${fill_price:.2f}, qty={order.quantity:.4f}")
                           
            elif existing_position.direction != order.direction:
                # Closing existing position (opposite direction)
                self._close_position(order.symbol, fill_price, timestamp, order.fees_paid, order)
            else:
                # Adding to existing position (same direction)
                logger.warning(f"Position modification not implemented for {order.symbol}")
                
        elif order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            # Closing a position via stop/target
            if order.symbol in self.positions:
                self._close_position(order.symbol, fill_price, timestamp, order.fees_paid, order)

    def _close_position(self, symbol: str, exit_price: float, timestamp: datetime, 
                       fees_paid: float, order: PaperOrder):
        """Close an existing position"""
        if symbol not in self.positions:
            logger.error(f"Attempted to close non-existent position: {symbol}")
            return
            
        position = self.positions[symbol]
        
        # Calculate final PnL
        if position.direction == TradeDirection.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        # Subtract fees
        total_fees = position.fees_paid + fees_paid
        net_pnl = pnl - total_fees
        
        # Update balance
        self.current_balance += net_pnl
        
        # Create completed trade record
        trade = PaperTrade(
            symbol=symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=timestamp,
            pnl=net_pnl,
            fees_paid=total_fees,
            duration=timestamp - position.entry_time,
            setup_quality=getattr(order, 'metadata', {}).get('setup_quality'),
            confluence_score=getattr(order, 'metadata', {}).get('confluence_score')
        )
        
        self.completed_trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Position closed: {symbol} @ ${exit_price:.2f}, "
                   f"P&L: ${net_pnl:.2f} ({trade.return_pct:.2f}%)")

    def _check_exit_conditions(self, symbol: str, current_price: float, timestamp: datetime):
        """Check if any positions should be closed based on stop loss or take profit"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        should_close = False
        exit_reason = ""
        
        # Check stop loss
        if position.stop_loss is not None:
            if (position.direction == TradeDirection.LONG and current_price <= position.stop_loss) or \
               (position.direction == TradeDirection.SHORT and current_price >= position.stop_loss):
                should_close = True
                exit_reason = "Stop Loss"
        
        # Check take profit
        if position.take_profit is not None:
            if (position.direction == TradeDirection.LONG and current_price >= position.take_profit) or \
               (position.direction == TradeDirection.SHORT and current_price <= position.take_profit):
                should_close = True
                exit_reason = "Take Profit"
        
        if should_close:
            # Create an exit order
            exit_order_id = self.place_order(
                symbol=symbol,
                direction=TradeDirection.SHORT if position.direction == TradeDirection.LONG else TradeDirection.LONG,
                order_type=OrderType.MARKET,
                quantity=position.quantity
            )
            
            # Execute immediately at current price
            exit_order = self.orders[exit_order_id]
            self._execute_order(exit_order, current_price, timestamp)
            
            logger.info(f"Position auto-closed: {symbol} - {exit_reason}")

    def _update_performance_metrics(self, timestamp: datetime):
        """Update performance tracking metrics"""
        # Calculate current equity (balance + unrealized PnL)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        current_equity = self.current_balance + unrealized_pnl
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'balance': self.current_balance,
            'unrealized_pnl': unrealized_pnl,
            'positions_count': len(self.positions)
        })
        
        # Update drawdown tracking
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown

    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive performance statistics"""
        if not self.completed_trades:
            return {'error': 'No completed trades'}
        
        # Basic statistics
        total_trades = len(self.completed_trades)
        winning_trades = [t for t in self.completed_trades if t.pnl > 0]
        losing_trades = [t for t in self.completed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        # PnL statistics
        total_pnl = sum(t.pnl for t in self.completed_trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Return statistics
        total_return_pct = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        returns = [t.return_pct for t in self.completed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Duration statistics
        avg_trade_duration = np.mean([t.duration.total_seconds() / 3600 for t in self.completed_trades])  # hours
        
        current_equity = self.current_balance + sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown_pct': self.max_drawdown * 100,
            'current_drawdown_pct': self.current_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration_hours': avg_trade_duration,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'current_equity': current_equity,
            'open_positions': len(self.positions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }

    def save_results(self, filepath: str):
        """Save trading results to file"""
        results = {
            'performance_stats': self.get_performance_stats(),
            'completed_trades': [asdict(trade) for trade in self.completed_trades],
            'equity_curve': self.equity_curve,
            'current_positions': [asdict(pos) for pos in self.positions.values()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")

    def get_trade_analysis(self) -> Dict[str, Any]:
        """Analyze trades by setup quality and confluence score"""
        if not self.completed_trades:
            return {}
        
        # Group trades by setup quality
        quality_groups = {}
        for trade in self.completed_trades:
            quality = trade.setup_quality or 0
            if quality not in quality_groups:
                quality_groups[quality] = []
            quality_groups[quality].append(trade)
        
        quality_analysis = {}
        for quality, trades in quality_groups.items():
            win_rate = len([t for t in trades if t.pnl > 0]) / len(trades) * 100
            avg_return = np.mean([t.return_pct for t in trades])
            quality_analysis[f'quality_{quality}'] = {
                'trade_count': len(trades),
                'win_rate': win_rate,
                'avg_return_pct': avg_return
            }
        
        # Confluence score analysis
        confluence_trades = [t for t in self.completed_trades if t.confluence_score is not None]
        if confluence_trades:
            high_conf = [t for t in confluence_trades if t.confluence_score >= 30]
            med_conf = [t for t in confluence_trades if 15 <= t.confluence_score < 30]
            low_conf = [t for t in confluence_trades if t.confluence_score < 15]
            
            confluence_analysis = {}
            for group_name, trades in [('high', high_conf), ('medium', med_conf), ('low', low_conf)]:
                if trades:
                    win_rate = len([t for t in trades if t.pnl > 0]) / len(trades) * 100
                    avg_return = np.mean([t.return_pct for t in trades])
                    confluence_analysis[f'{group_name}_confluence'] = {
                        'trade_count': len(trades),
                        'win_rate': win_rate,
                        'avg_return_pct': avg_return
                    }
        else:
            confluence_analysis = {}
        
        return {
            'by_setup_quality': quality_analysis,
            'by_confluence_score': confluence_analysis
        }