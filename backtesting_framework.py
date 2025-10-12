#!/usr/bin/env python3
"""
Advanced Backtesting Framework
Comprehensive strategy backtesting with performance metrics and trade simulation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for trading"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PARTIAL = "partial"

class TradeDirection(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"

@dataclass
class Order:
    """Trading order representation"""
    id: str
    symbol: str
    direction: TradeDirection
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'fill_price': self.fill_price,
            'fill_time': self.fill_time.isoformat() if self.fill_time else None,
            'commission': self.commission
        }

@dataclass
class Trade:
    """Completed trade representation"""
    id: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    commission: float
    duration: timedelta
    
    @classmethod
    def from_orders(cls, entry_order: Order, exit_order: Order) -> 'Trade':
        """Create trade from entry and exit orders"""
        
        # Calculate P&L
        if entry_order.direction == TradeDirection.LONG:
            pnl = (exit_order.fill_price - entry_order.fill_price) * entry_order.quantity
        else:
            pnl = (entry_order.fill_price - exit_order.fill_price) * entry_order.quantity
        
        pnl -= (entry_order.commission + exit_order.commission)
        pnl_percent = (pnl / (entry_order.fill_price * entry_order.quantity)) * 100
        
        return cls(
            id=f"{entry_order.id}_{exit_order.id}",
            symbol=entry_order.symbol,
            direction=entry_order.direction,
            entry_price=entry_order.fill_price,
            exit_price=exit_order.fill_price,
            quantity=entry_order.quantity,
            entry_time=entry_order.fill_time,
            exit_time=exit_order.fill_time,
            pnl=pnl,
            pnl_percent=pnl_percent,
            commission=entry_order.commission + exit_order.commission,
            duration=exit_order.fill_time - entry_order.fill_time
        )

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_position_size: float = 0.1  # 10% of capital per trade
    stop_loss_percent: float = 0.02  # 2% stop loss
    take_profit_percent: float = 0.04  # 4% take profit
    
    # Risk management
    max_daily_trades: int = 10
    max_concurrent_positions: int = 3
    risk_per_trade: float = 0.01  # 1% risk per trade
    
    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)

class Portfolio:
    """Portfolio management for backtesting"""
    
    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        
        # Positions and orders
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.orders: List[Order] = []
        self.completed_trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.drawdowns: List[float] = []
        self.daily_returns: List[float] = []
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0
        
        logger.info(f"Portfolio initialized with {initial_capital:,.2f} capital")
    
    def get_position_size(self, symbol: str, price: float, risk_amount: float) -> float:
        """Calculate position size based on risk amount"""
        
        # Simple position sizing: risk_amount / price
        # In practice, this would consider stop loss distance
        max_shares = self.current_capital * 0.1 / price  # Max 10% of capital
        risk_shares = risk_amount / price
        
        return min(max_shares, risk_shares)
    
    def place_order(self, order: Order) -> bool:
        """Place a trading order"""
        
        try:
            # Check if we have enough capital
            if order.direction == TradeDirection.LONG:
                required_capital = order.quantity * order.price if order.price else 0
                if required_capital > self.current_capital:
                    logger.warning(f"Insufficient capital for order {order.id}")
                    return False
            
            self.orders.append(order)
            logger.debug(f"Order placed: {order.id} - {order.direction.value} {order.quantity} {order.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False
    
    def process_market_data(self, timestamp: datetime, market_data: Dict[str, Dict]):
        """Process market data and execute orders"""
        
        try:
            # Process pending orders
            for order in self.orders:
                if order.status == OrderStatus.PENDING:
                    self._try_fill_order(order, timestamp, market_data)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(market_data)
            self.equity_curve.append((timestamp, portfolio_value))
            
            # Calculate daily return
            if len(self.equity_curve) > 1:
                prev_value = self.equity_curve[-2][1]
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def _try_fill_order(self, order: Order, timestamp: datetime, market_data: Dict[str, Dict]):
        """Try to fill a pending order"""
        
        if order.symbol not in market_data:
            return
        
        price_data = market_data[order.symbol]
        current_price = price_data.get('close', price_data.get('price', 0))
        
        if current_price <= 0:
            return
        
        # Apply slippage
        slippage_factor = 1.0005 if order.direction == TradeDirection.LONG else 0.9995
        fill_price = current_price * slippage_factor
        
        # Check if order should be filled
        should_fill = False
        
        if order.order_type == OrderType.MARKET:
            should_fill = True
        elif order.order_type == OrderType.LIMIT:
            if order.direction == TradeDirection.LONG and current_price <= order.price:
                should_fill = True
            elif order.direction == TradeDirection.SHORT and current_price >= order.price:
                should_fill = True
        
        if should_fill:
            # Fill the order
            order.status = OrderStatus.FILLED
            order.fill_price = fill_price
            order.fill_time = timestamp
            order.commission = order.quantity * fill_price * self.commission_rate
            
            # Update portfolio
            self._update_portfolio_after_fill(order)
            
            logger.debug(f"Order filled: {order.id} at {fill_price}")
    
    def _update_portfolio_after_fill(self, order: Order):
        """Update portfolio after order fill"""
        
        symbol = order.symbol
        
        if order.direction == TradeDirection.LONG:
            # Long position entry
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'direction': order.direction
                }
            
            pos = self.positions[symbol]
            total_quantity = pos['quantity'] + order.quantity
            total_value = (pos['quantity'] * pos['avg_price']) + (order.quantity * order.fill_price)
            pos['avg_price'] = total_value / total_quantity if total_quantity > 0 else 0
            pos['quantity'] = total_quantity
            
            # Update capital
            self.current_capital -= (order.quantity * order.fill_price + order.commission)
            
        else:
            # Short position or exit
            if symbol in self.positions and self.positions[symbol]['quantity'] > 0:
                # Exit long position
                pos = self.positions[symbol]
                
                # Calculate P&L for this exit
                pnl = (order.fill_price - pos['avg_price']) * min(order.quantity, pos['quantity'])
                pnl -= order.commission
                
                # Update position
                pos['quantity'] -= min(order.quantity, pos['quantity'])
                if pos['quantity'] <= 0:
                    del self.positions[symbol]
                
                # Update capital
                self.current_capital += (order.quantity * order.fill_price - order.commission)
                self.total_pnl += pnl
        
        self.total_commission += order.commission
    
    def _calculate_portfolio_value(self, market_data: Dict[str, Dict]) -> float:
        """Calculate current portfolio value"""
        
        total_value = self.current_capital
        
        # Add value of current positions
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].get('close', market_data[symbol].get('price', 0))
                position_value = position['quantity'] * current_price
                total_value += position_value
        
        return total_value
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not self.equity_curve:
            return {}
        
        # Basic metrics
        final_value = self.equity_curve[-1][1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate drawdown
        peak = self.initial_capital
        max_drawdown = 0
        current_drawdown = 0
        
        for timestamp, value in self.equity_curve:
            if value > peak:
                peak = value
                current_drawdown = 0
            else:
                current_drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, current_drawdown)
        
        # Sharpe ratio (simplified)
        if len(self.daily_returns) > 1:
            avg_return = np.mean(self.daily_returns)
            return_std = np.std(self.daily_returns)
            sharpe_ratio = (avg_return / return_std * np.sqrt(252)) if return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Win rate
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_percent': total_return,
            'total_pnl': self.total_pnl,
            'max_drawdown_percent': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_percent': win_rate,
            'total_commission': self.total_commission,
            'avg_trade_pnl': self.total_pnl / max(self.total_trades, 1),
            'profit_factor': self._calculate_profit_factor()
        }
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        
        gross_profit = sum(trade.pnl for trade in self.completed_trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.completed_trades if trade.pnl < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else 0

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.analyzer = None  # Will be set by backtester
        
    def initialize(self, backtester):
        """Initialize strategy with backtester reference"""
        self.backtester = backtester
        self.analyzer = backtester.analyzer
    
    def generate_signals(self, symbol: str, timestamp: datetime, market_data: Dict) -> List[Order]:
        """Generate trading signals - to be implemented by subclasses"""
        raise NotImplementedError("Strategy must implement generate_signals method")
    
    def should_exit_position(self, symbol: str, position: Dict, market_data: Dict) -> bool:
        """Check if position should be exited - to be implemented by subclasses"""
        return False

class ConfluenceStrategy(TradingStrategy):
    """Strategy based on confluence analysis"""
    
    def __init__(self, config: Dict = None):
        super().__init__("Confluence Strategy", config)
        
        # Strategy parameters
        self.min_confluence_score = config.get('min_confluence_score', 70)
        self.min_trend_alignment = config.get('min_trend_alignment', 75)
        self.position_size_pct = config.get('position_size_pct', 0.02)  # 2% of capital
    
    def generate_signals(self, symbol: str, timestamp: datetime, market_data: Dict) -> List[Order]:
        """Generate signals based on confluence analysis"""
        
        orders = []
        
        try:
            # Get analysis for symbol
            analysis = self.analyzer.analyze_multi_timeframe(symbol)
            
            if 'confluence_analysis' not in analysis:
                return orders
            
            confluence = analysis['confluence_analysis']
            overall = confluence.get('overall_confluence', {})
            trend_data = confluence.get('trend_alignment', {})
            
            confluence_score = overall.get('confluence_score', 0)
            trend_alignment = trend_data.get('alignment_percentage', 0)
            dominant_trend = trend_data.get('dominant_trend', 'NEUTRAL')
            
            # Check if criteria are met
            if (confluence_score >= self.min_confluence_score and 
                trend_alignment >= self.min_trend_alignment and
                dominant_trend != 'NEUTRAL'):
                
                current_price = market_data[symbol].get('close', market_data[symbol].get('price'))
                
                # Calculate position size
                portfolio_value = self.backtester.portfolio.current_capital
                position_value = portfolio_value * self.position_size_pct
                quantity = position_value / current_price
                
                # Create order
                direction = TradeDirection.LONG if dominant_trend == 'BULLISH' else TradeDirection.SHORT
                
                # Calculate stop loss and take profit
                stop_loss_pct = self.backtester.config.stop_loss_percent
                take_profit_pct = self.backtester.config.take_profit_percent
                
                if direction == TradeDirection.LONG:
                    stop_loss = current_price * (1 - stop_loss_pct)
                    take_profit = current_price * (1 + take_profit_pct)
                else:
                    stop_loss = current_price * (1 + stop_loss_pct)
                    take_profit = current_price * (1 - take_profit_pct)
                
                order = Order(
                    id=f"{symbol}_{timestamp.timestamp()}_{direction.value}",
                    symbol=symbol,
                    direction=direction,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=timestamp
                )
                
                orders.append(order)
                
                logger.info(f"Signal generated: {direction.value} {symbol} - Confluence: {confluence_score:.0f}%")
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
        
        return orders

class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, analyzer, config: BacktestConfig):
        self.analyzer = analyzer
        self.config = config
        self.portfolio = Portfolio(config.initial_capital, config.commission_rate)
        
        # Strategy
        self.strategy: Optional[TradingStrategy] = None
        
        # Data
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.results: Optional[Dict] = None
        
        logger.info("Backtester initialized")
    
    def set_strategy(self, strategy: TradingStrategy):
        """Set the trading strategy"""
        self.strategy = strategy
        self.strategy.initialize(self)
        logger.info(f"Strategy set: {strategy.name}")
    
    def load_historical_data(self, symbol: str, timeframe: str = '1h', periods: int = 1000):
        """Load historical data for backtesting"""
        
        try:
            # Use analyzer's exchange to fetch data
            ohlcv = self.analyzer.exchange.fetch_ohlcv(symbol, timeframe, limit=periods)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by date range if specified
            if self.config.start_date:
                df = df[df.index >= self.config.start_date]
            if self.config.end_date:
                df = df[df.index <= self.config.end_date]
            
            self.market_data[symbol] = df
            logger.info(f"Loaded {len(df)} data points for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
    
    def run_backtest(self, symbols: List[str]) -> Dict:
        """Run the backtesting simulation"""
        
        if not self.strategy:
            raise ValueError("No strategy set for backtesting")
        
        logger.info(f"Starting backtest with {len(symbols)} symbols")
        
        try:
            # Load data for all symbols
            for symbol in symbols:
                if symbol not in self.market_data:
                    self.load_historical_data(symbol)
            
            # Get common date range
            start_date = max(df.index.min() for df in self.market_data.values())
            end_date = min(df.index.max() for df in self.market_data.values())
            
            logger.info(f"Backtest period: {start_date} to {end_date}")
            
            # Create unified timeline
            timeline = pd.date_range(start=start_date, end=end_date, freq='H')  # Hourly
            
            # Run simulation
            for timestamp in timeline:
                # Get market data for this timestamp
                current_market_data = {}
                
                for symbol in symbols:
                    df = self.market_data[symbol]
                    
                    # Find closest data point
                    closest_idx = df.index.get_indexer([timestamp], method='nearest')[0]
                    if closest_idx >= 0 and closest_idx < len(df):
                        current_market_data[symbol] = df.iloc[closest_idx].to_dict()
                
                if current_market_data:
                    # Process market data (fill orders, update portfolio)
                    self.portfolio.process_market_data(timestamp, current_market_data)
                    
                    # Generate new signals
                    for symbol in symbols:
                        if symbol in current_market_data:
                            try:
                                signals = self.strategy.generate_signals(symbol, timestamp, current_market_data)
                                
                                for order in signals:
                                    self.portfolio.place_order(order)
                                    
                            except Exception as e:
                                logger.error(f"Error generating signals for {symbol}: {e}")
            
            # Calculate results
            self.results = self._compile_results()
            
            logger.info(f"Backtest completed. Final return: {self.results['performance']['total_return_percent']:.2f}%")
            return self.results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _compile_results(self) -> Dict:
        """Compile backtest results"""
        
        performance = self.portfolio.get_performance_metrics()
        
        results = {
            'config': self.config.to_dict(),
            'strategy': {
                'name': self.strategy.name,
                'config': self.strategy.config
            },
            'performance': performance,
            'equity_curve': [
                {'timestamp': ts.isoformat(), 'value': value}
                for ts, value in self.portfolio.equity_curve
            ],
            'trades': [asdict(trade) for trade in self.portfolio.completed_trades],
            'orders': [order.to_dict() for order in self.portfolio.orders],
            'summary': {
                'symbols_traded': len(set(order.symbol for order in self.portfolio.orders)),
                'backtest_duration': len(self.portfolio.equity_curve),
                'avg_trades_per_day': performance['total_trades'] / max(len(self.portfolio.equity_curve) / 24, 1)
            }
        }
        
        return results
    
    def save_results(self, filename: str):
        """Save backtest results to file"""
        
        if not self.results:
            logger.error("No results to save - run backtest first")
            return
        
        try:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results"""
        
        if not self.results:
            logger.error("No results to plot - run backtest first")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Equity curve
            equity_data = self.results['equity_curve']
            timestamps = [datetime.fromisoformat(d['timestamp']) for d in equity_data]
            values = [d['value'] for d in equity_data]
            
            axes[0, 0].plot(timestamps, values)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Portfolio Value')
            
            # Trade P&L distribution
            trades = self.results['trades']
            if trades:
                pnls = [trade['pnl'] for trade in trades]
                axes[0, 1].hist(pnls, bins=20, alpha=0.7)
                axes[0, 1].set_title('Trade P&L Distribution')
                axes[0, 1].set_xlabel('P&L')
                axes[0, 1].set_ylabel('Frequency')
            
            # Performance metrics
            perf = self.results['performance']
            metrics = ['Total Return %', 'Max Drawdown %', 'Win Rate %', 'Sharpe Ratio']
            values = [
                perf['total_return_percent'],
                perf['max_drawdown_percent'],
                perf['win_rate_percent'],
                perf['sharpe_ratio']
            ]
            
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_title('Key Performance Metrics')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Monthly returns heatmap (simplified)
            if len(timestamps) > 30:
                monthly_returns = pd.Series(values, index=timestamps).resample('M').last().pct_change().dropna()
                if len(monthly_returns) > 1:
                    axes[1, 1].plot(monthly_returns.index, monthly_returns * 100)
                    axes[1, 1].set_title('Monthly Returns')
                    axes[1, 1].set_xlabel('Month')
                    axes[1, 1].set_ylabel('Return %')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")

if __name__ == "__main__":
    # Demo usage
    from enhanced_multi_timeframe_analyzer import EnhancedMultiTimeframeAnalyzer
    
    # Create analyzer and backtester
    analyzer = EnhancedMultiTimeframeAnalyzer()
    
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        max_position_size=0.1,
        start_date=datetime.now() - timedelta(days=30)
    )
    
    backtester = Backtester(analyzer, config)
    
    # Set strategy
    strategy_config = {
        'min_confluence_score': 70,
        'min_trend_alignment': 75,
        'position_size_pct': 0.02
    }
    
    strategy = ConfluenceStrategy(strategy_config)
    backtester.set_strategy(strategy)
    
    print("Backtesting Framework Demo")
    print("==========================")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Strategy: {strategy.name}")
    print(f"Commission Rate: {config.commission_rate:.3f}")
    
    # In a real scenario, you would run:
    # results = backtester.run_backtest(['BTC/USD', 'ETH/USD'])
    # backtester.save_results('backtest_results.json')
    # backtester.plot_results('backtest_plot.png')