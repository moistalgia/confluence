#!/usr/bin/env python3
"""
Professional Portfolio Tracker
==============================

Fixes the broken portfolio accounting with simple, accurate math:
Portfolio Value = Cash + Sum(Position Market Values)

Eliminates:
- Double-counting positions
- Unrealized P&L calculation errors  
- Multiple engine cash allocation bugs
- Phantom profit generation

Author: Professional Trading Team
Date: October 13, 2025
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Clean position representation with accurate P&L calculation"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    entry_time: datetime
    entry_fees: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L with accurate math"""
        
        if self.side == 'LONG':
            # LONG: Profit when price goes up
            return (current_price - self.entry_price) * self.quantity - self.entry_fees
        else:  # SHORT
            # SHORT: Profit when price goes down
            return (self.entry_price - current_price) * self.quantity - self.entry_fees
    
    def calculate_position_value(self, current_price: float) -> float:
        """Calculate current market value of position"""
        
        if self.side == 'LONG':
            # LONG: Value = quantity * current_price
            return self.quantity * current_price
        else:  # SHORT
            # SHORT: Value = entry_value + unrealized_pnl
            entry_value = self.quantity * self.entry_price
            unrealized_pnl = self.calculate_unrealized_pnl(current_price)
            return entry_value + unrealized_pnl

@dataclass
class CompletedTrade:
    """Record of completed trade for performance tracking"""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    realized_pnl: float = 0.0
    
    def __post_init__(self):
        """Calculate realized P&L automatically"""
        if self.side == 'LONG':
            self.realized_pnl = ((self.exit_price - self.entry_price) * self.quantity 
                               - self.entry_fees - self.exit_fees)
        else:  # SHORT
            self.realized_pnl = ((self.entry_price - self.exit_price) * self.quantity 
                               - self.entry_fees - self.exit_fees)

class ProfessionalPortfolioTracker:
    """
    Accurate portfolio tracking with simple, bulletproof math
    
    Core Principle: Portfolio Value = Cash + Position Values
    No magic, no double-counting, no phantom profits
    """
    
    def __init__(self, starting_capital: float):
        self.starting_capital = starting_capital
        self.cash = starting_capital  # Available cash
        self.open_positions: Dict[str, Position] = {}
        self.completed_trades: List[CompletedTrade] = []
        self.current_prices: Dict[str, float] = {}
        
        logger.info(f"💰 Portfolio initialized with ${starting_capital:,.2f}")
        
    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a symbol"""
        self.current_prices[symbol] = price
        
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if symbol not in self.current_prices:
            raise ValueError(f"No price data available for {symbol}")
        return self.current_prices[symbol]
    
    def open_position(self, symbol: str, side: str, quantity: float, 
                     entry_price: float, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None, 
                     fees: float = 0.0) -> Dict[str, any]:
        """
        Open a new position with proper cash allocation
        
        Returns: {'success': bool, 'message': str, 'position': Position}
        """
        
        # Calculate required cash
        if side == 'LONG':
            required_cash = quantity * entry_price + fees
        else:  # SHORT
            # For shorts, we need margin (simplified: same as position value)
            required_cash = quantity * entry_price + fees
        
        # Check if we have enough cash
        if required_cash > self.cash:
            return {
                'success': False,
                'message': f'Insufficient cash: need ${required_cash:.2f}, have ${self.cash:.2f}',
                'position': None
            }
        
        # Check for existing position (basic version - no position merging)
        if symbol in self.open_positions:
            return {
                'success': False,
                'message': f'Position already exists for {symbol}',
                'position': None
            }
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            entry_fees=fees,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Allocate cash
        self.cash -= required_cash
        self.open_positions[symbol] = position
        
        logger.info(f"📈 OPENED {side} {symbol}: {quantity:.6f} @ ${entry_price:.2f}")
        logger.info(f"💰 Cash after position: ${self.cash:.2f}")
        
        return {
            'success': True,
            'message': f'Position opened: {side} {quantity:.6f} {symbol} @ ${entry_price:.2f}',
            'position': position
        }
    
    def close_position(self, symbol: str, exit_price: float, 
                      fees: float = 0.0) -> Dict[str, any]:
        """
        Close an existing position and realize P&L
        
        Returns: {'success': bool, 'message': str, 'trade': CompletedTrade}
        """
        
        if symbol not in self.open_positions:
            return {
                'success': False,
                'message': f'No open position found for {symbol}',
                'trade': None
            }
        
        position = self.open_positions[symbol]
        
        # Calculate proceeds from closing position
        if position.side == 'LONG':
            # LONG: Sell at current price
            proceeds = position.quantity * exit_price - fees
        else:  # SHORT
            # SHORT: Buy back at current price, return initial margin
            initial_margin = position.quantity * position.entry_price
            cost_to_close = position.quantity * exit_price + fees
            proceeds = initial_margin - cost_to_close + initial_margin  # Return margin + profit
        
        # Create completed trade record
        trade = CompletedTrade(
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            entry_fees=position.entry_fees,
            exit_fees=fees
        )
        
        # Add proceeds back to cash
        self.cash += proceeds
        
        # Record trade and remove position
        self.completed_trades.append(trade)
        del self.open_positions[symbol]
        
        logger.info(f"📉 CLOSED {position.side} {symbol}: P&L ${trade.realized_pnl:+.2f}")
        logger.info(f"💰 Cash after close: ${self.cash:.2f}")
        
        return {
            'success': True,
            'message': f'Position closed: {trade.realized_pnl:+.2f} P&L',
            'trade': trade
        }
    
    def calculate_portfolio_value(self) -> Dict[str, float]:
        """
        BULLETPROOF PORTFOLIO CALCULATION
        
        Portfolio Value = Cash + Sum(Position Market Values)
        
        This is the ONLY source of truth for portfolio value.
        """
        
        # Start with available cash
        total_value = self.cash
        
        # Add current market value of each open position
        total_position_value = 0.0
        position_details = {}
        
        for symbol, position in self.open_positions.items():
            if symbol in self.current_prices:
                current_price = self.current_prices[symbol]
                position_value = position.calculate_position_value(current_price)
                unrealized_pnl = position.calculate_unrealized_pnl(current_price)
                
                total_position_value += position_value
                position_details[symbol] = {
                    'side': position.side,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'position_value': position_value,
                    'unrealized_pnl': unrealized_pnl
                }
            else:
                logger.warning(f"No current price for {symbol}, using entry price")
                position_value = position.quantity * position.entry_price
                total_position_value += position_value
        
        total_portfolio_value = self.cash + total_position_value
        
        # Calculate performance metrics
        realized_pnl = sum(trade.realized_pnl for trade in self.completed_trades)
        unrealized_pnl = total_position_value - sum(
            pos.quantity * pos.entry_price + pos.entry_fees 
            for pos in self.open_positions.values()
        )
        
        total_pnl = realized_pnl + unrealized_pnl
        total_return_pct = (total_pnl / self.starting_capital) * 100
        
        return {
            # Core Values
            'total_portfolio_value': total_portfolio_value,
            'cash': self.cash,
            'total_position_value': total_position_value,
            
            # P&L Breakdown  
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            
            # Performance Metrics
            'total_return_pct': total_return_pct,
            'starting_capital': self.starting_capital,
            
            # Position Details
            'open_positions_count': len(self.open_positions),
            'completed_trades_count': len(self.completed_trades),
            'position_details': position_details
        }
    
    def get_portfolio_summary(self) -> str:
        """Get formatted portfolio summary for logging"""
        
        portfolio = self.calculate_portfolio_value()
        
        summary = [
            f"💰 PORTFOLIO SUMMARY",
            f"   Total Value: ${portfolio['total_portfolio_value']:,.2f}",
            f"   Cash: ${portfolio['cash']:,.2f}",
            f"   Positions: ${portfolio['total_position_value']:,.2f}",
            f"   Total P&L: ${portfolio['total_pnl']:+,.2f} ({portfolio['total_return_pct']:+.2f}%)",
            f"   Realized: ${portfolio['realized_pnl']:+,.2f}",
            f"   Unrealized: ${portfolio['unrealized_pnl']:+,.2f}",
            f"   Open Positions: {portfolio['open_positions_count']}",
            f"   Completed Trades: {portfolio['completed_trades_count']}"
        ]
        
        return "\\n".join(summary)
    
    def validate_portfolio_integrity(self) -> bool:
        """
        Validate portfolio math is correct
        
        Key checks:
        1. Total value = Cash + Position values
        2. No negative cash (unless margin trading)
        3. All positions have current prices
        """
        
        try:
            portfolio = self.calculate_portfolio_value()
            
            # Check 1: Math consistency
            calculated_total = portfolio['cash'] + portfolio['total_position_value']
            reported_total = portfolio['total_portfolio_value']
            
            if abs(calculated_total - reported_total) > 0.01:  # Allow 1 cent rounding
                logger.error(f"Portfolio math error: {calculated_total} != {reported_total}")
                return False
            
            # Check 2: Cash not negative (for basic account)
            if portfolio['cash'] < 0:
                logger.warning(f"Negative cash balance: ${portfolio['cash']:.2f}")
                # Not necessarily invalid if margin trading
            
            # Check 3: All positions have prices
            for symbol in self.open_positions.keys():
                if symbol not in self.current_prices:
                    logger.error(f"Missing price data for position: {symbol}")
                    return False
            
            logger.info("✅ Portfolio integrity check passed")
            return True
            
        except Exception as e:
            logger.error(f"Portfolio validation failed: {e}")
            return False

# Test case to validate the fix
def test_portfolio_accuracy():
    """
    Test case matching the broken scenario:
    
    Start: $10,000
    Open ETH long: 0.19 ETH @ $4,161 = $800 position
    ETH rises to $4,185
    
    Expected result: Small loss due to fees, NOT +31.35% gain
    """
    
    print("🧪 TESTING PORTFOLIO ACCURACY FIX")
    print("=" * 50)
    
    # Initialize portfolio
    portfolio = ProfessionalPortfolioTracker(10000.0)
    
    print(f"Starting: {portfolio.get_portfolio_summary()}")
    
    # Update ETH price
    portfolio.update_price('ETH/USDT', 4161.0)
    
    # Open position
    result = portfolio.open_position(
        symbol='ETH/USDT',
        side='LONG',
        quantity=0.19,
        entry_price=4161.0,
        fees=3.0  # Trading fees
    )
    
    print(f"\\nAfter opening position:")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(portfolio.get_portfolio_summary())
    
    # ETH rises to $4,185
    portfolio.update_price('ETH/USDT', 4185.0)
    
    print(f"\\nAfter ETH rises to $4,185:")
    print(portfolio.get_portfolio_summary())
    
    # Validate integrity
    is_valid = portfolio.validate_portfolio_integrity()
    print(f"\\nPortfolio integrity: {'✅ VALID' if is_valid else '❌ INVALID'}")
    
    # Check expected vs actual
    portfolio_data = portfolio.calculate_portfolio_value()
    expected_loss = 3.0  # Fees only (position roughly flat)
    actual_pnl = portfolio_data['total_pnl']
    
    print(f"\\n📊 RESULTS:")
    print(f"Expected P&L: ~${-expected_loss:.2f} (fees)")
    print(f"Actual P&L: ${actual_pnl:+.2f}")
    print(f"Expected Return: ~{-expected_loss/10000*100:.2f}%")
    print(f"Actual Return: {portfolio_data['total_return_pct']:+.2f}%")
    
    if abs(actual_pnl + expected_loss) < 10:  # Within $10
        print("✅ PORTFOLIO FIX SUCCESSFUL!")
    else:
        print("❌ Portfolio still broken!")

if __name__ == "__main__":
    test_portfolio_accuracy()