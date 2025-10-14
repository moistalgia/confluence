"""
Position Manager
🎯 Handles single position with 1% account risk management

Responsibilities:
- Position sizing with 1% account risk
- Stop-loss and profit target management
- Single position enforcement (one trade at a time)
- Position tracking and updates
- Exit signal monitoring

STRICT RULES:
- Maximum 1 open position at any time
- Exactly 1% account risk per trade
- No position scaling or averaging
- Disciplined stop-loss execution
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from core.data_provider import DataProvider, TechnicalIndicators

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """
    Data structure for tracking a single position
    """
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    position_size: float
    stop_loss_price: float
    profit_target_price: float
    
    # Risk metrics
    risk_amount: float
    account_balance_at_entry: float
    
    # Timestamps
    entry_time: str
    last_update_time: str
    
    # Status tracking
    status: str  # 'OPEN', 'STOPPED_OUT', 'TARGET_HIT', 'MANUALLY_CLOSED'
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # Trailing stop (if enabled)
    trailing_stop_price: Optional[float] = None
    highest_price_since_entry: Optional[float] = None  # For LONG
    lowest_price_since_entry: Optional[float] = None   # For SHORT
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create position from dictionary"""
        return cls(**data)


class PositionManager:
    """
    Manages single position with strict 1% risk discipline
    Enforces "one trade at a time" rule
    """
    
    def __init__(self, config: Dict):
        """
        Initialize position manager
        
        Args:
            config: Position management configuration
        """
        # Risk parameters
        self.max_account_risk = config.get('max_account_risk', 0.01)        # 1% max risk
        self.profit_target_ratio = config.get('profit_target_ratio', 2.0)   # 2:1 R:R
        
        # Stop-loss management
        self.trailing_stop_enabled = config.get('trailing_stop_enabled', True)
        self.trailing_activation_ratio = config.get('trailing_activation_ratio', 1.0)  # Start at 1R
        self.trailing_stop_distance = config.get('trailing_stop_distance', 0.02)       # 2% trailing
        
        # Position limits
        self.max_open_positions = 1  # Strict one position limit
        
        # Position storage
        self.position_file = config.get('position_file', 'data/current_position.json')
        self.current_position: Optional[Position] = None
        
        # Load existing position if any
        self._load_position()
        
        logger.info("📊 Position Manager initialized")
        logger.info(f"   Max account risk: {self.max_account_risk:.1%}")
        logger.info(f"   Max positions: {self.max_open_positions}")
        logger.info(f"   Trailing stops: {'Enabled' if self.trailing_stop_enabled else 'Disabled'}")
        
        if self.current_position:
            logger.info(f"   Current position: {self.current_position.symbol} {self.current_position.direction}")
    
    def can_open_new_position(self) -> Tuple[bool, str]:
        """
        Check if new position can be opened
        
        Returns:
            Tuple of (can_open, reason)
        """
        if self.current_position and self.current_position.status == 'OPEN':
            return False, f"Already have open position: {self.current_position.symbol}"
        
        return True, "Ready to open new position"
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                              stop_price: float) -> Tuple[float, float]:
        """
        Calculate position size for 1% account risk
        
        Args:
            account_balance: Current account balance
            entry_price: Planned entry price
            stop_price: Stop-loss price
            
        Returns:
            Tuple of (position_size, risk_amount)
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_price)
        
        # Calculate maximum risk amount (1% of account)
        max_risk_amount = account_balance * self.max_account_risk
        
        # Calculate position size
        position_size = max_risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        return position_size, max_risk_amount
    
    def open_position(self, symbol: str, direction: str, entry_price: float,
                     stop_price: float, position_size: float, risk_amount: float,
                     account_balance: float) -> Dict:
        """
        Open new position with calculated parameters
        
        Returns:
            Dict with position opening result
        """
        try:
            # Check if we can open new position
            can_open, reason = self.can_open_new_position()
            if not can_open:
                logger.error(f"❌ Cannot open position: {reason}")
                return {
                    'success': False,
                    'error': reason,
                    'position': None
                }
            
            # Calculate profit target
            risk_per_unit = abs(entry_price - stop_price)
            reward_per_unit = risk_per_unit * self.profit_target_ratio
            
            if direction == 'LONG':
                profit_target = entry_price + reward_per_unit
            else:  # SHORT
                profit_target = entry_price - reward_per_unit
            
            # Create position object
            now = datetime.utcnow().isoformat()
            
            position = Position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                position_size=position_size,
                stop_loss_price=stop_price,
                profit_target_price=profit_target,
                risk_amount=risk_amount,
                account_balance_at_entry=account_balance,
                entry_time=now,
                last_update_time=now,
                status='OPEN',
                current_price=entry_price
            )
            
            # Initialize tracking prices for trailing stops
            if direction == 'LONG':
                position.highest_price_since_entry = entry_price
            else:
                position.lowest_price_since_entry = entry_price
            
            # Store position
            self.current_position = position
            self._save_position()
            
            # Log position opening
            position_value = position_size * entry_price
            logger.info(f"✅ POSITION OPENED:")
            logger.info(f"   Symbol: {symbol} {direction}")
            logger.info(f"   Entry: ${entry_price:,.4f}")
            logger.info(f"   Size: {position_size:,.2f} units (${position_value:,.0f})")
            logger.info(f"   Stop: ${stop_price:,.4f}")
            logger.info(f"   Target: ${profit_target:,.4f}")
            logger.info(f"   Risk: ${risk_amount:,.0f} ({risk_amount/account_balance:.1%})")
            
            return {
                'success': True,
                'position': position,
                'message': f"Position opened: {symbol} {direction}"
            }
            
        except Exception as e:
            logger.error(f"🚨 Failed to open position: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'position': None
            }
    
    def update_position(self, current_price: float, data_provider: DataProvider) -> Dict:
        """
        Update current position with latest price and check exit conditions
        
        Args:
            current_price: Latest market price
            data_provider: For additional market data if needed
            
        Returns:
            Dict with update results and any exit signals
        """
        if not self.current_position or self.current_position.status != 'OPEN':
            return {
                'success': False,
                'error': 'No open position to update',
                'exit_signal': None
            }
        
        try:
            position = self.current_position
            
            # Update current price and timestamp
            position.current_price = current_price
            position.last_update_time = datetime.utcnow().isoformat()
            
            # Calculate unrealized P&L
            if position.direction == 'LONG':
                position.unrealized_pnl = (current_price - position.entry_price) * position.position_size
            else:  # SHORT
                position.unrealized_pnl = (position.entry_price - current_price) * position.position_size
            
            position.unrealized_pnl_pct = position.unrealized_pnl / (position.position_size * position.entry_price)
            
            # Update extreme prices for trailing stops
            if position.direction == 'LONG':
                if position.highest_price_since_entry is None or current_price > position.highest_price_since_entry:
                    position.highest_price_since_entry = current_price
            else:  # SHORT
                if position.lowest_price_since_entry is None or current_price < position.lowest_price_since_entry:
                    position.lowest_price_since_entry = current_price
            
            # Check exit conditions
            exit_signal = self._check_exit_conditions(position, current_price)
            
            # Update trailing stop if applicable
            if self.trailing_stop_enabled and not exit_signal:
                self._update_trailing_stop(position)
            
            # Save updated position
            self._save_position()
            
            # Log update
            pnl_color = "🟢" if position.unrealized_pnl >= 0 else "🔴"
            logger.debug(f"📊 Position Update: {position.symbol} "
                        f"{pnl_color} ${position.unrealized_pnl:+,.0f} ({position.unrealized_pnl_pct:+.1%})")
            
            return {
                'success': True,
                'position': position,
                'exit_signal': exit_signal,
                'unrealized_pnl': position.unrealized_pnl
            }
            
        except Exception as e:
            logger.error(f"🚨 Failed to update position: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'exit_signal': None
            }
    
    def _check_exit_conditions(self, position: Position, current_price: float) -> Optional[Dict]:
        """
        Check if any exit conditions are met
        
        Returns:
            Exit signal dict if exit needed, None otherwise
        """
        # Check stop-loss
        if position.direction == 'LONG':
            if current_price <= position.stop_loss_price:
                return {
                    'type': 'STOP_LOSS',
                    'exit_price': position.stop_loss_price,
                    'reason': f"Price ${current_price:,.4f} hit stop ${position.stop_loss_price:,.4f}"
                }
            # Check trailing stop
            if position.trailing_stop_price and current_price <= position.trailing_stop_price:
                return {
                    'type': 'TRAILING_STOP',
                    'exit_price': position.trailing_stop_price,
                    'reason': f"Price ${current_price:,.4f} hit trailing stop ${position.trailing_stop_price:,.4f}"
                }
        else:  # SHORT
            if current_price >= position.stop_loss_price:
                return {
                    'type': 'STOP_LOSS',
                    'exit_price': position.stop_loss_price,
                    'reason': f"Price ${current_price:,.4f} hit stop ${position.stop_loss_price:,.4f}"
                }
            # Check trailing stop
            if position.trailing_stop_price and current_price >= position.trailing_stop_price:
                return {
                    'type': 'TRAILING_STOP',
                    'exit_price': position.trailing_stop_price,
                    'reason': f"Price ${current_price:,.4f} hit trailing stop ${position.trailing_stop_price:,.4f}"
                }
        
        # Check profit target
        if position.direction == 'LONG':
            if current_price >= position.profit_target_price:
                return {
                    'type': 'PROFIT_TARGET',
                    'exit_price': position.profit_target_price,
                    'reason': f"Price ${current_price:,.4f} hit target ${position.profit_target_price:,.4f}"
                }
        else:  # SHORT
            if current_price <= position.profit_target_price:
                return {
                    'type': 'PROFIT_TARGET',
                    'exit_price': position.profit_target_price,
                    'reason': f"Price ${current_price:,.4f} hit target ${position.profit_target_price:,.4f}"
                }
        
        return None
    
    def _update_trailing_stop(self, position: Position) -> None:
        """
        Update trailing stop price based on favorable price movement
        """
        if not self.trailing_stop_enabled:
            return
            
        risk_per_unit = abs(position.entry_price - position.stop_loss_price)
        activation_threshold = risk_per_unit * self.trailing_activation_ratio
        
        if position.direction == 'LONG':
            # Only activate trailing stop after favorable movement
            favorable_move = position.highest_price_since_entry - position.entry_price
            
            if favorable_move >= activation_threshold:
                # Calculate new trailing stop
                trail_distance = position.highest_price_since_entry * self.trailing_stop_distance
                new_trailing_stop = position.highest_price_since_entry - trail_distance
                
                # Only update if it's better than current stop
                current_stop = position.trailing_stop_price or position.stop_loss_price
                if new_trailing_stop > current_stop:
                    position.trailing_stop_price = new_trailing_stop
                    logger.info(f"🔄 Trailing stop updated: ${new_trailing_stop:,.4f}")
                    
        else:  # SHORT
            # Only activate trailing stop after favorable movement  
            favorable_move = position.entry_price - position.lowest_price_since_entry
            
            if favorable_move >= activation_threshold:
                # Calculate new trailing stop
                trail_distance = position.lowest_price_since_entry * self.trailing_stop_distance
                new_trailing_stop = position.lowest_price_since_entry + trail_distance
                
                # Only update if it's better than current stop
                current_stop = position.trailing_stop_price or position.stop_loss_price
                if new_trailing_stop < current_stop:
                    position.trailing_stop_price = new_trailing_stop
                    logger.info(f"🔄 Trailing stop updated: ${new_trailing_stop:,.4f}")
    
    def close_position(self, exit_price: float, exit_reason: str) -> Dict:
        """
        Close current position at specified price
        
        Returns:
            Dict with closing results
        """
        if not self.current_position or self.current_position.status != 'OPEN':
            return {
                'success': False,
                'error': 'No open position to close',
                'pnl': 0
            }
        
        try:
            position = self.current_position
            
            # Calculate final P&L
            if position.direction == 'LONG':
                realized_pnl = (exit_price - position.entry_price) * position.position_size
            else:  # SHORT
                realized_pnl = (position.entry_price - exit_price) * position.position_size
            
            # Update position status
            position.status = exit_reason.upper().replace(' ', '_')
            position.current_price = exit_price
            position.unrealized_pnl = realized_pnl
            position.last_update_time = datetime.utcnow().isoformat()
            
            # Save final position state
            self._save_position()
            
            # Log position closing
            pnl_color = "🟢" if realized_pnl >= 0 else "🔴"
            pnl_pct = realized_pnl / (position.position_size * position.entry_price)
            logger.info(f"🔒 POSITION CLOSED:")
            logger.info(f"   Symbol: {position.symbol} {position.direction}")
            logger.info(f"   Entry: ${position.entry_price:,.4f} → Exit: ${exit_price:,.4f}")
            logger.info(f"   Reason: {exit_reason}")
            logger.info(f"   P&L: {pnl_color} ${realized_pnl:+,.0f} ({pnl_pct:+.1%})")
            
            # Clear current position (ready for next trade)
            self.current_position = None
            
            return {
                'success': True,
                'position': position,
                'realized_pnl': realized_pnl,
                'pnl_pct': pnl_pct,
                'message': f"Position closed: {exit_reason}"
            }
            
        except Exception as e:
            logger.error(f"🚨 Failed to close position: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pnl': 0
            }
    
    def get_position_summary(self) -> Dict:
        """
        Get summary of current position
        
        Returns:
            Dict with position summary
        """
        if not self.current_position:
            return {
                'has_position': False,
                'can_open_new': True,
                'summary': 'No open position - ready for new trade'
            }
        
        position = self.current_position
        
        return {
            'has_position': True,
            'can_open_new': position.status != 'OPEN',
            'symbol': position.symbol,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'unrealized_pnl': position.unrealized_pnl,
            'unrealized_pnl_pct': position.unrealized_pnl_pct,
            'risk_amount': position.risk_amount,
            'status': position.status,
            'entry_time': position.entry_time,
            'summary': f"Open {position.direction} {position.symbol} @ ${position.entry_price:,.4f}"
        }
    
    def _load_position(self) -> None:
        """Load position from file if exists"""
        try:
            with open(self.position_file, 'r') as f:
                position_data = json.load(f)
                self.current_position = Position.from_dict(position_data)
                logger.info(f"📂 Loaded position: {self.current_position.symbol} {self.current_position.status}")
        except FileNotFoundError:
            logger.debug("No existing position file found")
        except Exception as e:
            logger.warning(f"Failed to load position: {str(e)}")
    
    def _save_position(self) -> None:
        """Save current position to file"""
        try:
            import os
            os.makedirs(os.path.dirname(self.position_file), exist_ok=True)
            
            if self.current_position:
                with open(self.position_file, 'w') as f:
                    json.dump(self.current_position.to_dict(), f, indent=2)
            else:
                # Remove position file if no position
                if os.path.exists(self.position_file):
                    os.remove(self.position_file)
        except Exception as e:
            logger.error(f"Failed to save position: {str(e)}")