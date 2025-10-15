#!/usr/bin/env python3
"""
Intelligent Exit Manager - Rule-Based Position Management

This module implements the intelligent exit system that enhances our disciplined
trading approach with objective, pre-defined exit rules. The key philosophy:

"We follow pre-defined, objective rules for both entry AND exit"
- NOT emotional decisions ("I feel nervous")  
- NOT gut feelings ("This doesn't look right")
- YES rule-based logic ("RSI divergence + volume dryup = exit")

This maintains our discipline while protecting profits intelligently.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ExitSignal:
    """Represents an exit signal with all relevant information"""
    should_exit: bool
    exit_type: str  # 'STOP', 'TARGET', 'TRAIL', 'MOMENTUM', 'RESISTANCE', 'TIME'
    exit_price: float
    reason: str
    priority: int  # Lower = higher priority
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    signals: Optional[Dict[str, Any]] = None

@dataclass
class PositionStatus:
    """Current position status information"""
    symbol: str
    entry_price: float
    current_price: float
    target_price: float
    current_stop: float
    profit_pct: float
    profit_usd: float
    progress_to_target: float
    time_in_trade_hours: float
    position_size: float

class IntelligentExitManager:
    """
    Manages intelligent exit logic for positions
    
    Supports 4 levels of sophistication:
    1. Basic Trailing Stops (always recommended)
    2. Momentum Reversal Detection (technical indicators)
    3. Resistance Rejection Analysis (price action) 
    4. Time Decay Management (time-based exits)
    
    Each level can be enabled/disabled via configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exit_config = config.get('intelligent_exits', {})
        
        # Configuration for each exit level
        self.trailing_enabled = self.exit_config.get('trailing_stop', {}).get('enabled', True)
        self.momentum_enabled = self.exit_config.get('momentum_reversal', {}).get('enabled', False)
        self.resistance_enabled = self.exit_config.get('resistance_rejection', {}).get('enabled', False)
        self.time_decay_enabled = self.exit_config.get('time_decay', {}).get('enabled', False)
        
        # Tracking for trailing stops
        self.trailing_stops: Dict[str, float] = {}  # symbol -> current trailing stop
        
        logger.info(f"🧠 Intelligent Exit Manager initialized")
        logger.info(f"   Trailing Stops: {'✅' if self.trailing_enabled else '❌'}")
        logger.info(f"   Momentum Detection: {'✅' if self.momentum_enabled else '❌'}")
        logger.info(f"   Resistance Analysis: {'✅' if self.resistance_enabled else '❌'}")
        logger.info(f"   Time Decay: {'✅' if self.time_decay_enabled else '❌'}")
    
    def check_exit_conditions(self, position: Dict[str, Any], current_price: float) -> ExitSignal:
        """
        Check all enabled exit conditions in priority order
        
        Returns the highest priority exit signal if any trigger
        """
        symbol = position['symbol']
        entry_price = position['entry_price']
        entry_time = position.get('entry_time', datetime.now())
        original_stop = position['stop_loss']
        target = position['take_profit']
        
        # Initialize trailing stop if not exists
        if symbol not in self.trailing_stops:
            self.trailing_stops[symbol] = original_stop
        
        current_stop = self.trailing_stops[symbol]
        profit_pct = (current_price - entry_price) / entry_price
        in_profit = profit_pct > 0
        
        # =====================================================
        # PRIORITY 1: Stop Loss (Always checked first)
        # =====================================================
        if current_price <= current_stop:
            return ExitSignal(
                should_exit=True,
                exit_type='STOP',
                exit_price=current_stop,
                reason=f'Stop loss hit at ${current_stop:.4f}',
                priority=1,
                confidence='HIGH'
            )
        
        # =====================================================
        # PRIORITY 2: Target Hit (Original plan worked!)
        # =====================================================
        if current_price >= target:
            return ExitSignal(
                should_exit=True,
                exit_type='TARGET',
                exit_price=target,
                reason=f'Target reached at ${target:.4f}',
                priority=2,
                confidence='HIGH'
            )
        
        # Only check intelligent exits if in profit
        if not in_profit:
            return ExitSignal(False, 'HOLD', current_price, 'Not in profit yet', 999, 'LOW')
        
        # =====================================================
        # PRIORITY 3: Update Trailing Stop (Risk Management)
        # =====================================================
        if self.trailing_enabled:
            trailing_result = self._check_trailing_stop(entry_price, current_price, current_stop, profit_pct)
            if trailing_result['updated']:
                old_stop = current_stop
                self.trailing_stops[symbol] = trailing_result['new_stop']
                logger.info(f"📈 {symbol} trailing stop: ${old_stop:.4f} → ${trailing_result['new_stop']:.4f}")
                logger.info(f"   Reason: {trailing_result['reason']}")
        
        # =====================================================
        # PRIORITY 4: Momentum Reversal (Early warning)
        # =====================================================
        if self.momentum_enabled:
            momentum_config = self.exit_config.get('momentum_reversal', {})
            min_profit = momentum_config.get('min_profit', 0.015)
            
            if profit_pct >= min_profit:
                momentum_signal = self._check_momentum_reversal(symbol, momentum_config)
                if momentum_signal['should_exit']:
                    return ExitSignal(
                        should_exit=True,
                        exit_type='MOMENTUM_REVERSAL',
                        exit_price=current_price,
                        reason=f"Momentum reversal: {momentum_signal['signal_count']}/4 signals",
                        priority=4,
                        confidence=momentum_signal['confidence'],
                        signals=momentum_signal['signals']
                    )
        
        # =====================================================
        # PRIORITY 5: Resistance Rejection (Technical)
        # =====================================================
        if self.resistance_enabled:
            resistance_config = self.exit_config.get('resistance_rejection', {})
            min_profit = resistance_config.get('min_profit', 0.02)
            
            if profit_pct >= min_profit:
                resistance_signal = self._check_resistance_rejection(symbol, current_price, resistance_config)
                if resistance_signal['should_exit']:
                    return ExitSignal(
                        should_exit=True,
                        exit_type='RESISTANCE_REJECTION',
                        exit_price=current_price,
                        reason=resistance_signal['reason'],
                        priority=5,
                        confidence='MEDIUM'
                    )
        
        # =====================================================
        # PRIORITY 6: Time Decay (Last resort)
        # =====================================================
        if self.time_decay_enabled:
            time_config = self.exit_config.get('time_decay', {})
            min_profit = time_config.get('min_profit', 0.01)
            
            if profit_pct >= min_profit:
                time_signal = self._check_time_decay(entry_time, entry_price, target, current_price, time_config)
                if time_signal['should_exit']:
                    return ExitSignal(
                        should_exit=True,
                        exit_type='TIME_DECAY',
                        exit_price=current_price,
                        reason=time_signal['reason'],
                        priority=6,
                        confidence='LOW'
                    )
        
        # =====================================================
        # No exit triggered - hold position
        # =====================================================
        return ExitSignal(False, 'HOLD', current_price, 'All conditions checked - holding', 999, 'LOW')
    
    def _check_trailing_stop(self, entry_price: float, current_price: float, 
                           current_stop: float, profit_pct: float) -> Dict[str, Any]:
        """
        Level 1: Basic Trailing Stop Logic
        
        This is the foundation - protects 80% of reversal scenarios
        """
        config = self.exit_config.get('trailing_stop', {})
        
        # Stage 1: Move to breakeven protection
        breakeven_threshold = config.get('breakeven_at', 0.01)  # 1% profit
        if profit_pct >= breakeven_threshold and current_stop < entry_price:
            return {
                'updated': True,
                'new_stop': entry_price,
                'reason': f'Breakeven protection at {profit_pct*100:.1f}% profit'
            }
        
        # Stage 2: Trail at 50% of profit
        trail_50_threshold = config.get('trail_50_at', 0.02)  # 2% profit
        trail_50_pct = config.get('trail_50_percent', 0.5)  # 50% of profit
        if profit_pct >= trail_50_threshold:
            profit_amount = current_price - entry_price
            potential_stop = entry_price + (profit_amount * trail_50_pct)
            if potential_stop > current_stop:
                return {
                    'updated': True,
                    'new_stop': potential_stop,
                    'reason': f'Trailing 50% of profit at {profit_pct*100:.1f}%'
                }
        
        # Stage 3: Trail at 70% of profit for positions close to target
        trail_70_threshold = config.get('trail_70_at', 0.03)  # 3% profit
        trail_70_pct = config.get('trail_70_percent', 0.7)  # 70% of profit
        if profit_pct >= trail_70_threshold:
            profit_amount = current_price - entry_price
            potential_stop = entry_price + (profit_amount * trail_70_pct)
            if potential_stop > current_stop:
                return {
                    'updated': True,
                    'new_stop': potential_stop,
                    'reason': f'Trailing 70% of profit at {profit_pct*100:.1f}%'
                }
        
        return {'updated': False, 'new_stop': current_stop, 'reason': 'No trailing update needed'}
    
    def _check_momentum_reversal(self, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Level 2: Momentum Reversal Detection
        
        TODO: Implement technical indicator analysis
        - RSI divergence detection
        - MACD bearish crossover
        - Volume dryup analysis
        - Rejection pattern recognition
        
        For now, this is a placeholder that returns no signals
        """
        # Placeholder implementation
        signals = {
            'rsi_divergence': False,
            'macd_crossover': False, 
            'volume_dryup': False,
            'rejection_pattern': False
        }
        
        signal_count = sum(signals.values())
        signals_required = config.get('signals_required', 2)
        
        return {
            'should_exit': signal_count >= signals_required,
            'signal_count': signal_count,
            'signals': signals,
            'confidence': 'HIGH' if signal_count >= 3 else 'MEDIUM' if signal_count == 2 else 'LOW'
        }
    
    def _check_resistance_rejection(self, symbol: str, current_price: float, 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Level 3: Resistance Rejection Analysis
        
        TODO: Implement price action analysis
        - Find swing high resistance levels
        - Detect rejection patterns (long upper wicks)
        - Analyze bearish close patterns
        
        For now, this is a placeholder that returns no signals
        """
        # Placeholder implementation
        return {
            'should_exit': False,
            'at_resistance': False,
            'reason': 'Resistance analysis not yet implemented'
        }
    
    def _check_time_decay(self, entry_time: datetime, entry_price: float, 
                         target_price: float, current_price: float, 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Level 4: Time Decay Management
        
        Exit if not making expected progress toward target within timeframe
        """
        max_hold_hours = config.get('max_hold_hours', 6)
        trigger_at_pct = config.get('trigger_at_percent', 0.75)  # 75% of hold time
        min_progress = config.get('min_progress', 0.7)  # 70% progress to target
        
        time_in_trade = (datetime.now() - entry_time).total_seconds() / 3600
        trigger_time = max_hold_hours * trigger_at_pct
        
        if time_in_trade >= trigger_time:
            target_profit_pct = (target_price - entry_price) / entry_price
            current_profit_pct = (current_price - entry_price) / entry_price
            progress_ratio = current_profit_pct / target_profit_pct if target_profit_pct > 0 else 0
            
            if progress_ratio < min_progress:
                return {
                    'should_exit': True,
                    'reason': f'Time decay: {time_in_trade:.1f}h in trade, only {progress_ratio*100:.0f}% to target'
                }
        
        return {'should_exit': False}
    
    def get_position_status(self, position: Dict[str, Any], current_price: float) -> PositionStatus:
        """Generate comprehensive position status report"""
        symbol = position['symbol']
        entry_price = position['entry_price']
        entry_time = position.get('entry_time', datetime.now())
        target = position['take_profit']
        position_size = position.get('quantity', 0)
        
        current_stop = self.trailing_stops.get(symbol, position['stop_loss'])
        profit_pct = (current_price - entry_price) / entry_price
        profit_usd = (current_price - entry_price) * position_size
        target_pct = (target - entry_price) / entry_price
        progress = profit_pct / target_pct if target_pct > 0 else 0
        time_in_trade = (datetime.now() - entry_time).total_seconds() / 3600
        
        return PositionStatus(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            target_price=target,
            current_stop=current_stop,
            profit_pct=profit_pct,
            profit_usd=profit_usd,
            progress_to_target=progress,
            time_in_trade_hours=time_in_trade,
            position_size=position_size
        )
    
    def print_position_status(self, status: PositionStatus):
        """Print formatted position status report"""
        print(f"\n{'='*60}")
        print(f"🎯 POSITION STATUS - {status.symbol}")
        print(f"{'='*60}")
        print(f"Entry Price:     ${status.entry_price:.4f}")
        print(f"Current Price:   ${status.current_price:.4f}")
        print(f"Target Price:    ${status.target_price:.4f}")
        print(f"Current Stop:    ${status.current_stop:.4f}")
        print(f"")
        print(f"Current P&L:     ${status.profit_usd:.2f} ({status.profit_pct*100:+.2f}%)")
        print(f"Progress:        {status.progress_to_target*100:.0f}% to target")
        print(f"Time in Trade:   {status.time_in_trade_hours:.1f} hours")
        print(f"Position Size:   {status.position_size:.4f} {status.symbol.split('/')[0]}")
        print(f"{'='*60}\n")

def get_default_exit_config() -> Dict[str, Any]:
    """
    Return default intelligent exit configuration
    
    Start conservative - enable only trailing stops initially
    """
    return {
        'intelligent_exits': {
            'trailing_stop': {
                'enabled': True,
                'breakeven_at': 0.01,       # Move to breakeven at 1% profit
                'trail_50_at': 0.02,        # Trail 50% at 2% profit
                'trail_50_percent': 0.5,    # Lock in 50% of gains
                'trail_70_at': 0.03,        # Trail 70% at 3% profit  
                'trail_70_percent': 0.7     # Lock in 70% of gains
            },
            'momentum_reversal': {
                'enabled': False,           # Disabled until technical analysis implemented
                'min_profit': 0.015,        # Only use if > 1.5% profit
                'signals_required': 2,      # Need 2 of 4 signals
                'check_timeframe': '15m'    # Check on 15min chart
            },
            'resistance_rejection': {
                'enabled': False,           # Disabled until price action analysis implemented
                'min_profit': 0.02,         # Only use if > 2% profit
                'rejection_threshold': 2.0  # Need strong rejection score
            },
            'time_decay': {
                'enabled': False,           # Start conservative
                'max_hold_hours': 6,        # Default 6-hour hold period
                'trigger_at_percent': 0.75, # Check after 75% of time (4.5h)
                'min_progress': 0.7,        # Exit if < 70% progress to target
                'min_profit': 0.01          # Only if > 1% profit
            }
        }
    }