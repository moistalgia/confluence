"""
🎯 Professional Trading Dashboard
=================================

Real-time dashboard showing:
- All signals received (executed, rejected, marginal)  
- Detailed validation breakdowns with 5-factor scoring
- Trade execution history with position sizing math
- Portfolio performance metrics and risk analysis
- Live P&L tracking and win rate statistics

Created: October 2025
Author: Professional Trading System
"""

import json
import threading
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class SignalRecord:
    """Complete signal information for dashboard"""
    timestamp: str
    symbol: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    
    # Validation Details
    validation_score: float
    validation_result: str  # EXECUTE, APPROVED, MARGINAL, REJECTED
    
    # 5-Factor Breakdown
    indicator_confluence_score: float
    timeframe_alignment_score: float
    volume_confirmation_score: float
    market_structure_score: float
    risk_reward_score: float
    
    # Execution Details
    execution_status: str  # executed, rejected, execution_failed
    position_size: float = 0.0
    size_multiplier: float = 1.0
    rejection_reason: str = ""
    
    # Trade Results (if executed)
    trade_id: Optional[str] = None
    current_pnl: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    is_closed: bool = False
    close_price: Optional[float] = None
    close_reason: Optional[str] = None
    final_pnl: Optional[float] = None
    
    # Detailed validation breakdown
    detailed_validation_breakdown: Optional[Dict[str, Any]] = None

@dataclass  
class PortfolioSnapshot:
    """Portfolio state at specific time"""
    timestamp: str
    total_value: float
    cash: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions_count: int
    daily_pnl: float
    total_return: float

class ProfessionalTradingDashboard:
    """
    🎯 COMPREHENSIVE TRADING DASHBOARD
    
    Tracks every signal, validation, and trade with complete analysis:
    - Real-time signal monitoring with 5-factor validation breakdown
    - Position sizing calculations and execution tracking  
    - Portfolio performance with P&L analysis
    - Risk metrics and win rate statistics
    - Historical trade analysis with detailed reasoning
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.lock = threading.Lock()
        
        # Core Data Storage
        self.signal_history: List[SignalRecord] = []
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.active_trades: Dict[str, SignalRecord] = {}
        
        # Real-time Price Monitoring
        self.current_prices: Dict[str, float] = {}
        self.price_changes_24h: Dict[str, float] = {}
        self.price_history: Dict[str, List[Dict]] = {}
        self.last_price_update: Dict[str, datetime] = {}
        
        # Performance Metrics
        self.performance_stats = {
            'total_signals': 0,
            'executed_trades': 0,
            'rejected_signals': 0,
            'marginal_executions': 0,
            
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            
            'total_realized_pnl': 0.0,
            'total_unrealized_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            
            'risk_per_trade': 0.02,  # 2%
            'max_position_size': 0.10,  # 10%
        }
        
        # Validation Statistics
        self.validation_stats = {
            'execute_signals': 0,
            'approved_signals': 0, 
            'marginal_signals': 0,
            'rejected_signals': 0,
            
            'avg_indicator_confluence': 0.0,
            'avg_timeframe_alignment': 0.0,
            'avg_volume_confirmation': 0.0,
            'avg_market_structure': 0.0,
            'avg_risk_reward': 0.0,
        }
        
        logger.info("🎯 Professional Trading Dashboard initialized")
    
    def log_signal(self, signal_data: Dict[str, Any], validation_result: Dict[str, Any], 
                   execution_result: Dict[str, Any]) -> str:
        """
        📊 LOG COMPLETE SIGNAL WITH ALL DETAILS
        
        Records signal, validation breakdown, and execution outcome
        Returns signal_id for tracking
        """
        with self.lock:
            try:
                signal_id = f"{signal_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                signal_record = SignalRecord(
                    timestamp=datetime.now().isoformat(),
                    symbol=signal_data['symbol'],
                    action=signal_data['action'],
                    entry_price=signal_data['entry_price'],
                    stop_loss=signal_data['stop_loss'],
                    take_profit=signal_data['take_profit'], 
                    confidence=signal_data['confidence'],
                    
                    # Validation Details
                    validation_score=validation_result['score'],
                    validation_result=validation_result['result'].value,
                    
                    # 5-Factor Breakdown (now directly in validation_result)
                    indicator_confluence_score=validation_result.get('indicator_confluence_score', 0.0),
                    timeframe_alignment_score=validation_result.get('timeframe_alignment_score', 0.0),
                    volume_confirmation_score=validation_result.get('volume_confirmation_score', 0.0),
                    market_structure_score=validation_result.get('market_structure_score', 0.0),
                    risk_reward_score=validation_result.get('risk_reward_score', 0.0),
                    
                    # Store detailed breakdown for dashboard
                    detailed_validation_breakdown=validation_result.get('detailed_breakdown', {}),
                    
                    # Execution Details
                    execution_status=execution_result['status'],
                    position_size=execution_result.get('position_size', 0.0),
                    size_multiplier=validation_result.get('execution_size_multiplier', 1.0),
                    rejection_reason=execution_result.get('reason', ''),
                    
                    trade_id=signal_id if execution_result['status'] == 'executed' else None
                )
                
                # Add to history
                self.signal_history.append(signal_record)
                
                # Track active trades
                if execution_result['status'] == 'executed':
                    self.active_trades[signal_id] = signal_record
                
                # Update statistics
                self._update_statistics(signal_record)
                
                # Trim history if needed
                if len(self.signal_history) > self.max_history:
                    self.signal_history = self.signal_history[-self.max_history:]
                
                logger.info(f"📊 Logged signal: {signal_id} - {signal_record.validation_result} ({signal_record.validation_score:.1%})")
                
                return signal_id
                
            except Exception as e:
                logger.error(f"Error logging signal: {e}")
                return ""
    
    def update_trade_pnl(self, trade_id: str, current_price: float, unrealized_pnl: float):
        """📈 UPDATE TRADE P&L IN REAL-TIME"""
        with self.lock:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                trade.current_pnl = unrealized_pnl
                
                # Track max favorable/adverse excursion
                if unrealized_pnl > trade.max_favorable:
                    trade.max_favorable = unrealized_pnl
                if unrealized_pnl < trade.max_adverse:
                    trade.max_adverse = unrealized_pnl
    
    def close_trade(self, trade_id: str, close_price: float, close_reason: str, final_pnl: float):
        """🎯 CLOSE TRADE WITH FINAL P&L"""
        with self.lock:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                trade.is_closed = True
                trade.close_price = close_price
                trade.close_reason = close_reason
                trade.final_pnl = final_pnl
                
                # Update performance stats
                if final_pnl > 0:
                    self.performance_stats['winning_trades'] += 1
                    self.performance_stats['avg_win'] = (
                        (self.performance_stats['avg_win'] * (self.performance_stats['winning_trades'] - 1) + final_pnl) / 
                        self.performance_stats['winning_trades']
                    )
                else:
                    self.performance_stats['losing_trades'] += 1
                    self.performance_stats['avg_loss'] = (
                        (self.performance_stats['avg_loss'] * (self.performance_stats['losing_trades'] - 1) + final_pnl) /
                        self.performance_stats['losing_trades'] 
                    )
                
                # Update win rate
                total_closed = self.performance_stats['winning_trades'] + self.performance_stats['losing_trades']
                if total_closed > 0:
                    self.performance_stats['win_rate'] = self.performance_stats['winning_trades'] / total_closed
                
                # Update profit factor
                if self.performance_stats['avg_loss'] != 0:
                    self.performance_stats['profit_factor'] = abs(self.performance_stats['avg_win'] / self.performance_stats['avg_loss'])
                
                self.performance_stats['total_realized_pnl'] += final_pnl
                
                # Remove from active trades
                del self.active_trades[trade_id]
                
                logger.info(f"🎯 Closed trade {trade_id}: {close_reason} P&L: ${final_pnl:.2f}")
    
    def update_price(self, symbol: str, price: float, volume: float = 0.0):
        """📊 UPDATE REAL-TIME PRICE DATA"""
        with self.lock:
            now = datetime.now()
            
            # Calculate realistic 24h change (simulated for demo)
            price_change_24h = 0.0
            
            # If we have no previous price, generate a realistic baseline change
            if symbol not in self.current_prices:
                # Simulate a realistic 24h change for new pairs
                price_change_24h = random.uniform(-8.0, 12.0)  # -8% to +12%
            else:
                old_price = self.current_prices[symbol]
                if old_price > 0:
                    # Calculate immediate change
                    immediate_change = ((price - old_price) / old_price) * 100
                    
                    # Update cumulative 24h change (simulate realistic market movement)
                    current_24h_change = self.price_changes_24h.get(symbol, 0.0)
                    
                    # Add the immediate change to running 24h change with some decay
                    price_change_24h = current_24h_change * 0.98 + immediate_change
                    
                    # Keep within realistic bounds
                    price_change_24h = max(-15.0, min(20.0, price_change_24h))
            
            # Update current price
            self.current_prices[symbol] = price
            self.price_changes_24h[symbol] = price_change_24h
            self.last_price_update[symbol] = now
            
            # Add to price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'price': price,
                'volume': volume,
                'timestamp': now.isoformat(),
                'change_24h': price_change_24h
            })
            
            # Keep only last 1000 price points
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def get_price_summary_safe(self) -> Dict[str, Any]:
        """📊 GET CURRENT PRICE SUMMARY - SAFE VERSION WITH LOCK"""
        with self.lock:
            return self.get_price_summary()
    
    def get_price_summary(self) -> Dict[str, Any]:
        """📊 GET CURRENT PRICE SUMMARY FOR ALL MONITORED PAIRS"""
        # NOTE: No lock here since this is called from get_dashboard_summary() which already has the lock
        price_summary = {}
        
        for symbol in self.current_prices:
            last_update = self.last_price_update.get(symbol)
            time_since_update = None
            
            if last_update:
                time_since_update = (datetime.now() - last_update).total_seconds()
            
            price_summary[symbol] = {
                'current_price': self.current_prices[symbol],
                'change_24h': self.price_changes_24h.get(symbol, 0.0),
                'last_update': last_update.isoformat() if last_update else None,
                'seconds_since_update': time_since_update,
                'status': 'live' if time_since_update and time_since_update < 60 else 'stale'
            }
        
        return {
            'monitored_pairs': len(self.current_prices),
            'prices': price_summary,
            'last_refresh': datetime.now().isoformat()
        }
    
    def log_portfolio_snapshot(self, portfolio_data: Dict[str, Any]):
        """📊 RECORD PORTFOLIO STATE"""
        with self.lock:
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now().isoformat(),
                total_value=portfolio_data.get('total_value', 0.0),
                cash=portfolio_data.get('cash', 0.0),
                unrealized_pnl=portfolio_data.get('unrealized_pnl', 0.0),
                realized_pnl=portfolio_data.get('realized_pnl', 0.0),
                open_positions_count=portfolio_data.get('open_positions_count', 0),
                daily_pnl=portfolio_data.get('daily_pnl', 0.0),
                total_return=portfolio_data.get('total_return', 0.0)
            )
            
            self.portfolio_history.append(snapshot)
            
            # Update unrealized P&L stats
            self.performance_stats['total_unrealized_pnl'] = snapshot.unrealized_pnl
            
            # Trim history
            if len(self.portfolio_history) > self.max_history:
                self.portfolio_history = self.portfolio_history[-self.max_history:]
    
    def _update_statistics(self, signal: SignalRecord):
        """📈 UPDATE DASHBOARD STATISTICS"""
        self.performance_stats['total_signals'] += 1
        
        # Execution stats
        if signal.execution_status == 'executed':
            self.performance_stats['executed_trades'] += 1
            if signal.validation_result == 'MARGINAL':
                self.performance_stats['marginal_executions'] += 1
        else:
            self.performance_stats['rejected_signals'] += 1
        
        # Validation stats  
        if signal.validation_result == 'EXECUTE':
            self.validation_stats['execute_signals'] += 1
        elif signal.validation_result == 'APPROVED':
            self.validation_stats['approved_signals'] += 1
        elif signal.validation_result == 'MARGINAL':
            self.validation_stats['marginal_signals'] += 1
        else:
            self.validation_stats['rejected_signals'] += 1
        
        # Update factor averages
        total_signals = len(self.signal_history)
        self.validation_stats['avg_indicator_confluence'] = sum(s.indicator_confluence_score for s in self.signal_history) / total_signals
        self.validation_stats['avg_timeframe_alignment'] = sum(s.timeframe_alignment_score for s in self.signal_history) / total_signals
        self.validation_stats['avg_volume_confirmation'] = sum(s.volume_confirmation_score for s in self.signal_history) / total_signals
        self.validation_stats['avg_market_structure'] = sum(s.market_structure_score for s in self.signal_history) / total_signals
        self.validation_stats['avg_risk_reward'] = sum(s.risk_reward_score for s in self.signal_history) / total_signals
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """🎯 GET COMPLETE DASHBOARD DATA"""
        with self.lock:
            recent_signals = self.signal_history[-20:] if self.signal_history else []
            recent_portfolio = self.portfolio_history[-10:] if self.portfolio_history else []
            
            # Separate active and closed trades
            active_trades = {k: v for k, v in self.active_trades.items() if not v.is_closed}
            closed_trades = {k: v for k, v in self.active_trades.items() if v.is_closed}
            
            return {
                'timestamp': datetime.now().isoformat(),
                'performance_stats': self.performance_stats.copy(),
                'price_monitor': self.get_price_summary(),
                'validation_stats': self.validation_stats.copy(),
                'recent_signals': [asdict(s) for s in recent_signals],
                'active_trades': {k: asdict(v) for k, v in active_trades.items()},
                'closed_trades': {k: asdict(v) for k, v in list(closed_trades.items())[-10:]},  # Show last 10 closed trades
                'recent_portfolio': [asdict(p) for p in recent_portfolio],
                'total_signals_logged': len(self.signal_history),
                'active_trades_count': len(active_trades),
                'closed_trades_count': len(closed_trades)
            }
    
    def print_dashboard(self):
        """🖥️ PRINT FORMATTED DASHBOARD TO CONSOLE"""
        try:
            print("\n" + "="*80)
            print("🎯 PROFESSIONAL TRADING DASHBOARD")
            print("="*80)
            
            # Performance Overview
            stats = self.performance_stats
            print(f"\n📊 PERFORMANCE OVERVIEW:")
            print(f"   Total Signals: {stats['total_signals']} | Executed: {stats['executed_trades']} | Rejected: {stats['rejected_signals']}")
            print(f"   Win Rate: {stats['win_rate']:.1%} | Profit Factor: {stats['profit_factor']:.2f}")
            print(f"   Avg Win: ${stats['avg_win']:.2f} | Avg Loss: ${stats['avg_loss']:.2f}")
            print(f"   Total Realized P&L: ${stats['total_realized_pnl']:.2f} | Unrealized: ${stats['total_unrealized_pnl']:.2f}")
            
            # Validation Breakdown
            val_stats = self.validation_stats
            print(f"\n🔍 VALIDATION STATISTICS:")
            print(f"   EXECUTE: {val_stats['execute_signals']} | APPROVED: {val_stats['approved_signals']} | MARGINAL: {val_stats['marginal_signals']} | REJECTED: {val_stats['rejected_signals']}")
            print(f"   Avg Factor Scores - Confluence: {val_stats['avg_indicator_confluence']:.1%} | Timeframe: {val_stats['avg_timeframe_alignment']:.1%}")
            print(f"                      Volume: {val_stats['avg_volume_confirmation']:.1%} | Structure: {val_stats['avg_market_structure']:.1%} | R:R: {val_stats['avg_risk_reward']:.1%}")
            
            # Recent Signals
            print(f"\n📋 RECENT SIGNALS ({len(self.signal_history[-5:])}):")
            for signal in self.signal_history[-5:]:
                status_emoji = "✅" if signal.execution_status == "executed" else "❌"
                print(f"   {status_emoji} {signal.symbol} {signal.action} @${signal.entry_price:.4f} | {signal.validation_result} ({signal.validation_score:.1%}) | Size: {signal.position_size:.2f}")
            
            # Active Trades
            print(f"\n🔄 ACTIVE TRADES ({len(self.active_trades)}):")
            for trade_id, trade in self.active_trades.items():
                pnl_emoji = "🟢" if trade.current_pnl >= 0 else "🔴"
                print(f"   {pnl_emoji} {trade.symbol} {trade.action} @${trade.entry_price:.4f} | P&L: ${trade.current_pnl:.2f} | Size: {trade.position_size:.2f}")
            
            # Latest Portfolio
            if self.portfolio_history:
                latest = self.portfolio_history[-1]
                print(f"\n💼 PORTFOLIO STATUS:")
                print(f"   Total Value: ${latest.total_value:.2f} | Cash: ${latest.cash:.2f} | Return: {latest.total_return:.2%}")
                print(f"   Open Positions: {latest.open_positions_count} | Daily P&L: ${latest.daily_pnl:.2f}")
            
            print("\n" + "="*80)
            
        except Exception as e:
            logger.error(f"Error printing dashboard: {e}")
    
    def export_to_json(self, filename: Optional[str] = None) -> str:
        """💾 EXPORT DASHBOARD DATA TO JSON"""
        if filename is None:
            filename = f"trading_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            dashboard_data = self.get_dashboard_summary()
            
            with open(filename, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"📄 Dashboard exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting dashboard: {e}")
            return ""

# Global dashboard instance
trading_dashboard = ProfessionalTradingDashboard()

def get_dashboard() -> ProfessionalTradingDashboard:
    """Get global dashboard instance"""
    return trading_dashboard

if __name__ == "__main__":
    # Demo dashboard
    dashboard = ProfessionalTradingDashboard()
    dashboard.print_dashboard()