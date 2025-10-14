"""
Trade Logger
🎯 Comprehensive trade tracking and performance analysis

Features:
- Individual trade logging with full context
- Performance metrics calculation
- Daily/monthly/yearly summaries
- Filter effectiveness tracking
- Risk management validation
- Export capabilities for analysis

Tracks every aspect of the "One Good Trade Per Day" system
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
import pandas as pd
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """
    Complete trade record with all context
    """
    # Basic trade info
    trade_id: str
    timestamp: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    
    # Entry details
    entry_price: float
    entry_time: str
    position_size: float
    entry_method: str  # 'LIMIT', 'MARKET'
    
    # Exit details
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None  # 'STOP_LOSS', 'PROFIT_TARGET', 'MANUAL'
    
    # P&L and risk
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None  # After costs
    pnl_percentage: Optional[float] = None
    risk_amount: float = 0.0
    reward_risk_ratio: Optional[float] = None
    
    # Account context
    account_balance_before: float = 0.0
    account_balance_after: Optional[float] = None
    
    # Filter scores (for analysis)
    regime_score: Optional[int] = None
    setup_quality: Optional[float] = None
    confluence_score: Optional[float] = None
    risk_score: Optional[int] = None
    
    # Market context
    market_volatility: Optional[float] = None
    volume_ratio: Optional[float] = None
    
    # Timing analysis
    scan_to_entry_minutes: Optional[int] = None
    entry_to_exit_hours: Optional[float] = None
    
    # Notes and tags
    notes: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        return cls(**data)


class TradeLogger:
    """
    Comprehensive trade logging and performance tracking system
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trade logger
        
        Args:
            config: Logger configuration
        """
        self.config = config
        self.logger_config = config.get('trade_logger', {})
        
        # File paths
        self.trades_file = self.logger_config.get('trades_file', 'data/trades.json')
        self.daily_log_file = self.logger_config.get('daily_log_file', 'data/daily_logs.json')
        self.performance_file = self.logger_config.get('performance_file', 'data/performance.json')
        
        # Logging settings
        self.log_level = self.logger_config.get('log_level', 'INFO')
        self.auto_backup = self.logger_config.get('auto_backup', True)
        self.max_backup_files = self.logger_config.get('max_backup_files', 10)
        
        # Load existing data
        self.trades: List[TradeRecord] = []
        self.daily_logs: List[Dict] = []
        self._load_existing_data()
        
        logger.info("📝 Trade Logger initialized")
        logger.info(f"   Existing trades: {len(self.trades)}")
        logger.info(f"   Trades file: {self.trades_file}")
    
    def log_daily_scan(self, scan_date: str, workflow_result: Dict) -> None:
        """
        Log daily workflow scan results
        
        Args:
            scan_date: Date of scan
            workflow_result: Complete workflow result
        """
        daily_log = {
            'date': scan_date,
            'timestamp': datetime.utcnow().isoformat(),
            'workflow_status': workflow_result.get('workflow_status'),
            'decision': workflow_result.get('message', ''),
            'filter_results': workflow_result.get('filter_results', {}),
            'duration_seconds': workflow_result.get('duration_seconds', 0)
        }
        
        # Extract filter statistics
        filter_results = workflow_result.get('filter_results', {})
        
        if 'regime' in filter_results:
            regime_summary = filter_results['regime'].get('summary', {})
            daily_log['regime_tradeable'] = regime_summary.get('tradeable_count', 0)
            daily_log['regime_total'] = regime_summary.get('total_checked', 0)
        
        if 'setup' in filter_results:
            setup_summary = filter_results['setup'].get('summary', {})
            daily_log['setups_found'] = setup_summary.get('setups_count', 0)
            
        if 'confluence' in filter_results:
            confluence_summary = filter_results['confluence'].get('summary', {})
            daily_log['confluence_approved'] = confluence_summary.get('approved_count', 0)
        
        if 'risk' in filter_results:
            risk_summary = filter_results['risk'].get('summary', {})
            daily_log['risk_approved'] = risk_summary.get('final_approved_count', 0)
        
        # Add to daily logs
        self.daily_logs.append(daily_log)
        self._save_daily_logs()
        
        # Log summary
        status = workflow_result.get('workflow_status', 'UNKNOWN')
        if status == 'TRADE_READY':
            symbol = workflow_result.get('the_one_trade', 'Unknown')
            direction = workflow_result.get('trade_direction', 'Unknown')
            logger.info(f"📋 Daily scan logged: {scan_date} - TRADE READY ({symbol} {direction})")
        else:
            logger.info(f"📋 Daily scan logged: {scan_date} - {workflow_result.get('message', status)}")
    
    def start_trade(self, symbol: str, direction: str, entry_plan: Dict, 
                   workflow_result: Dict, account_balance: float) -> str:
        """
        Start tracking a new trade
        
        Returns:
            Trade ID
        """
        trade_id = f"TRADE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract filter scores from workflow result
        filter_results = workflow_result.get('filter_results', {})
        
        trade_record = TradeRecord(
            trade_id=trade_id,
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            direction=direction,
            entry_price=entry_plan.get('entry_price', 0),
            entry_time=datetime.utcnow().isoformat(),
            position_size=entry_plan.get('position_size', 0),
            entry_method='LIMIT',  # Default for our system
            risk_amount=entry_plan.get('risk_amount', 0),
            account_balance_before=account_balance,
            regime_score=self._extract_regime_score(filter_results),
            setup_quality=self._extract_setup_quality(filter_results),
            confluence_score=self._extract_confluence_score(filter_results),
            risk_score=self._extract_risk_score(filter_results),
            scan_to_entry_minutes=0  # Will be updated when actually filled
        )
        
        self.trades.append(trade_record)
        self._save_trades()
        
        logger.info(f"📈 Trade started: {trade_id} - {symbol} {direction}")
        logger.info(f"   Entry: ${entry_plan.get('entry_price', 0):,.4f}")
        logger.info(f"   Size: {entry_plan.get('position_size', 0):.2f}")
        logger.info(f"   Risk: ${entry_plan.get('risk_amount', 0):,.0f}")
        
        return trade_id
    
    def update_trade_entry(self, trade_id: str, actual_entry_price: float, 
                          entry_time: str, entry_method: str = 'LIMIT') -> None:
        """
        Update trade with actual entry details
        """
        trade = self._find_trade(trade_id)
        if not trade:
            logger.error(f"Trade not found for update: {trade_id}")
            return
        
        trade.entry_price = actual_entry_price
        trade.entry_time = entry_time
        trade.entry_method = entry_method
        
        self._save_trades()
        logger.info(f"📊 Trade entry updated: {trade_id} @ ${actual_entry_price:,.4f}")
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str,
                   account_balance_after: float) -> Dict:
        """
        Close trade and calculate final metrics
        
        Returns:
            Trade performance summary
        """
        trade = self._find_trade(trade_id)
        if not trade:
            logger.error(f"Trade not found for closing: {trade_id}")
            return {}
        
        # Update exit details
        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow().isoformat()
        trade.exit_reason = exit_reason
        trade.account_balance_after = account_balance_after
        
        # Calculate P&L
        if trade.direction == 'LONG':
            trade.gross_pnl = (exit_price - trade.entry_price) * trade.position_size
        else:  # SHORT
            trade.gross_pnl = (trade.entry_price - exit_price) * trade.position_size
        
        # Estimate costs (commission + slippage)
        position_value = trade.position_size * trade.entry_price
        estimated_costs = position_value * 0.002  # 0.2% total costs
        trade.net_pnl = trade.gross_pnl - estimated_costs
        
        # Calculate percentages
        trade.pnl_percentage = trade.net_pnl / (trade.position_size * trade.entry_price)
        
        # Calculate risk/reward ratio
        if trade.risk_amount > 0:
            trade.reward_risk_ratio = trade.net_pnl / trade.risk_amount
        
        # Calculate timing
        if trade.entry_time and trade.exit_time:
            entry_dt = datetime.fromisoformat(trade.entry_time.replace('Z', ''))
            exit_dt = datetime.fromisoformat(trade.exit_time.replace('Z', ''))
            trade.entry_to_exit_hours = (exit_dt - entry_dt).total_seconds() / 3600
        
        self._save_trades()
        
        # Log trade close
        pnl_color = "🟢" if trade.net_pnl >= 0 else "🔴"
        logger.info(f"🔒 Trade closed: {trade_id}")
        logger.info(f"   {pnl_color} P&L: ${trade.net_pnl:+,.0f} ({trade.pnl_percentage:+.1%})")
        logger.info(f"   Reason: {exit_reason}")
        logger.info(f"   Hold time: {trade.entry_to_exit_hours:.1f} hours")
        
        return self._create_trade_summary(trade)
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """
        Get performance summary for specified period
        
        Args:
            days: Number of days to include (0 = all time)
            
        Returns:
            Performance summary dict
        """
        # Filter trades by date
        if days > 0:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            filtered_trades = [
                t for t in self.trades 
                if t.exit_time and datetime.fromisoformat(t.exit_time.replace('Z', '')) >= cutoff_date
            ]
        else:
            filtered_trades = [t for t in self.trades if t.exit_time]  # Only closed trades
        
        if not filtered_trades:
            return {'period_days': days, 'total_trades': 0, 'message': 'No completed trades in period'}
        
        # Calculate basic statistics
        total_trades = len(filtered_trades)
        winning_trades = len([t for t in filtered_trades if t.net_pnl > 0])
        losing_trades = len([t for t in filtered_trades if t.net_pnl < 0])
        
        # P&L statistics
        total_pnl = sum(t.net_pnl for t in filtered_trades)
        wins = [t.net_pnl for t in filtered_trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in filtered_trades if t.net_pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Risk metrics
        total_risk = sum(t.risk_amount for t in filtered_trades)
        avg_risk = total_risk / total_trades if total_trades > 0 else 0
        
        # Ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')
        avg_reward_risk = sum(t.reward_risk_ratio for t in filtered_trades if t.reward_risk_ratio) / total_trades
        
        # Filter effectiveness (from closed trades)
        filter_stats = self._calculate_filter_effectiveness(filtered_trades)
        
        return {
            'period_days': days,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_risk_per_trade': avg_risk,
            'avg_reward_risk_ratio': avg_reward_risk,
            'filter_effectiveness': filter_stats,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def export_trades_to_csv(self, filename: str, days: int = 0) -> None:
        """
        Export trades to CSV for external analysis
        
        Args:
            filename: Output CSV filename
            days: Days to include (0 = all)
        """
        try:
            # Filter trades
            if days > 0:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                trades_to_export = [
                    t for t in self.trades
                    if datetime.fromisoformat(t.timestamp.replace('Z', '')) >= cutoff_date
                ]
            else:
                trades_to_export = self.trades
            
            if not trades_to_export:
                logger.warning(f"No trades to export for {days} days")
                return
            
            # Convert to DataFrame
            trades_data = [trade.to_dict() for trade in trades_to_export]
            df = pd.DataFrame(trades_data)
            
            # Save to CSV
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename, index=False)
            
            logger.info(f"📁 Exported {len(trades_to_export)} trades to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export trades: {str(e)}")
    
    def _find_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Find trade by ID"""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None
    
    def _extract_regime_score(self, filter_results: Dict) -> int:
        """Extract regime filter score from results"""
        return filter_results.get('regime', {}).get('summary', {}).get('tradeable_count', 0)
    
    def _extract_setup_quality(self, filter_results: Dict) -> float:
        """Extract setup quality score from results"""
        setup_results = filter_results.get('setup', {}).get('results', {})
        if setup_results:
            first_setup = next(iter(setup_results.values()), {})
            return first_setup.get('quality_score', 0.0)
        return 0.0
    
    def _extract_confluence_score(self, filter_results: Dict) -> float:
        """Extract confluence score from results"""
        confluence_results = filter_results.get('confluence', {}).get('results', {})
        if confluence_results:
            first_confluence = next(iter(confluence_results.values()), {})
            return first_confluence.get('weighted_score', 0.0)
        return 0.0
    
    def _extract_risk_score(self, filter_results: Dict) -> int:
        """Extract risk score from results"""
        return filter_results.get('risk', {}).get('summary', {}).get('final_approved_count', 0)
    
    def _calculate_filter_effectiveness(self, trades: List[TradeRecord]) -> Dict:
        """Calculate filter effectiveness statistics"""
        if not trades:
            return {}
        
        # Correlate filter scores with trade outcomes
        high_regime_wins = len([t for t in trades if t.regime_score and t.regime_score > 3 and t.net_pnl > 0])
        high_setup_wins = len([t for t in trades if t.setup_quality and t.setup_quality > 7 and t.net_pnl > 0])
        high_confluence_wins = len([t for t in trades if t.confluence_score and t.confluence_score > 7 and t.net_pnl > 0])
        
        total_high_regime = len([t for t in trades if t.regime_score and t.regime_score > 3])
        total_high_setup = len([t for t in trades if t.setup_quality and t.setup_quality > 7])
        total_high_confluence = len([t for t in trades if t.confluence_score and t.confluence_score > 7])
        
        return {
            'high_regime_win_rate': high_regime_wins / total_high_regime if total_high_regime > 0 else 0,
            'high_setup_win_rate': high_setup_wins / total_high_setup if total_high_setup > 0 else 0,
            'high_confluence_win_rate': high_confluence_wins / total_high_confluence if total_high_confluence > 0 else 0
        }
    
    def _create_trade_summary(self, trade: TradeRecord) -> Dict:
        """Create trade summary dict"""
        return {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'direction': trade.direction,
            'net_pnl': trade.net_pnl,
            'pnl_percentage': trade.pnl_percentage,
            'reward_risk_ratio': trade.reward_risk_ratio,
            'hold_time_hours': trade.entry_to_exit_hours,
            'exit_reason': trade.exit_reason
        }
    
    def _load_existing_data(self) -> None:
        """Load existing trades and logs"""
        try:
            # Load trades
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    trades_data = json.load(f)
                    self.trades = [TradeRecord.from_dict(t) for t in trades_data]
            
            # Load daily logs  
            if os.path.exists(self.daily_log_file):
                with open(self.daily_log_file, 'r') as f:
                    self.daily_logs = json.load(f)
                    
        except Exception as e:
            logger.warning(f"Failed to load existing trade data: {str(e)}")
            self.trades = []
            self.daily_logs = []
    
    def _save_trades(self) -> None:
        """Save trades to file"""
        try:
            os.makedirs(os.path.dirname(self.trades_file), exist_ok=True)
            
            with open(self.trades_file, 'w') as f:
                json.dump([t.to_dict() for t in self.trades], f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save trades: {str(e)}")
    
    def _save_daily_logs(self) -> None:
        """Save daily logs to file"""
        try:
            os.makedirs(os.path.dirname(self.daily_log_file), exist_ok=True)
            
            with open(self.daily_log_file, 'w') as f:
                json.dump(self.daily_logs, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save daily logs: {str(e)}")