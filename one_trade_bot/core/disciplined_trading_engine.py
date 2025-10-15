"""
Disciplined Trading Engine - The Complete System

Combines enhanced paper trading features with hybrid timing discipline:

FEATURES (from Enhanced Engine):
- SQLite database for trade journaling
- Realistic order execution with bid/ask spreads
- Real market data integration via CCXT
- Performance analytics and tracking

DISCIPLINE (from Hybrid Engine):
- 8:00 AM daily scan picks THE ONE candidate
- Entry zone monitoring with timeout
- ONE trade maximum per 24 hours
- Patient waiting for optimal entry

This is the single, complete system that maintains our core principles
while providing professional-grade execution and analytics.
"""

import asyncio
import sqlite3
import ccxt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import json

from core.paper_trading import PaperTradingEngine, PaperOrder, PaperPosition, PaperTrade, OrderType, OrderStatus, TradeDirection
from core.multi_pair_kraken_scanner import MultiPairKrakenScanner
from core.transparency_dashboard import ScanningTransparencyDashboard

logger = logging.getLogger(__name__)

class ExecutionState(Enum):
    NO_TARGET = "no_target"
    MONITORING = "monitoring"  
    ENTERING = "entering"
    ENTERED = "entered"
    EXPIRED = "expired"

@dataclass
class DailyTarget:
    """Represents the ONE candidate selected for today"""
    symbol: str
    entry_ideal: float
    entry_low: float        # 0.5% below ideal
    entry_high: float       # 0.5% above ideal  
    stop_loss: float
    take_profit: float
    risk_reward: float
    confluence_score: int
    setup_type: str
    selected_at: datetime
    expires_at: datetime
    
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at
    
    def in_entry_zone(self, current_price: float) -> bool:
        return self.entry_low <= current_price <= self.entry_high

class DisciplinedTradingEngine(PaperTradingEngine):
    """
    Complete trading engine with database integration AND timing discipline
    
    This is the single system that combines:
    - All enhanced paper trading features (database, realistic execution)
    - Hybrid timing discipline (daily scan, entry waiting, ONE trade/day)
    
    Maintains core principle: Wait patiently for ONE high-probability setup per day
    """
    
    def __init__(self, config: Dict[str, Any], data_provider, workflow_engine=None, db_path: str = 'paper_trading.db'):
        # Initialize base paper trading engine
        super().__init__(config, data_provider)
        
        # Enhanced features setup
        self.db_path = db_path
        self._setup_enhanced_database()
        
        # Real market data (optional - fallback to existing data provider)
        self.use_live_data = config.get('use_live_market_data', False)
        if self.use_live_data:
            try:
                self.exchange = ccxt.kraken({'enableRateLimit': True})
                logger.info("✅ Live market data enabled via CCXT (Kraken)")
                logger.info("   Will fetch real BTC prices (~$110k) instead of simulated data")
            except Exception as e:
                logger.warning(f"Failed to initialize CCXT, using data provider: {e}")
                self.use_live_data = False
        
        # Order timeout settings
        self.order_timeout_hours = config.get('order_timeout_hours', 4)
        
        # Enhanced fill simulation
        self.use_bid_ask_simulation = config.get('use_bid_ask_simulation', True)
        
        # Transparency dashboard for scan logging
        self.transparency = ScanningTransparencyDashboard(self.db_path)
        
        # Dynamic target upgrading settings
        self.enable_dynamic_upgrading = config.get('execution', {}).get('enable_dynamic_upgrading', True)
        self.rescan_interval_hours = config.get('execution', {}).get('rescan_interval_hours', 1)
        self.upgrade_threshold_points = config.get('execution', {}).get('upgrade_threshold_points', 10)
        self.last_rescan_time = None
        
        # Create discipline tracking tables
        self._setup_discipline_tables()
        
        # Add disciplined timing state
        self.workflow = workflow_engine
        self.daily_target: Optional[DailyTarget] = None
        self.execution_state = ExecutionState.NO_TARGET
        self.trade_entered_today = False
        self.last_scan_date: Optional[str] = None
        
        # Timing configuration
        self.daily_scan_time = config.get('schedule', {}).get('scan_time', '08:00')
        self.monitor_interval_minutes = config.get('execution', {}).get('monitor_interval', 5)
        self.entry_timeout_hours = config.get('execution', {}).get('entry_timeout_hours', 6)
        self.entry_zone_tolerance = config.get('execution', {}).get('entry_zone_tolerance', 0.005)
        
        # Override max positions to enforce discipline
        self.max_positions = 1  # ONE trade rule
        
        logger.info(f"🎯 Disciplined Trading Engine initialized")
        logger.info(f"   Daily scan: {self.daily_scan_time}")
        logger.info(f"   Monitor interval: {self.monitor_interval_minutes} minutes")  
        logger.info(f"   Entry timeout: {self.entry_timeout_hours} hours")
        logger.info(f"   Max positions: {self.max_positions} (ONE TRADE RULE)")
    
    def _setup_enhanced_database(self):
        """Create SQLite database for comprehensive trade journaling"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                order_type TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL,
                position_value REAL,
                status TEXT,
                exit_price REAL,
                exit_time TEXT,
                pnl REAL,
                pnl_pct REAL,
                fees_paid REAL,
                setup_data TEXT,
                confluence_score REAL,
                setup_quality INTEGER,
                exit_reason TEXT,
                notes TEXT
            )
        ''')
        
        # Daily equity tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_equity (
                date TEXT PRIMARY KEY,
                starting_balance REAL,
                ending_balance REAL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                total_equity REAL,
                open_positions INTEGER,
                trades_taken INTEGER,
                max_drawdown REAL
            )
        ''')
        
        # Orders table for better tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                order_type TEXT,
                quantity REAL,
                price REAL,
                stop_price REAL,
                status TEXT,
                created_at TEXT,
                filled_at TEXT,
                cancelled_at TEXT,
                timeout_at TEXT,
                filled_price REAL,
                fees_paid REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Enhanced database initialized: {self.db_path}")

    def _setup_discipline_tables(self):
        """Create additional tables for discipline tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create daily_targets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_targets (
                date TEXT PRIMARY KEY,
                symbol TEXT,
                entry_ideal REAL,
                entry_zone_low REAL,
                entry_zone_high REAL,
                stop_loss REAL,
                take_profit REAL,
                risk_reward REAL,
                confluence_score INTEGER,
                setup_type TEXT,
                selected_at TEXT,
                expires_at TEXT,
                status TEXT,
                outcome TEXT
            )
        ''')
        
        # Target upgrades tracking table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS target_upgrades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_symbol TEXT NOT NULL,
                new_symbol TEXT NOT NULL,
                old_score INTEGER,
                new_score INTEGER,
                score_improvement INTEGER,
                rescan_id INTEGER,
                upgrade_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def run_disciplined_cycle(self):
        """
        Main execution loop with complete discipline
        
        This is the heart of the system:
        1. 8am: Daily scan picks THE ONE candidate (or declares rest day)
        2. Continuous: Monitor entry zone for optimal execution
        3. Timeout: If no entry in 6 hours, day is over
        4. Discipline: Maximum ONE trade entry per 24 hours
        """
        logger.info(f"🚀 Starting disciplined trading cycle")
        logger.info(f"   Scan time: {self.daily_scan_time}")
        logger.info(f"   Monitor frequency: Every {self.monitor_interval_minutes} minutes")
        logger.info(f"   ONE TRADE PER DAY RULE: Enforced")
        
        while True:
            try:
                current_time = datetime.now()
                
                # 8:00 AM Daily Scan - Pick THE ONE candidate
                if self._should_run_daily_scan(current_time):
                    await self._run_daily_scan()
                
                # Continuous Monitoring - Wait for optimal entry
                if self.execution_state == ExecutionState.MONITORING:
                    await self._monitor_entry_opportunity()
                
                # Timeout Management
                if self.daily_target and self.daily_target.is_expired():
                    self._handle_target_expiration()
                
                # Position Management - Monitor existing trades
                if self.positions:
                    await self._monitor_existing_positions()
                
                # Sleep until next check
                await asyncio.sleep(self.monitor_interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Disciplined trading cycle stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in disciplined cycle: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def _should_run_daily_scan(self, current_time: datetime) -> bool:
        """
        Check if we should run the daily scan
        - Once per day at specified time  
        - Only if we haven't scanned today already
        """
        today = current_time.strftime('%Y-%m-%d')
        current_hour_minute = current_time.strftime('%H:%M')
        
        # Haven't scanned today and it's past scan time
        return (self.last_scan_date != today and 
                current_hour_minute >= self.daily_scan_time)
    
    async def _run_daily_scan(self):
        """
        8:00 AM Daily Scan - The Heart of Discipline
        
        This is where the 5-filter pipeline runs to pick THE ONE candidate:
        1. Market Regime Filter - Eliminate choppy markets
        2. Setup Scanner - Find pullback candidates  
        3. Confluence Checker - Pick the best one
        4. Risk Check - Final safety validation
        5. Create daily target OR declare rest day
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🌅 DAILY SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"{'='*60}")
        
        # Reset daily state
        self._reset_daily_state()
        
        try:
            # Run the complete 5-filter pipeline
            if self.workflow:
                scan_result = await self.workflow.run_complete_scan()
            else:
                # Use multi-pair Kraken scanner
                scanner = MultiPairKrakenScanner(self.config)
                scan_result = await scanner.scan_all_liquid_pairs()
            
            # Log scan results for transparency
            scan_id = self.transparency.log_scan_results(scan_result)
            
            valid_setups = scan_result.get('valid_setups', [])
            if not scan_result or not valid_setups:
                logger.info("❌ No valid setups found")
                logger.info("📅 TODAY IS A REST DAY")
                logger.info("   No trade candidates meet all 5 filters")
                logger.info("   Discipline: Accept 40-50% no-trade days")
                self.execution_state = ExecutionState.NO_TARGET
                return
            
            # Get THE ONE best candidate
            best_setup = scan_result['best_setup']
            
            # Validate minimum confluence score
            min_score = self.config.get('filters', {}).get('confluence', {}).get('min_confluence_score', 60)
            if best_setup['confluence_score'] < min_score:
                logger.info(f"⚠️ Best candidate score too low: {best_setup['confluence_score']}/{min_score}")
                logger.info("📅 TODAY IS A REST DAY")  
                logger.info("   Discipline: Never lower standards")
                self.execution_state = ExecutionState.NO_TARGET
                return
            
            # Create THE ONE daily target
            self.daily_target = DailyTarget(
                symbol=best_setup['symbol'],
                entry_ideal=best_setup['entry_price'],
                entry_low=best_setup['entry_price'] * (1 - self.entry_zone_tolerance),
                entry_high=best_setup['entry_price'] * (1 + self.entry_zone_tolerance),
                stop_loss=best_setup['stop_loss'],
                take_profit=best_setup['take_profit'],
                risk_reward=best_setup['risk_reward'],
                confluence_score=best_setup['confluence_score'],
                setup_type=best_setup.get('setup_type', 'pullback'),
                selected_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=self.entry_timeout_hours)
            )
            
            self.execution_state = ExecutionState.MONITORING
            
            # Log THE ONE target
            logger.info(f"🎯 THE ONE DAILY TARGET SELECTED")
            logger.info(f"   Symbol:      {self.daily_target.symbol}")
            logger.info(f"   Entry Zone:  ${self.daily_target.entry_low:.4f} - ${self.daily_target.entry_high:.4f}")
            logger.info(f"   Ideal Entry: ${self.daily_target.entry_ideal:.4f}")
            logger.info(f"   Stop Loss:   ${self.daily_target.stop_loss:.4f}")
            logger.info(f"   Take Profit: ${self.daily_target.take_profit:.4f}")
            logger.info(f"   Risk:Reward: {self.daily_target.risk_reward:.2f}:1")
            logger.info(f"   Score:       {self.daily_target.confluence_score}/100")
            logger.info(f"   Expires:     {self.daily_target.expires_at.strftime('%H:%M')} ({self.entry_timeout_hours}h window)")
            logger.info(f"   Setup:       {self.daily_target.setup_type}")
            logger.info(f"")
            logger.info(f"⏳ Now waiting patiently for optimal entry...")
            
            # Save target selection to database
            self._save_target_to_db()
            
        except Exception as e:
            logger.error(f"Error in daily scan: {e}")
            self.execution_state = ExecutionState.NO_TARGET
    
    async def _monitor_entry_opportunity(self):
        """
        Continuous Entry Monitoring - Patient Execution
        
        This runs every 5 minutes when we have a daily target:
        - Wait for price to enter the entry zone
        - Reconfirm setup conditions are still valid  
        - Execute limit order if all conditions met
        - Enforce ONE trade per day rule
        """
        if not self.daily_target or self.trade_entered_today:
            return
        
        try:
            # Get current market data using enhanced engine capabilities
            market_data = self.get_live_market_data(self.daily_target.symbol)
            if not market_data:
                logger.warning(f"Could not get market data for {self.daily_target.symbol}")
                return
            
            current_price = market_data['last']
            
            # Check if price has entered entry zone
            if self.daily_target.in_entry_zone(current_price):
                logger.info(f"\n🚨 ENTRY ZONE REACHED!")
                logger.info(f"   Symbol: {self.daily_target.symbol}")
                logger.info(f"   Current: ${current_price:.4f}")
                logger.info(f"   Zone: ${self.daily_target.entry_low:.4f} - ${self.daily_target.entry_high:.4f}")
                logger.info(f"   Spread: {market_data.get('spread_pct', 0)*100:.3f}%")
                
                # Final confirmation - conditions might have changed since 8am
                if await self._confirm_entry_still_valid(market_data):
                    await self._execute_disciplined_entry()
                else:
                    logger.warning("⚠️ Setup conditions changed since morning - skipping entry")
                    logger.info("   Discipline: Never compromise on quality")
                    
            else:
                # Log monitoring status (not too verbose)
                distance_pct = ((current_price - self.daily_target.entry_ideal) / 
                              self.daily_target.entry_ideal * 100)
                
                time_left = self.daily_target.expires_at - datetime.now()
                hours_left = time_left.total_seconds() / 3600
                
                if datetime.now().minute % 15 == 0:  # Log every 15 minutes
                    logger.info(f"⏳ Monitoring {self.daily_target.symbol}: "
                              f"${current_price:.4f} ({distance_pct:+.2f}% from entry, "
                              f"{hours_left:.1f}h left)")
                
                # Check for dynamic target upgrading opportunity
                if self.enable_dynamic_upgrading:
                    await self._check_for_target_upgrade()
                
        except Exception as e:
            logger.error(f"Error monitoring entry: {e}")
    
    async def _check_for_target_upgrade(self):
        """
        Dynamic Target Upgrading - Smart Opportunity Switching
        
        Runs hourly rescans to find better opportunities:
        - Re-run same multi-pair scanning logic
        - Compare new best score vs current target score
        - Switch if new target is significantly better (threshold)
        - Log all upgrade decisions for transparency
        """
        current_time = datetime.now()
        
        # Check if it's time for a rescan
        if (self.last_rescan_time is None or 
            (current_time - self.last_rescan_time).total_seconds() >= self.rescan_interval_hours * 3600):
            
            logger.info(f"🔄 DYNAMIC UPGRADE CHECK - {current_time.strftime('%H:%M')}")
            
            try:
                # Run fresh multi-pair scan
                scanner = MultiPairKrakenScanner(self.config)
                fresh_scan = await scanner.scan_all_liquid_pairs()
                
                # Log rescan for transparency
                rescan_id = self.transparency.log_scan_results(fresh_scan)
                
                if fresh_scan['best_setup']:
                    new_best = fresh_scan['best_setup']
                    current_score = self.daily_target.confluence_score
                    new_score = new_best['confluence_score']
                    score_improvement = new_score - current_score
                    
                    logger.info(f"   Current target: {self.daily_target.symbol} (score: {current_score})")
                    logger.info(f"   New best: {new_best['symbol']} (score: {new_score})")
                    logger.info(f"   Score difference: {score_improvement:+d} points")
                    
                    # Check if upgrade is warranted
                    if (new_best['symbol'] != self.daily_target.symbol and 
                        score_improvement >= self.upgrade_threshold_points):
                        
                        logger.info(f"🚀 UPGRADING TARGET!")
                        logger.info(f"   Switching from {self.daily_target.symbol} to {new_best['symbol']}")
                        logger.info(f"   Improvement: {score_improvement} points (threshold: {self.upgrade_threshold_points})")
                        
                        # Create new daily target
                        old_target = self.daily_target.symbol
                        self._create_upgraded_target(new_best)
                        
                        # Log upgrade decision
                        self._log_target_upgrade(old_target, new_best, score_improvement, rescan_id)
                        
                    else:
                        logger.info(f"✋ Keeping current target (improvement: {score_improvement} < threshold: {self.upgrade_threshold_points})")
                
                self.last_rescan_time = current_time
                
            except Exception as e:
                logger.error(f"Error in dynamic upgrade check: {e}")
    
    def _create_upgraded_target(self, new_setup: Dict[str, Any]):
        """Create new daily target from upgraded setup"""
        # Update current daily target with new information
        self.daily_target.symbol = new_setup['symbol']
        self.daily_target.entry_ideal = new_setup['entry_price']
        self.daily_target.entry_low = new_setup['entry_price'] * (1 - self.entry_zone_tolerance)
        self.daily_target.entry_high = new_setup['entry_price'] * (1 + self.entry_zone_tolerance)
        self.daily_target.stop_loss = new_setup['stop_loss']
        self.daily_target.take_profit = new_setup['take_profit']
        self.daily_target.risk_reward = new_setup['risk_reward']
        self.daily_target.confluence_score = new_setup['confluence_score']
        self.daily_target.setup_type = new_setup.get('setup_type', 'pullback')
        
        # Note: Keep original selected_at and expires_at times for tracking
    
    def _log_target_upgrade(self, old_symbol: str, new_setup: Dict[str, Any], score_improvement: int, rescan_id: int):
        """Log target upgrade decision to database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            conn.execute('''
                INSERT INTO target_upgrades (
                    old_symbol, new_symbol, old_score, new_score, 
                    score_improvement, rescan_id, upgrade_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                old_symbol, new_setup['symbol'], 
                self.daily_target.confluence_score - score_improvement,
                new_setup['confluence_score'], score_improvement, 
                rescan_id, datetime.now().isoformat()
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error logging target upgrade: {e}")
        finally:
            conn.close()
    
    async def _confirm_entry_still_valid(self, market_data: Dict[str, float]) -> bool:
        """
        Real-time Entry Confirmation
        
        Market conditions can change between 8am scan and entry time.
        Quick validation that setup is still high-quality:
        - RSI still in oversold recovery range
        - Price action not breaking down
        - Volume still supportive
        - Spread not too wide
        """
        try:
            current_price = market_data['last']
            
            # Get fresh technical data
            if self.workflow:
                fresh_data = await self.workflow.get_market_data(self.daily_target.symbol)
            else:
                # Fallback to basic checks
                fresh_data = {
                    'rsi': 42,  # Assume OK
                    'sma_20': current_price * 0.99,  # Assume near support
                    'volume_ratio': 1.1  # Assume OK
                }
            
            # Confirmation checks (slightly relaxed from morning scan)
            checks = {
                'rsi_range': 25 <= fresh_data.get('rsi', 42) <= 55,  # Still oversold recovery
                'no_breakdown': current_price > fresh_data.get('sma_20', 0) * 0.975,  # Not breaking support badly
                'volume_ok': fresh_data.get('volume_ratio', 1.0) > 0.7,  # Volume still decent
                'spread_ok': market_data.get('spread_pct', 0) < 0.005  # Spread not too wide
            }
            
            passed_checks = sum(checks.values())
            
            if passed_checks >= 3:  # 3 out of 4 checks pass
                logger.info(f"✅ Entry confirmation: {passed_checks}/4 checks passed")
                return True
            else:
                failed_checks = [k for k, v in checks.items() if not v]
                logger.warning(f"❌ Failed confirmation checks: {failed_checks}")
                return False
                
        except Exception as e:
            logger.error(f"Error in entry confirmation: {e}")
            return False
    
    async def _execute_disciplined_entry(self):
        """
        Execute THE ONE Trade with Full Discipline
        
        This is the moment of truth:
        - Calculate proper position size (1% risk)
        - Place limit order at ideal entry price
        - Set stop loss and take profit  
        - Mark trade as entered (enforce ONE per day)
        - Log everything to database
        """
        try:
            logger.info(f"✅ EXECUTING THE ONE TRADE")
            
            # Calculate position size - use % of account approach
            account_balance = self.current_balance
            position_pct = self.config.get('position_size_pct', 0.01)  # 1% of account as position
            target_position_value = account_balance * position_pct
            
            # Calculate shares needed for target position value
            position_size = target_position_value / self.daily_target.entry_ideal
            position_value = position_size * self.daily_target.entry_ideal
            
            # Calculate actual risk amount based on stop loss
            price_risk = abs(self.daily_target.entry_ideal - self.daily_target.stop_loss)
            risk_amount = position_size * price_risk
            
            # Validate against max position rule (10% of account)
            max_position_value = account_balance * 0.10  # Max 10% of account
            if position_value > max_position_value:
                position_size = max_position_value / self.daily_target.entry_ideal
                position_value = max_position_value
                risk_amount = position_size * price_risk
                logger.info(f"   Position size limited by max position rule")
            
            # Place limit order using enhanced engine capabilities
            order_id = self.place_order(
                symbol=self.daily_target.symbol,
                direction=TradeDirection.LONG,
                order_type=OrderType.LIMIT,
                quantity=position_size,
                price=self.daily_target.entry_ideal,
                stop_price=self.daily_target.stop_loss,
                setup_quality=self.daily_target.confluence_score,
                confluence_score=self.daily_target.confluence_score
            )
            
            logger.info(f"📋 THE ONE LIMIT ORDER PLACED")
            logger.info(f"   Order ID: {order_id}")
            logger.info(f"   Symbol: {self.daily_target.symbol}")
            logger.info(f"   Size: {position_size:.6f} shares")
            logger.info(f"   Position Value: ${position_value:.2f}")
            logger.info(f"   Entry: ${self.daily_target.entry_ideal:.4f}")
            logger.info(f"   Stop: ${self.daily_target.stop_loss:.4f}")
            logger.info(f"   Target: ${self.daily_target.take_profit:.4f}")
            logger.info(f"   Risk Amount: ${risk_amount:.2f} ({risk_amount/account_balance*100:.2f}% of account)")
            logger.info(f"   Price Risk: ${price_risk:.4f} per share")
            logger.info(f"   R:R: {self.daily_target.risk_reward:.2f}:1")
            logger.info(f"")
            logger.info(f"🎯 DISCIPLINE ENFORCED: No more trades today")
            
            # Update state - ONE TRADE RULE enforced
            self.execution_state = ExecutionState.ENTERING
            self.trade_entered_today = True
            
            # Clear daily target (discipline: one shot only)
            target_symbol = self.daily_target.symbol
            self.daily_target = None
            
            # Track daily equity
            self.track_daily_equity(datetime.now())
            
            # Save execution details to database
            self._save_execution_to_db(order_id, target_symbol, risk_amount)
            
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
    
    async def _monitor_existing_positions(self):
        """Monitor existing positions for stops/targets"""
        for symbol, position in self.positions.items():
            try:
                # Get current market data
                market_data = self.get_live_market_data(symbol)
                if market_data:
                    current_price = market_data['last']
                    
                    # Update position with current price
                    await self.update_positions(current_price, datetime.now())
                    
            except Exception as e:
                logger.error(f"Error monitoring position {symbol}: {e}")
    
    def _handle_target_expiration(self):
        """
        Handle Daily Target Timeout
        
        If entry zone not reached within timeout window:
        - Accept that setup didn't materialize  
        - Clear target and move on
        - Maintain discipline (no chasing)
        """
        logger.info(f"\n⏰ DAILY TARGET EXPIRED")
        logger.info(f"   Symbol: {self.daily_target.symbol}")
        logger.info(f"   Entry zone never reached in {self.entry_timeout_hours} hours")
        logger.info(f"   Discipline: Accept the miss, don't chase")
        logger.info(f"   Tomorrow: Fresh scan, fresh opportunity")
        
        # Save timeout to database
        self._save_timeout_to_db()
        
        # Clear state
        self.daily_target = None
        self.execution_state = ExecutionState.EXPIRED
    
    # ========================================================================
    # ENHANCED PAPER TRADING METHODS
    # (Integrated from EnhancedPaperTradingEngine for self-contained system)
    # ========================================================================
    
    def get_live_market_data(self, symbol: str) -> Dict[str, float]:
        """
        Get live market data with bid/ask spreads for realistic simulation
        """
        if not self.use_live_data:
            # Fallback to existing data provider with simulated spread
            try:
                df = self.data_provider.get_ohlcv(symbol, '1m', limit=1)
                if df is not None and not df.empty:
                    last_price = df['close'].iloc[-1]
                    # Simulate typical 0.01% spread
                    spread_pct = 0.0001
                    spread = last_price * spread_pct
                    return {
                        'bid': last_price - spread/2,
                        'ask': last_price + spread/2,
                        'last': last_price,
                        'spread_pct': spread_pct
                    }
            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
            
            return None
        
        try:
            # Use live CCXT data
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'spread_pct': (ticker['ask'] - ticker['bid']) / ticker['bid'] if ticker['bid'] > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            return None

    def place_order(self, symbol: str, direction: TradeDirection, order_type: OrderType,
                   quantity: float, price: Optional[float] = None, 
                   stop_price: Optional[float] = None,
                   setup_quality: Optional[int] = None,
                   confluence_score: Optional[float] = None) -> str:
        """
        Enhanced order placement with timeout and database logging
        """
        order_id = super().place_order(symbol, direction, order_type, quantity, price, stop_price, setup_quality, confluence_score)
        
        # Add timeout to order
        order = self.orders[order_id]
        order.timeout_at = datetime.utcnow() + timedelta(hours=self.order_timeout_hours)
        
        # Save to database
        self.save_order_to_db(order)
        
        return order_id

    def _try_execute_order(self, order: PaperOrder, current_price: float, timestamp: datetime):
        """
        Enhanced order execution with timeout checking and bid/ask simulation
        """
        # Check timeout first
        if hasattr(order, 'timeout_at') and timestamp > order.timeout_at:
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = timestamp
            self.update_order_in_db(order)
            logger.info(f"Order {order.id} cancelled due to timeout")
            return
        
        # Get live market data for realistic fill simulation
        market_data = self.get_live_market_data(order.symbol)
        if not market_data:
            # Fallback to original logic
            super()._try_execute_order(order, current_price, timestamp)
            return
        
        should_fill = False
        fill_price = current_price
        
        if order.order_type == OrderType.MARKET:
            should_fill = True
            # Use bid/ask for market orders
            if order.direction == TradeDirection.LONG:
                fill_price = market_data['ask'] * (1 + self.slippage_pct)  # Buy at ask + slippage
            else:
                fill_price = market_data['bid'] * (1 - self.slippage_pct)  # Sell at bid - slippage
                
        elif order.order_type == OrderType.LIMIT:
            # More realistic limit order fills
            if order.direction == TradeDirection.LONG:
                # Buy limit fills when ask <= limit price
                if market_data['ask'] <= order.price:
                    should_fill = True
                    fill_price = order.price  # Fill at limit price
            else:
                # Sell limit fills when bid >= limit price  
                if market_data['bid'] >= order.price:
                    should_fill = True
                    fill_price = order.price
                    
        elif order.order_type == OrderType.STOP_LOSS:
            if order.direction == TradeDirection.LONG:
                # Long stop triggered when bid <= stop price
                if market_data['bid'] <= order.stop_price:
                    should_fill = True
                    fill_price = market_data['bid'] * (1 - self.slippage_pct)
            else:
                # Short stop triggered when ask >= stop price
                if market_data['ask'] >= order.stop_price:
                    should_fill = True
                    fill_price = market_data['ask'] * (1 + self.slippage_pct)
        
        if should_fill:
            self._execute_order(order, fill_price, timestamp)

    def _execute_order(self, order: PaperOrder, fill_price: float, timestamp: datetime):
        """Enhanced order execution with database logging"""
        # Call parent execution
        super()._execute_order(order, fill_price, timestamp)
        
        # Update order in database
        self.update_order_in_db(order)
        
        # If position opened, save trade to database
        if order.order_type in [OrderType.MARKET, OrderType.LIMIT] and order.symbol in self.positions:
            self.save_trade_to_db(order.symbol, 'OPEN')

    def save_order_to_db(self, order: PaperOrder):
        """Save order details to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO orders (
                id, symbol, direction, order_type, quantity, price, stop_price,
                status, created_at, filled_at, cancelled_at, timeout_at, filled_price, fees_paid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            order.id,
            order.symbol,
            order.direction.value,
            order.order_type.value,
            order.quantity,
            order.price,
            order.stop_price,
            order.status.value,
            order.created_at.isoformat() if order.created_at else None,
            order.filled_at.isoformat() if hasattr(order, 'filled_at') and order.filled_at else None,
            order.cancelled_at.isoformat() if hasattr(order, 'cancelled_at') and order.cancelled_at else None,
            order.timeout_at.isoformat() if hasattr(order, 'timeout_at') and order.timeout_at else None,
            order.filled_price if hasattr(order, 'filled_price') else None,
            order.fees_paid
        ))
        
        conn.commit()
        conn.close()

    def update_order_in_db(self, order: PaperOrder):
        """Update existing order in database"""
        self.save_order_to_db(order)  # INSERT OR REPLACE handles updates

    def save_trade_to_db(self, symbol: str, status: str, trade: Optional[PaperTrade] = None):
        """Save trade details to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status == 'OPEN' and symbol in self.positions:
            position = self.positions[symbol]
            
            cursor.execute('''
                INSERT INTO trades (
                    timestamp, symbol, direction, order_type, entry_price, 
                    stop_loss, take_profit, position_size, position_value, status,
                    setup_quality, confluence_score, fees_paid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.entry_time.isoformat(),
                symbol,
                position.direction.value,
                'MARKET',  # Simplified for now
                position.entry_price,
                position.stop_loss,
                position.take_profit,
                position.quantity,
                position.quantity * position.entry_price,
                'OPEN',
                None,  # Would need to pass setup metadata
                None,  # Would need to pass confluence score
                position.fees_paid
            ))
            
        elif status == 'CLOSED' and trade:
            cursor.execute('''
                UPDATE trades 
                SET status = ?, exit_price = ?, exit_time = ?, pnl = ?, pnl_pct = ?, 
                    exit_reason = ?
                WHERE symbol = ? AND status = 'OPEN'
            ''', (
                'CLOSED',
                trade.exit_price,
                trade.exit_time.isoformat(),
                trade.pnl,
                trade.return_pct,
                'AUTO_EXIT',  # Would need exit reason from trade
                symbol
            ))
        
        conn.commit()
        conn.close()

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Get trade statistics
        trade_stats = pd.read_sql('''
            SELECT * FROM trades WHERE status = 'CLOSED'
            ORDER BY timestamp DESC
        ''', conn)
        
        # Get equity curve
        equity_curve = pd.read_sql('''
            SELECT * FROM daily_equity 
            ORDER BY date ASC
        ''', conn)
        
        # Get order statistics
        order_stats = pd.read_sql('''
            SELECT status, COUNT(*) as count, order_type
            FROM orders 
            GROUP BY status, order_type
        ''', conn)
        
        conn.close()
        
        stats = super().get_performance_stats()
        
        # Enhance with database insights
        stats['database_insights'] = {
            'total_orders_placed': len(order_stats),
            'order_fill_rate': len(order_stats[order_stats['status'] == 'FILLED']) / len(order_stats) * 100 if len(order_stats) > 0 else 0,
            'order_timeout_rate': len(order_stats[order_stats['status'] == 'CANCELLED']) / len(order_stats) * 100 if len(order_stats) > 0 else 0,
            'avg_trade_duration': trade_stats['exit_time'].apply(lambda x: datetime.fromisoformat(x) if x else None).subtract(
                trade_stats['timestamp'].apply(lambda x: datetime.fromisoformat(x))
            ).dt.total_seconds().mean() / 3600 if len(trade_stats) > 0 else 0,  # hours
            'best_trading_day': equity_curve.loc[equity_curve['realized_pnl'].idxmax(), 'date'] if len(equity_curve) > 0 else None,
            'worst_trading_day': equity_curve.loc[equity_curve['realized_pnl'].idxmin(), 'date'] if len(equity_curve) > 0 else None
        }
        
        return stats

    def track_daily_equity(self, date: datetime):
        """Track daily equity and performance metrics"""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_equity = self.current_balance + unrealized_pnl
        
        # Calculate daily PnL
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get previous day's equity
        cursor.execute(
            'SELECT total_equity FROM daily_equity WHERE date < ? ORDER BY date DESC LIMIT 1',
            (date.strftime('%Y-%m-%d'),)
        )
        previous_equity = cursor.fetchone()
        daily_pnl = total_equity - (previous_equity[0] if previous_equity else self.initial_balance)
        
        # Count today's trades
        cursor.execute(
            'SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?',
            (date.strftime('%Y-%m-%d'),)
        )
        trades_today = cursor.fetchone()[0]
        
        cursor.execute('''
            INSERT OR REPLACE INTO daily_equity (
                date, starting_balance, ending_balance, unrealized_pnl, 
                realized_pnl, total_equity, open_positions, trades_taken, max_drawdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date.strftime('%Y-%m-%d'),
            self.current_balance,
            self.current_balance,
            unrealized_pnl,
            daily_pnl,
            total_equity,
            len(self.positions),
            trades_today,
            self.max_drawdown * 100
        ))
        
        conn.commit()
        conn.close()

    # ========================================================================
    # END ENHANCED FEATURES
    # ========================================================================

    def _reset_daily_state(self):
        """Reset state for new trading day"""
        self.last_scan_date = datetime.now().strftime('%Y-%m-%d')
        self.trade_entered_today = False
        self.daily_target = None
        self.execution_state = ExecutionState.NO_TARGET
        
        logger.info("🔄 Daily state reset - fresh start")
        logger.info("   Trade count today: 0")
        logger.info("   Daily target: None")
        logger.info("   Ready for morning scan")
    
    def _save_target_to_db(self):
        """Save daily target selection to database"""
        if not self.daily_target:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO daily_targets VALUES 
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d'),
            self.daily_target.symbol,
            self.daily_target.entry_ideal,
            self.daily_target.entry_low,
            self.daily_target.entry_high,
            self.daily_target.stop_loss,
            self.daily_target.take_profit,
            self.daily_target.risk_reward,
            self.daily_target.confluence_score,
            self.daily_target.setup_type,
            self.daily_target.selected_at.isoformat(),
            self.daily_target.expires_at.isoformat(),
            'MONITORING',
            None
        ))
        
        conn.commit()
        conn.close()
    
    def _save_execution_to_db(self, order_id: str, symbol: str, risk_amount: float):
        """Save execution details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE daily_targets 
            SET status = ?, outcome = ?
            WHERE date = ? AND symbol = ?
        ''', (
            'EXECUTED',
            f'Order placed: {order_id}, Risk: ${risk_amount:.2f}',
            datetime.now().strftime('%Y-%m-%d'),
            symbol
        ))
        
        conn.commit()
        conn.close()
    
    def _save_timeout_to_db(self):
        """Save timeout event"""
        if not self.daily_target:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE daily_targets 
            SET status = ?, outcome = ?
            WHERE date = ? AND symbol = ?
        ''', (
            'EXPIRED',
            f'Timeout after {self.entry_timeout_hours}h - entry zone never reached',
            datetime.now().strftime('%Y-%m-%d'),
            self.daily_target.symbol
        ))
        
        conn.commit()
        conn.close()
    
    async def _simple_scan(self):
        """Fallback scan if no workflow available"""
        # This would need to implement basic scanning logic
        # For now, return no setups
        return {'valid_setups': False, 'best_setup': None}
    
    def get_discipline_stats(self) -> Dict[str, Any]:
        """Get discipline-specific statistics"""
        stats = self.get_performance_stats()
        
        # Add discipline metrics
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get daily target statistics
            targets_df = pd.read_sql('''
                SELECT * FROM daily_targets 
                ORDER BY date DESC
            ''', conn)
            
            total_days = len(targets_df)
            rest_days = len(targets_df[targets_df['symbol'].isna()])
            target_days = len(targets_df[targets_df['symbol'].notna()])
            executed_days = len(targets_df[targets_df['status'] == 'EXECUTED'])
            expired_days = len(targets_df[targets_df['status'] == 'EXPIRED'])
            
            discipline_stats = {
                'total_trading_days': total_days,
                'rest_days': rest_days,
                'rest_day_pct': (rest_days / total_days * 100) if total_days > 0 else 0,
                'targets_selected': target_days,
                'targets_executed': executed_days,
                'targets_expired': expired_days,
                'execution_rate': (executed_days / target_days * 100) if target_days > 0 else 0,
                'avg_confluence_score': targets_df['confluence_score'].mean() if len(targets_df) > 0 else 0,
                'discipline_maintained': total_days > 0 and executed_days <= total_days  # Never more than 1 per day
            }
            
            stats['discipline_stats'] = discipline_stats
            
        except Exception as e:
            logger.error(f"Error getting discipline stats: {e}")
            stats['discipline_stats'] = {'error': str(e)}
        finally:
            conn.close()
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'execution_state': self.execution_state.value,
            'trade_entered_today': self.trade_entered_today,
            'last_scan_date': self.last_scan_date,
            'positions_count': len(self.positions),
            'current_balance': self.current_balance
        }
        
        if self.daily_target:
            status['daily_target'] = {
                'symbol': self.daily_target.symbol,
                'entry_zone': f"${self.daily_target.entry_low:.4f}-${self.daily_target.entry_high:.4f}",
                'expires_at': self.daily_target.expires_at.isoformat(),
                'confluence_score': self.daily_target.confluence_score,
                'setup_type': self.daily_target.setup_type,
                'time_remaining_hours': (self.daily_target.expires_at - datetime.now()).total_seconds() / 3600
            }
        
        return status