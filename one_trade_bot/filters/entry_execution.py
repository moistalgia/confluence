"""
Filter 5: Entry Execution
🎯 Disciplined limit order placement and fill management

Handles the final trade execution:
- Precise entry price calculation
- Disciplined limit order placement
- Fill monitoring and adjustment
- Stop-loss and target placement
- Position tracking initiation

NO MARKET ORDERS - Only disciplined limit entries
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.data_provider import DataProvider, TechnicalIndicators

logger = logging.getLogger(__name__)

class EntryExecution:
    """
    Handles disciplined entry execution for THE ONE TRADE
    Focus on quality fills at good prices, not speed
    """
    
    def __init__(self, config: Dict):
        """
        Initialize entry execution handler
        
        Args:
            config: Execution configuration
        """
        # Entry timing parameters
        self.max_entry_wait_minutes = config.get('max_entry_wait_minutes', 60)    # Max 1 hour wait
        self.entry_patience_factor = config.get('entry_patience_factor', 0.5)     # How patient for entry
        self.limit_order_buffer_bps = config.get('limit_order_buffer_bps', 10)    # 10 basis points buffer
        
        # Stop-loss parameters
        self.stop_loss_method = config.get('stop_loss_method', 'atr')            # 'atr' or 'support'
        self.atr_stop_multiplier = config.get('atr_stop_multiplier', 1.5)        # 1.5x ATR for stops
        self.min_stop_distance_pct = config.get('min_stop_distance_pct', 0.02)   # Min 2% stop distance
        self.max_stop_distance_pct = config.get('max_stop_distance_pct', 0.08)   # Max 8% stop distance
        
        # Target parameters  
        self.profit_target_ratio = config.get('profit_target_ratio', 2.0)        # 2:1 risk:reward
        self.trailing_stop_activation = config.get('trailing_stop_activation', 1.0) # Start trailing at 1R
        
        # Position sizing
        self.account_risk_pct = config.get('account_risk_pct', 0.01)             # 1% account risk
        
        logger.info("⚡ Entry Execution initialized")
        logger.info(f"   Max entry wait: {self.max_entry_wait_minutes} minutes")
        logger.info(f"   Risk:Reward target: 1:{self.profit_target_ratio}")
        logger.info(f"   Account risk: {self.account_risk_pct:.1%} per trade")
        
    def calculate_entry_plan(self, symbol: str, trade_direction: str, account_balance: float,
                           data_provider: DataProvider) -> Dict:
        """
        Calculate precise entry plan for the approved trade
        
        Args:
            symbol: Trading pair
            trade_direction: 'LONG' or 'SHORT'
            account_balance: Current account balance
            data_provider: Data source
            
        Returns:
            Dict with complete entry execution plan
        """
        try:
            logger.info(f"⚡ CALCULATING ENTRY PLAN: {symbol} {trade_direction}")
            
            # Get current market data
            current_data = self._get_current_market_data(symbol, data_provider)
            current_price = current_data['current_price']
            atr = current_data['atr']
            
            # Calculate optimal entry price (limit order)
            entry_price = self._calculate_entry_price(
                current_price, trade_direction, current_data
            )
            
            # Calculate stop-loss price
            stop_loss_price = self._calculate_stop_loss(
                entry_price, trade_direction, atr, current_data
            )
            
            # Calculate position size based on risk
            position_size, risk_amount = self._calculate_position_size(
                account_balance, entry_price, stop_loss_price
            )
            
            # Calculate profit target
            profit_target_price = self._calculate_profit_target(
                entry_price, stop_loss_price, trade_direction
            )
            
            # Validate the plan
            plan_validation = self._validate_entry_plan(
                entry_price, stop_loss_price, profit_target_price, 
                position_size, current_price, trade_direction
            )
            
            if not plan_validation['valid']:
                return {
                    'symbol': symbol,
                    'trade_direction': trade_direction,
                    'plan_ready': False,
                    'error': plan_validation['reason'],
                    'filter_name': 'EntryExecution'
                }
            
            # Calculate key metrics
            risk_reward_ratio = abs(profit_target_price - entry_price) / abs(entry_price - stop_loss_price)
            stop_distance_pct = abs(entry_price - stop_loss_price) / entry_price
            entry_buffer_pct = abs(current_price - entry_price) / current_price
            
            entry_plan = {
                'symbol': symbol,
                'trade_direction': trade_direction,
                'plan_ready': True,
                
                # Prices
                'current_price': current_price,
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'profit_target_price': profit_target_price,
                
                # Position details
                'position_size': position_size,
                'position_value_usd': position_size * entry_price,
                'risk_amount': risk_amount,
                'potential_profit': abs(profit_target_price - entry_price) * position_size,
                
                # Metrics
                'risk_reward_ratio': risk_reward_ratio,
                'stop_distance_pct': stop_distance_pct,
                'entry_buffer_pct': entry_buffer_pct,
                'account_risk_pct': risk_amount / account_balance,
                
                # Timing
                'plan_created_at': datetime.utcnow().isoformat(),
                'entry_expires_at': (datetime.utcnow() + timedelta(minutes=self.max_entry_wait_minutes)).isoformat(),
                
                # Technical context
                'atr_value': atr,
                'market_data': current_data,
                
                'filter_name': 'EntryExecution'
            }
            
            # Log the plan
            logger.info(f"📋 ENTRY PLAN CALCULATED:")
            logger.info(f"   Entry: ${entry_price:,.4f} (limit order)")
            logger.info(f"   Stop: ${stop_loss_price:,.4f} ({stop_distance_pct:.1%} risk)")
            logger.info(f"   Target: ${profit_target_price:,.4f} (1:{risk_reward_ratio:.1f} RR)")
            logger.info(f"   Size: {position_size:,.2f} units (${position_size * entry_price:,.0f})")
            logger.info(f"   Risk: ${risk_amount:,.0f} ({risk_amount/account_balance:.1%})")
            
            return entry_plan
            
        except Exception as e:
            logger.error(f"🚨 Entry plan calculation failed for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'trade_direction': trade_direction,
                'plan_ready': False,
                'error': str(e),
                'filter_name': 'EntryExecution'
            }
    
    def _get_current_market_data(self, symbol: str, data_provider: DataProvider) -> Dict:
        """
        Get current market data needed for entry calculations
        
        Returns:
            Dict with market data
        """
        # Get recent price data
        df_1h = data_provider.get_ohlcv(symbol, '1h', days=3)
        df_1d = data_provider.get_ohlcv(symbol, '1d', days=30)
        
        current_price = df_1h['close'].iloc[-1]
        
        # Calculate ATR for stop placement
        atr = TechnicalIndicators.atr(df_1d, 14).iloc[-1]
        
        # Get support/resistance levels
        recent_high = df_1h['high'].tail(24).max()  # Last 24 hours
        recent_low = df_1h['low'].tail(24).min()
        
        # Volume analysis
        avg_volume = df_1h['volume'].tail(20).mean()
        current_volume = df_1h['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Momentum indicators
        rsi = TechnicalIndicators.rsi(df_1h['close'], 14).iloc[-1]
        sma_20 = TechnicalIndicators.sma(df_1h['close'], 20).iloc[-1]
        
        return {
            'current_price': current_price,
            'atr': atr,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'volume_ratio': volume_ratio,
            'rsi': rsi,
            'sma_20': sma_20,
            'price_vs_sma20': (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
        }
    
    def _calculate_entry_price(self, current_price: float, trade_direction: str, 
                              market_data: Dict) -> float:
        """
        Calculate optimal entry price for limit order
        
        Returns:
            Entry price for limit order
        """
        buffer_amount = current_price * (self.limit_order_buffer_bps / 10000)
        
        if trade_direction == "LONG":
            # For long, place limit order below current price
            entry_price = current_price - buffer_amount
            
            # Don't place below recent support
            recent_support = market_data['recent_low']
            if entry_price < recent_support:
                entry_price = recent_support * 1.001  # Just above support
                
        else:  # SHORT
            # For short, place limit order above current price  
            entry_price = current_price + buffer_amount
            
            # Don't place above recent resistance
            recent_resistance = market_data['recent_high']
            if entry_price > recent_resistance:
                entry_price = recent_resistance * 0.999  # Just below resistance
        
        return entry_price
    
    def _calculate_stop_loss(self, entry_price: float, trade_direction: str, 
                           atr: float, market_data: Dict) -> float:
        """
        Calculate stop-loss price based on configured method
        
        Returns:
            Stop-loss price
        """
        if self.stop_loss_method == 'atr':
            # ATR-based stop
            stop_distance = atr * self.atr_stop_multiplier
            
            if trade_direction == "LONG":
                stop_price = entry_price - stop_distance
            else:  # SHORT
                stop_price = entry_price + stop_distance
                
        else:  # support/resistance method
            if trade_direction == "LONG":
                # Place stop below recent support
                support_level = market_data['recent_low']
                stop_price = support_level * 0.995  # 0.5% below support
            else:  # SHORT
                # Place stop above recent resistance
                resistance_level = market_data['recent_high']
                stop_price = resistance_level * 1.005  # 0.5% above resistance
        
        # Validate stop distance is within acceptable range
        stop_distance_pct = abs(entry_price - stop_price) / entry_price
        
        if stop_distance_pct < self.min_stop_distance_pct:
            # Stop too close - widen it
            if trade_direction == "LONG":
                stop_price = entry_price * (1 - self.min_stop_distance_pct)
            else:
                stop_price = entry_price * (1 + self.min_stop_distance_pct)
                
        elif stop_distance_pct > self.max_stop_distance_pct:
            # Stop too wide - tighten it
            if trade_direction == "LONG":
                stop_price = entry_price * (1 - self.max_stop_distance_pct)
            else:
                stop_price = entry_price * (1 + self.max_stop_distance_pct)
        
        return stop_price
    
    def _calculate_position_size(self, account_balance: float, entry_price: float, 
                               stop_price: float) -> Tuple[float, float]:
        """
        Calculate position size based on 1% account risk
        
        Returns:
            Tuple of (position_size, risk_amount)
        """
        # Calculate maximum risk amount
        max_risk_amount = account_balance * self.account_risk_pct
        
        # Calculate stop distance per unit
        stop_distance_per_unit = abs(entry_price - stop_price)
        
        # Calculate position size
        position_size = max_risk_amount / stop_distance_per_unit
        
        return position_size, max_risk_amount
    
    def _calculate_profit_target(self, entry_price: float, stop_price: float, 
                               trade_direction: str) -> float:
        """
        Calculate profit target based on risk:reward ratio
        
        Returns:
            Profit target price
        """
        risk_per_unit = abs(entry_price - stop_price)
        reward_per_unit = risk_per_unit * self.profit_target_ratio
        
        if trade_direction == "LONG":
            target_price = entry_price + reward_per_unit
        else:  # SHORT
            target_price = entry_price - reward_per_unit
        
        return target_price
    
    def _validate_entry_plan(self, entry_price: float, stop_price: float, 
                           target_price: float, position_size: float, 
                           current_price: float, trade_direction: str) -> Dict:
        """
        Validate that the entry plan makes sense
        
        Returns:
            Dict with validation result
        """
        issues = []
        
        # Check price logic
        if trade_direction == "LONG":
            if entry_price >= stop_price:
                issues.append("Entry price must be above stop for LONG")
            if target_price <= entry_price:
                issues.append("Target price must be above entry for LONG")
        else:  # SHORT
            if entry_price <= stop_price:
                issues.append("Entry price must be below stop for SHORT")  
            if target_price >= entry_price:
                issues.append("Target price must be below entry for SHORT")
        
        # Check position size
        if position_size <= 0:
            issues.append("Position size must be positive")
        
        # Check if entry is too far from current price
        entry_distance_pct = abs(current_price - entry_price) / current_price
        if entry_distance_pct > 0.05:  # More than 5% away
            issues.append(f"Entry too far from current price ({entry_distance_pct:.1%})")
        
        # Check minimum position value
        min_position_value = 50  # $50 minimum
        position_value = position_size * entry_price
        if position_value < min_position_value:
            issues.append(f"Position too small (${position_value:.0f} vs ${min_position_value} min)")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'reason': '; '.join(issues) if issues else 'Plan validation passed'
        }
    
    def monitor_entry_opportunity(self, entry_plan: Dict, data_provider: DataProvider) -> Dict:
        """
        Monitor market for entry opportunity based on plan
        
        Args:
            entry_plan: Entry plan from calculate_entry_plan
            data_provider: Data source
            
        Returns:
            Dict with entry status and recommendations
        """
        try:
            symbol = entry_plan['symbol']
            trade_direction = entry_plan['trade_direction']
            target_entry_price = entry_plan['entry_price']
            
            # Get current market price
            df = data_provider.get_ohlcv(symbol, '1h', days=1)
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Check if entry price has been hit
            entry_hit = False
            if trade_direction == "LONG":
                entry_hit = current_price <= target_entry_price
            else:  # SHORT
                entry_hit = current_price >= target_entry_price
            
            # Check if entry window has expired
            entry_expires_at = datetime.fromisoformat(entry_plan['entry_expires_at'].replace('Z', '+00:00'))
            entry_expired = datetime.utcnow() > entry_expires_at.replace(tzinfo=None)
            
            # Calculate how long we've been waiting
            plan_created_at = datetime.fromisoformat(entry_plan['plan_created_at'].replace('Z', '+00:00'))
            wait_time_minutes = (datetime.utcnow() - plan_created_at.replace(tzinfo=None)).total_seconds() / 60
            
            # Determine entry status
            if entry_hit and not entry_expired:
                entry_status = "READY_TO_ENTER"
                recommendation = f"Execute limit order at ${target_entry_price:,.4f}"
            elif entry_expired:
                entry_status = "EXPIRED"
                recommendation = "Entry opportunity expired - reassess market"
            elif wait_time_minutes > self.max_entry_wait_minutes * 0.75:  # 75% of wait time elapsed
                entry_status = "PATIENCE_RUNNING_OUT"
                recommendation = "Consider market order if price approaches entry level"
            else:
                entry_status = "WAITING"
                distance_pct = abs(current_price - target_entry_price) / target_entry_price
                recommendation = f"Wait for price to reach ${target_entry_price:,.4f} ({distance_pct:.1%} away)"
            
            result = {
                'symbol': symbol,
                'entry_status': entry_status,
                'recommendation': recommendation,
                'current_price': current_price,
                'target_entry_price': target_entry_price,
                'entry_hit': entry_hit,
                'entry_expired': entry_expired,
                'wait_time_minutes': wait_time_minutes,
                'max_wait_minutes': self.max_entry_wait_minutes,
                'current_volume': current_volume,
                'price_distance_pct': abs(current_price - target_entry_price) / target_entry_price,
                'checked_at': datetime.utcnow().isoformat()
            }
            
            # Log status
            if entry_status == "READY_TO_ENTER":
                logger.info(f"🎯 ENTRY READY: {symbol} hit target price ${target_entry_price:,.4f}")
            elif entry_status == "EXPIRED":
                logger.warning(f"⏰ ENTRY EXPIRED: {symbol} after {wait_time_minutes:.0f} minutes")
            else:
                distance_pct = abs(current_price - target_entry_price) / target_entry_price
                logger.debug(f"⏳ WAITING: {symbol} entry ({distance_pct:.1%} away, {wait_time_minutes:.0f}min elapsed)")
            
            return result
            
        except Exception as e:
            logger.error(f"🚨 Entry monitoring failed for {entry_plan.get('symbol', 'unknown')}: {str(e)}")
            return {
                'symbol': entry_plan.get('symbol', 'unknown'),
                'entry_status': 'ERROR',
                'recommendation': f'Monitoring error: {str(e)}',
                'error': str(e)
            }


def execute_entry_workflow(final_trade_symbol: str, trade_direction: str, 
                         account_balance: float, data_provider: DataProvider, 
                         config: Dict) -> Dict:
    """
    Execute complete entry workflow for THE ONE TRADE
    
    Args:
        final_trade_symbol: The selected symbol for entry
        trade_direction: 'LONG' or 'SHORT'
        account_balance: Current account balance
        data_provider: Data source
        config: Entry execution configuration
        
    Returns:
        Dict with complete entry workflow result
    """
    entry_executor = EntryExecution(config)
    
    logger.info(f"⚡ ENTRY WORKFLOW: Executing THE ONE TRADE - {final_trade_symbol} {trade_direction}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Calculate entry plan
        logger.info("Step 1: Calculating entry plan...")
        entry_plan = entry_executor.calculate_entry_plan(
            final_trade_symbol, trade_direction, account_balance, data_provider
        )
        
        if not entry_plan.get('plan_ready', False):
            logger.error(f"❌ Entry plan failed: {entry_plan.get('error', 'Unknown error')}")
            return {
                'workflow_status': 'FAILED',
                'step': 'plan_calculation',
                'error': entry_plan.get('error', 'Plan calculation failed'),
                'entry_plan': entry_plan
            }
        
        # Step 2: Monitor for entry opportunity
        logger.info("Step 2: Monitoring for entry opportunity...")
        entry_monitor = entry_executor.monitor_entry_opportunity(entry_plan, data_provider)
        
        workflow_result = {
            'workflow_status': 'SUCCESS',
            'step': 'monitoring',
            'final_trade_symbol': final_trade_symbol,
            'trade_direction': trade_direction,
            'entry_plan': entry_plan,
            'entry_monitor': entry_monitor,
            'next_action': None
        }
        
        # Determine next action based on entry status
        entry_status = entry_monitor.get('entry_status', 'UNKNOWN')
        
        if entry_status == 'READY_TO_ENTER':
            workflow_result['next_action'] = 'EXECUTE_LIMIT_ORDER'
            logger.info("🎯 READY FOR EXECUTION: Place limit order now!")
            
        elif entry_status == 'WAITING':
            workflow_result['next_action'] = 'CONTINUE_MONITORING'
            wait_time = entry_monitor.get('wait_time_minutes', 0)
            logger.info(f"⏳ CONTINUE WAITING: {wait_time:.0f} minutes elapsed")
            
        elif entry_status == 'PATIENCE_RUNNING_OUT':
            workflow_result['next_action'] = 'CONSIDER_MARKET_ORDER'
            logger.warning("⏰ PATIENCE RUNNING OUT: Consider market order if price approaches")
            
        elif entry_status == 'EXPIRED':
            workflow_result['next_action'] = 'REASSESS_MARKET'
            workflow_result['workflow_status'] = 'EXPIRED'
            logger.warning("⏰ ENTRY EXPIRED: Need to reassess market conditions")
            
        else:
            workflow_result['next_action'] = 'ERROR_HANDLING'
            logger.error(f"❌ Unknown entry status: {entry_status}")
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"📊 ENTRY WORKFLOW SUMMARY:")
        logger.info(f"   Symbol: {final_trade_symbol}")
        logger.info(f"   Direction: {trade_direction}")
        logger.info(f"   Entry Price: ${entry_plan.get('entry_price', 0):,.4f}")
        logger.info(f"   Position Size: {entry_plan.get('position_size', 0):,.2f} units")
        logger.info(f"   Risk Amount: ${entry_plan.get('risk_amount', 0):,.0f}")
        logger.info(f"   Status: {entry_status}")
        logger.info(f"   Next Action: {workflow_result['next_action']}")
        
        return workflow_result
        
    except Exception as e:
        logger.error(f"🚨 Entry workflow failed: {str(e)}")
        return {
            'workflow_status': 'ERROR',
            'step': 'workflow_execution',
            'error': str(e),
            'final_trade_symbol': final_trade_symbol,
            'trade_direction': trade_direction
        }