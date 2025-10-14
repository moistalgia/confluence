"""
Filter 4: Risk Check Filter
🎯 Final safety validation before trade execution

Performs comprehensive risk assessment:
- Position sizing (1% account risk)
- Correlation with existing positions 
- Market stress conditions
- Liquidity and execution risk
- Final safety veto power

This is the LAST LINE OF DEFENSE
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.data_provider import DataProvider, TechnicalIndicators

logger = logging.getLogger(__name__)

class RiskCheckFilter:
    """
    Final risk validation filter with veto power over all trades
    Focuses on capital preservation and risk management
    """
    
    def __init__(self, config: Dict):
        """
        Initialize risk check filter
        
        Args:
            config: Risk configuration parameters
        """
        # Account risk parameters
        self.max_account_risk = config.get('max_account_risk', 0.01)     # 1% per trade
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.05)  # 5% total open risk
        self.max_correlation = config.get('max_correlation', 0.7)         # Max position correlation
        
        # Market stress indicators
        self.max_volatility_percentile = config.get('max_volatility_percentile', 95)  # VIX equivalent
        self.min_volume_ratio = config.get('min_volume_ratio', 0.5)       # vs 20-day average
        self.max_spread_bps = config.get('max_spread_bps', 50)           # Max 50 basis points spread
        
        # Liquidity requirements  
        self.min_market_cap_usd = config.get('min_market_cap_usd', 1_000_000_000)  # $1B min cap
        self.max_position_of_volume = config.get('max_position_of_volume', 0.1)    # Max 10% of daily volume
        
        # Timing restrictions
        self.weekend_trading = config.get('weekend_trading', False)       # Allow weekend trades
        self.major_news_buffer_hours = config.get('major_news_buffer_hours', 2)  # Hours before/after news
        
        logger.info("🛡️ Risk Check Filter initialized")
        logger.info(f"   Max account risk: {self.max_account_risk:.1%} per trade")
        logger.info(f"   Max portfolio risk: {self.max_portfolio_risk:.1%} total")
        logger.info(f"   Max correlation: {self.max_correlation:.1%}")
        
    def perform_risk_check(self, symbol: str, trade_direction: str, account_balance: float,
                          open_positions: List[Dict], data_provider: DataProvider) -> Dict:
        """
        Perform comprehensive risk assessment for trade
        
        Args:
            symbol: Trading pair to check
            trade_direction: 'LONG' or 'SHORT'
            account_balance: Current account balance (USD)
            open_positions: List of current open positions
            data_provider: Data source
            
        Returns:
            Dict with risk assessment and final approval
        """
        try:
            logger.info(f"🛡️ RISK CHECK: {symbol} {trade_direction} (balance: ${account_balance:,.0f})")
            
            risk_checks = {}
            risk_score = 0
            max_score = 8  # Total number of risk checks
            
            # Check 1: Position sizing feasibility
            sizing_check = self._check_position_sizing(symbol, account_balance, data_provider)
            risk_checks['position_sizing'] = sizing_check
            if sizing_check['passed']:
                risk_score += 1
            
            # Check 2: Portfolio risk limits
            portfolio_check = self._check_portfolio_risk(account_balance, open_positions)
            risk_checks['portfolio_risk'] = portfolio_check  
            if portfolio_check['passed']:
                risk_score += 1
            
            # Check 3: Position correlation
            correlation_check = self._check_position_correlation(symbol, open_positions, data_provider)
            risk_checks['correlation'] = correlation_check
            if correlation_check['passed']:
                risk_score += 1
            
            # Check 4: Market volatility/stress
            volatility_check = self._check_market_volatility(symbol, data_provider)
            risk_checks['volatility'] = volatility_check
            if volatility_check['passed']:
                risk_score += 1
            
            # Check 5: Liquidity assessment
            liquidity_check = self._check_liquidity_risk(symbol, account_balance, data_provider)
            risk_checks['liquidity'] = liquidity_check
            if liquidity_check['passed']:
                risk_score += 1
            
            # Check 6: Spread/execution risk
            execution_check = self._check_execution_risk(symbol, data_provider)
            risk_checks['execution'] = execution_check
            if execution_check['passed']:
                risk_score += 1
                
            # Check 7: Timing restrictions
            timing_check = self._check_timing_restrictions()
            risk_checks['timing'] = timing_check
            if timing_check['passed']:
                risk_score += 1
                
            # Check 8: Emergency market conditions
            emergency_check = self._check_emergency_conditions(symbol, data_provider)
            risk_checks['emergency'] = emergency_check
            if emergency_check['passed']:
                risk_score += 1
            
            # Calculate final approval (need 7/8 checks to pass)
            min_required_score = 7
            risk_approved = risk_score >= min_required_score
            
            # Get failed check reasons
            failed_checks = [name for name, check in risk_checks.items() if not check['passed']]
            
            result = {
                'symbol': symbol,
                'trade_direction': trade_direction,
                'risk_approved': risk_approved,
                'risk_score': risk_score,
                'max_risk_score': max_score,
                'required_score': min_required_score,
                'failed_checks': failed_checks,
                'risk_checks': risk_checks,
                'filter_name': 'RiskCheckFilter'
            }
            
            # Log result
            if risk_approved:
                logger.info(f"✅ RISK APPROVED {symbol}: {risk_score}/{max_score} checks passed")
            else:
                logger.warning(f"🛡️ RISK REJECTED {symbol}: {risk_score}/{max_score} checks passed")
                logger.warning(f"   Failed: {', '.join(failed_checks)}")
            
            return result
            
        except Exception as e:
            logger.error(f"🚨 Risk check failed for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'trade_direction': trade_direction,
                'risk_approved': False,
                'error': str(e),
                'filter_name': 'RiskCheckFilter'
            }
    
    def _check_position_sizing(self, symbol: str, account_balance: float, 
                              data_provider: DataProvider) -> Dict:
        """
        Check if position can be sized within risk limits
        
        Returns:
            Dict with sizing assessment
        """
        try:
            # Get current price for sizing calculation
            df = data_provider.get_ohlcv(symbol, '1h', days=1)
            current_price = df['close'].iloc[-1]
            
            # Calculate ATR for stop distance
            atr_df = data_provider.get_ohlcv(symbol, '1d', days=30)
            atr = TechnicalIndicators.atr(atr_df, 14).iloc[-1]
            
            # Typical stop distance (1.5x ATR)
            stop_distance = atr * 1.5
            stop_distance_pct = stop_distance / current_price
            
            # Max position size based on 1% account risk
            max_risk_dollars = account_balance * self.max_account_risk
            max_position_size = max_risk_dollars / stop_distance
            min_position_usd = max_position_size * current_price
            
            # Check if minimum tradeable position makes sense
            min_trade_size = 50  # $50 minimum trade
            feasible = min_position_usd >= min_trade_size and stop_distance_pct <= 0.15  # Max 15% stop
            
            return {
                'passed': feasible,
                'current_price': current_price,
                'stop_distance_pct': stop_distance_pct,
                'max_position_usd': min_position_usd,
                'max_risk_dollars': max_risk_dollars,
                'reason': 'Position sizing feasible' if feasible else 
                         f'Stop too wide ({stop_distance_pct:.1%}) or position too small (${min_position_usd:.0f})'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'reason': f'Sizing calculation error: {str(e)}'
            }
    
    def _check_portfolio_risk(self, account_balance: float, open_positions: List[Dict]) -> Dict:
        """
        Check total portfolio risk exposure
        
        Returns:
            Dict with portfolio risk assessment  
        """
        try:
            total_risk_dollars = 0
            
            # Calculate total open risk
            for position in open_positions:
                position_risk = position.get('risk_amount', 0)
                total_risk_dollars += position_risk
            
            # Add proposed trade risk (1% of account)
            proposed_risk = account_balance * self.max_account_risk
            total_risk_with_new = total_risk_dollars + proposed_risk
            
            # Check against max portfolio risk
            max_portfolio_risk_dollars = account_balance * self.max_portfolio_risk
            within_limits = total_risk_with_new <= max_portfolio_risk_dollars
            
            current_portfolio_risk_pct = total_risk_dollars / account_balance
            new_portfolio_risk_pct = total_risk_with_new / account_balance
            
            return {
                'passed': within_limits,
                'current_portfolio_risk_pct': current_portfolio_risk_pct,
                'new_portfolio_risk_pct': new_portfolio_risk_pct,
                'max_portfolio_risk_pct': self.max_portfolio_risk,
                'open_positions_count': len(open_positions),
                'reason': 'Portfolio risk within limits' if within_limits else 
                         f'Portfolio risk too high ({new_portfolio_risk_pct:.1%} vs {self.max_portfolio_risk:.1%} max)'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'reason': f'Portfolio risk calculation error: {str(e)}'
            }
    
    def _check_position_correlation(self, symbol: str, open_positions: List[Dict],
                                   data_provider: DataProvider) -> Dict:
        """
        Check correlation with existing positions
        
        Returns:
            Dict with correlation assessment
        """
        try:
            if len(open_positions) == 0:
                return {
                    'passed': True,
                    'max_correlation': 0.0,
                    'reason': 'No existing positions to correlate with'
                }
            
            correlations = []
            
            # Get symbol's price data for correlation
            symbol_df = data_provider.get_ohlcv(symbol, '1d', days=30)
            symbol_returns = symbol_df['close'].pct_change().dropna()
            
            for position in open_positions:
                try:
                    position_symbol = position.get('symbol', '')
                    if position_symbol and position_symbol != symbol:
                        
                        # Get position's price data
                        pos_df = data_provider.get_ohlcv(position_symbol, '1d', days=30)
                        pos_returns = pos_df['close'].pct_change().dropna()
                        
                        # Calculate correlation (min 20 overlapping days)
                        if len(symbol_returns) >= 20 and len(pos_returns) >= 20:
                            # Align the series
                            aligned_symbol, aligned_pos = symbol_returns.align(pos_returns, join='inner')
                            
                            if len(aligned_symbol) >= 20:
                                correlation = aligned_symbol.corr(aligned_pos)
                                correlations.append(abs(correlation))  # Use absolute correlation
                                
                except Exception as e:
                    logger.warning(f"Could not calculate correlation with {position.get('symbol', 'unknown')}: {str(e)}")
                    continue
            
            # Check maximum correlation
            max_correlation = max(correlations) if correlations else 0.0
            correlation_ok = max_correlation <= self.max_correlation
            
            return {
                'passed': correlation_ok,
                'max_correlation': max_correlation,
                'correlation_limit': self.max_correlation,
                'correlations_checked': len(correlations),
                'reason': 'Correlation acceptable' if correlation_ok else 
                         f'Too correlated with existing position ({max_correlation:.1%} vs {self.max_correlation:.1%} max)'
            }
            
        except Exception as e:
            # If correlation check fails, be conservative and reject
            return {
                'passed': False,
                'reason': f'Correlation check error: {str(e)}'
            }
    
    def _check_market_volatility(self, symbol: str, data_provider: DataProvider) -> Dict:
        """
        Check if market volatility is reasonable for trading
        
        Returns:
            Dict with volatility assessment
        """
        try:
            # Get volatility data
            df = data_provider.get_ohlcv(symbol, '1d', days=60)
            
            # Calculate rolling volatility (20-day)
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std() * np.sqrt(365)  # Annualized
            current_vol = volatility.iloc[-1]
            
            # Calculate volatility percentile (last 60 days)
            vol_percentile = (volatility.rank(pct=True).iloc[-1]) * 100
            
            # Check if volatility is not extreme
            volatility_ok = vol_percentile <= self.max_volatility_percentile
            
            return {
                'passed': volatility_ok,
                'current_volatility': current_vol,
                'volatility_percentile': vol_percentile,
                'max_percentile': self.max_volatility_percentile,
                'reason': 'Volatility acceptable' if volatility_ok else 
                         f'Volatility too high ({vol_percentile:.0f}th percentile vs {self.max_volatility_percentile}th max)'
            }
            
        except Exception as e:
            return {
                'passed': True,  # Default to pass if we can't calculate
                'reason': f'Volatility check skipped: {str(e)}'
            }
    
    def _check_liquidity_risk(self, symbol: str, account_balance: float,
                             data_provider: DataProvider) -> Dict:
        """
        Check liquidity and market impact risk
        
        Returns:
            Dict with liquidity assessment
        """
        try:
            # Get volume data
            df = data_provider.get_ohlcv(symbol, '1d', days=20)
            avg_volume_usd = (df['volume'] * df['close']).mean()
            recent_volume_usd = (df['volume'] * df['close']).iloc[-1]
            
            # Calculate position size
            proposed_position_usd = account_balance * self.max_account_risk / 0.02  # Assume 2% stop
            
            # Check volume ratios
            volume_ratio = recent_volume_usd / avg_volume_usd if avg_volume_usd > 0 else 0
            position_vs_volume = proposed_position_usd / recent_volume_usd if recent_volume_usd > 0 else 1
            
            # Liquidity requirements
            sufficient_volume = volume_ratio >= self.min_volume_ratio
            position_size_ok = position_vs_volume <= self.max_position_of_volume
            min_volume_met = avg_volume_usd >= 1_000_000  # $1M daily volume minimum
            
            liquidity_ok = sufficient_volume and position_size_ok and min_volume_met
            
            reasons = []
            if not sufficient_volume:
                reasons.append(f'Low volume ({volume_ratio:.1%} of average)')
            if not position_size_ok:
                reasons.append(f'Position too large ({position_vs_volume:.1%} of daily volume)')
            if not min_volume_met:
                reasons.append(f'Volume too low (${avg_volume_usd:,.0f} vs $1M min)')
            
            return {
                'passed': liquidity_ok,
                'avg_volume_usd': avg_volume_usd,
                'volume_ratio': volume_ratio,
                'position_vs_volume': position_vs_volume,
                'reason': 'Liquidity sufficient' if liquidity_ok else '; '.join(reasons)
            }
            
        except Exception as e:
            return {
                'passed': False,
                'reason': f'Liquidity check error: {str(e)}'
            }
    
    def _check_execution_risk(self, symbol: str, data_provider: DataProvider) -> Dict:
        """
        Check spread and execution risk
        
        Returns:
            Dict with execution risk assessment
        """
        try:
            # Get recent bars to estimate spread
            df = data_provider.get_ohlcv(symbol, '1h', days=1)
            
            # Estimate spread from high-low range
            recent_bar = df.iloc[-1]
            high_low_range = recent_bar['high'] - recent_bar['low']
            mid_price = (recent_bar['high'] + recent_bar['low']) / 2
            
            # Estimate spread as percentage of mid price
            estimated_spread_pct = (high_low_range / mid_price) if mid_price > 0 else 1.0
            spread_bps = estimated_spread_pct * 10000  # Convert to basis points
            
            # Check if spread is reasonable
            spread_ok = spread_bps <= self.max_spread_bps
            
            return {
                'passed': spread_ok,
                'estimated_spread_bps': spread_bps,
                'max_spread_bps': self.max_spread_bps,
                'reason': 'Execution risk acceptable' if spread_ok else 
                         f'Spread too wide ({spread_bps:.0f} bps vs {self.max_spread_bps} bps max)'
            }
            
        except Exception as e:
            return {
                'passed': True,  # Default to pass if we can't calculate
                'reason': f'Execution risk check skipped: {str(e)}'
            }
    
    def _check_timing_restrictions(self) -> Dict:
        """
        Check if current time is suitable for trading
        
        Returns:
            Dict with timing assessment
        """
        try:
            now = datetime.utcnow()
            
            # Check if weekend (and weekend trading disabled)
            is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
            weekend_ok = self.weekend_trading or not is_weekend
            
            # Crypto markets are 24/7, so main concern is weekends
            timing_ok = weekend_ok
            
            reasons = []
            if not weekend_ok:
                reasons.append('Weekend trading disabled')
            
            return {
                'passed': timing_ok,
                'current_time': now.isoformat(),
                'is_weekend': is_weekend,
                'weekend_trading_allowed': self.weekend_trading,
                'reason': 'Timing acceptable' if timing_ok else '; '.join(reasons)
            }
            
        except Exception as e:
            return {
                'passed': True,  # Default to pass
                'reason': f'Timing check error: {str(e)}'
            }
    
    def _check_emergency_conditions(self, symbol: str, data_provider: DataProvider) -> Dict:
        """
        Check for emergency market conditions that should halt trading
        
        Returns:
            Dict with emergency conditions assessment
        """
        try:
            # Get recent price action
            df = data_provider.get_ohlcv(symbol, '1h', days=1)
            
            # Check for extreme price moves (flash crash/pump)
            if len(df) >= 2:
                last_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                extreme_move = abs(last_change) > 0.20  # 20% move in 1 hour
                
                if extreme_move:
                    return {
                        'passed': False,
                        'last_hour_change': last_change,
                        'reason': f'Extreme price move detected ({last_change:.1%} in 1 hour)'
                    }
            
            # Check for zero volume (market halt)
            recent_volume = df['volume'].iloc[-1] if len(df) > 0 else 0
            if recent_volume <= 0:
                return {
                    'passed': False,
                    'recent_volume': recent_volume,
                    'reason': 'Zero volume detected - possible market halt'
                }
            
            return {
                'passed': True,
                'reason': 'No emergency conditions detected'
            }
            
        except Exception as e:
            return {
                'passed': True,  # Default to pass if we can't check
                'reason': f'Emergency check skipped: {str(e)}'
            }


def perform_final_risk_validation(approved_symbols: List[str], confluence_results: Dict,
                                account_balance: float, open_positions: List[Dict],
                                data_provider: DataProvider, config: Dict) -> Dict:
    """
    Perform final risk validation on confluence-approved symbols
    
    Args:
        approved_symbols: Symbols approved by confluence checker
        confluence_results: Results from confluence checker
        account_balance: Current account balance
        open_positions: Current open positions
        data_provider: Data source
        config: Risk configuration
        
    Returns:
        Dict with final risk validation results
    """
    risk_filter = RiskCheckFilter(config)
    
    final_approved = []
    risk_rejected = []
    risk_results = {}
    
    logger.info(f"🛡️ FINAL RISK CHECK: Validating {len(approved_symbols)} confluence-approved symbols")
    logger.info("=" * 60)
    
    for symbol in approved_symbols:
        try:
            # Get trade direction from confluence results
            confluence_result = confluence_results.get(symbol, {})
            trade_direction = confluence_result.get('trade_direction', 'UNKNOWN')
            
            if trade_direction == 'UNKNOWN':
                logger.warning(f"⚠️ {symbol}: Cannot determine trade direction for risk check")
                risk_rejected.append(symbol)
                continue
            
            # Perform comprehensive risk check
            result = risk_filter.perform_risk_check(
                symbol, trade_direction, account_balance, open_positions, data_provider
            )
            risk_results[symbol] = result
            
            if result['risk_approved']:
                final_approved.append(symbol)
                score = result['risk_score']
                max_score = result['max_risk_score']
                logger.info(f"✅ {symbol}: Final approval granted ({score}/{max_score} risk checks passed)")
            else:
                risk_rejected.append(symbol)
                failed = result.get('failed_checks', [])
                logger.warning(f"🛡️ {symbol}: Risk rejected - {', '.join(failed)}")
                
        except Exception as e:
            logger.error(f"🚨 {symbol}: Risk validation failed - {str(e)}")
            risk_rejected.append(symbol)
            risk_results[symbol] = {
                'symbol': symbol,
                'risk_approved': False,
                'error': str(e),
                'filter_name': 'RiskCheckFilter'
            }
    
    logger.info("=" * 60)
    logger.info(f"📊 FINAL RISK RESULTS:")
    logger.info(f"   FINAL APPROVED: {len(final_approved)}/{len(approved_symbols)} symbols")
    
    if len(final_approved) == 1:
        # Perfect - exactly one trade candidate
        symbol = final_approved[0]
        logger.info(f"🎯 THE ONE TRADE: {symbol}")
        logger.info("   Ready for entry execution!")
    elif len(final_approved) > 1:
        # Multiple candidates - need to pick the best one
        logger.info(f"   Multiple candidates found - will select best setup")
        # Sort by confluence score to pick the best
        sorted_candidates = sorted(final_approved,
                                 key=lambda s: confluence_results.get(s, {}).get('weighted_score', 0),
                                 reverse=True)
        best_candidate = sorted_candidates[0]
        logger.info(f"🎯 THE ONE TRADE (best): {best_candidate}")
    elif len(final_approved) == 0:
        logger.warning("⚠️ NO TRADES passed final risk check - sitting on hands today")
    
    return {
        'final_approved_symbols': final_approved,
        'risk_rejected_symbols': risk_rejected,
        'results': risk_results,
        'the_one_trade': final_approved[0] if len(final_approved) >= 1 else None,
        'summary': {
            'total_checked': len(approved_symbols),
            'final_approved_count': len(final_approved),
            'risk_rejection_rate': len(risk_rejected) / len(approved_symbols) if approved_symbols else 0
        }
    }