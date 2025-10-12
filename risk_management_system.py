#!/usr/bin/env python3
"""
Risk Management System - Advanced Risk Assessment
Provides comprehensive risk analysis for trading decisions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)

class RiskManagementSystem:
    """Advanced risk management and position sizing system"""
    
    def __init__(self):
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk per trade
        self.max_position_size = 0.10   # 10% max position size
        self.min_risk_reward = 2.0      # Minimum 2:1 risk/reward
        self.volatility_lookback = 20   # Days for volatility calculation
        
    def analyze_risk_profile(self, df: pd.DataFrame, symbol: str, 
                           entry_price: float, stop_loss: float, 
                           take_profit: float, portfolio_value: float = 10000) -> Dict:
        """Comprehensive risk analysis for a trade setup"""
        
        try:
            risk_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'trade_setup': {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                },
                'portfolio_value': portfolio_value
            }
            
            # Basic risk metrics
            risk_per_share = abs(entry_price - stop_loss)
            reward_per_share = abs(take_profit - entry_price)
            risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
            
            risk_analysis['basic_metrics'] = {
                'risk_per_share': risk_per_share,
                'reward_per_share': reward_per_share,
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'risk_percentage': round((risk_per_share / entry_price) * 100, 2)
            }
            
            # Position sizing
            position_sizing = self._calculate_position_sizing(
                portfolio_value, entry_price, stop_loss, risk_reward_ratio
            )
            risk_analysis['position_sizing'] = position_sizing
            
            # Volatility analysis
            volatility_metrics = self._calculate_volatility_metrics(df)
            risk_analysis['volatility'] = volatility_metrics
            
            # Market condition risk
            market_risk = self._assess_market_condition_risk(df)
            risk_analysis['market_conditions'] = market_risk
            
            # Liquidity risk
            liquidity_risk = self._assess_liquidity_risk(df)
            risk_analysis['liquidity'] = liquidity_risk
            
            # Technical risk factors
            technical_risk = self._assess_technical_risk_factors(df, entry_price, stop_loss)
            risk_analysis['technical_factors'] = technical_risk
            
            # Overall risk assessment
            overall_assessment = self._calculate_overall_risk_score(risk_analysis)
            risk_analysis['overall_assessment'] = overall_assessment
            
            # Risk management recommendations
            recommendations = self._generate_risk_recommendations(risk_analysis)
            risk_analysis['recommendations'] = recommendations
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Risk analysis error for {symbol}: {e}")
            return self._get_default_risk_analysis(symbol, entry_price, stop_loss, take_profit)
    
    def _calculate_position_sizing(self, portfolio_value: float, entry_price: float, 
                                 stop_loss: float, risk_reward_ratio: float) -> Dict:
        """Calculate optimal position sizing based on risk parameters"""
        
        risk_per_share = abs(entry_price - stop_loss)
        
        # Position size based on fixed percentage risk
        max_portfolio_risk_amount = portfolio_value * self.max_portfolio_risk
        shares_by_risk = max_portfolio_risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # Position size based on maximum position percentage
        max_position_value = portfolio_value * self.max_position_size
        shares_by_position_limit = max_position_value / entry_price if entry_price > 0 else 0
        
        # Take the smaller of the two
        recommended_shares = min(shares_by_risk, shares_by_position_limit)
        position_value = recommended_shares * entry_price
        
        # Calculate actual risk amounts
        actual_risk_amount = recommended_shares * risk_per_share
        actual_risk_percentage = (actual_risk_amount / portfolio_value) * 100
        
        return {
            'recommended_shares': round(recommended_shares, 6),
            'position_value': round(position_value, 2),
            'position_percentage': round((position_value / portfolio_value) * 100, 2),
            'risk_amount': round(actual_risk_amount, 2),
            'risk_percentage': round(actual_risk_percentage, 4),
            'max_loss': round(actual_risk_amount, 2),
            'potential_profit': round(recommended_shares * abs(entry_price - stop_loss) * risk_reward_ratio, 2),
            'meets_risk_criteria': actual_risk_percentage <= (self.max_portfolio_risk * 100),
            'meets_rr_criteria': risk_reward_ratio >= self.min_risk_reward
        }
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive volatility metrics"""
        
        if len(df) < self.volatility_lookback:
            return {'status': 'insufficient_data', 'periods_available': len(df)}
        
        # Price-based volatility
        returns = df['close'].pct_change().dropna()
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(365)
        
        # True Range based volatility (ATR)
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr_20 = true_range.rolling(window=20).mean().iloc[-1]
        atr_percentage = (atr_20 / df['close'].iloc[-1]) * 100
        
        # Volatility regime
        recent_vol = returns.tail(5).std() * np.sqrt(365)
        vol_regime = 'High' if recent_vol > annualized_volatility * 1.2 else \
                     'Low' if recent_vol < annualized_volatility * 0.8 else 'Normal'
        
        return {
            'daily_volatility': round(daily_volatility * 100, 3),
            'annualized_volatility': round(annualized_volatility * 100, 2),
            'atr_20_value': round(atr_20, 6),
            'atr_percentage': round(atr_percentage, 2),
            'volatility_regime': vol_regime,
            'recent_vs_historical': round((recent_vol / annualized_volatility), 2),
            'risk_level': 'High' if atr_percentage > 5 else 'Medium' if atr_percentage > 2 else 'Low'
        }
    
    def _assess_market_condition_risk(self, df: pd.DataFrame) -> Dict:
        """Assess risk based on current market conditions"""
        
        if len(df) < 20:
            return {'status': 'insufficient_data'}
        
        # Trend strength
        close_prices = df['close']
        sma_20 = close_prices.rolling(window=20).mean()
        current_price = close_prices.iloc[-1]
        sma_current = sma_20.iloc[-1]
        
        trend_strength = ((current_price - sma_current) / sma_current) * 100
        
        # Market momentum
        momentum_5d = ((current_price - close_prices.iloc[-6]) / close_prices.iloc[-6]) * 100 if len(df) >= 6 else 0
        momentum_20d = ((current_price - close_prices.iloc[-21]) / close_prices.iloc[-21]) * 100 if len(df) >= 21 else 0
        
        # Volume trend
        volume_avg = df['volume'].rolling(window=10).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_avg.iloc[-1] if volume_avg.iloc[-1] > 0 else 1
        
        # Risk assessment
        if abs(trend_strength) > 10:
            trend_risk = 'High' if trend_strength < -10 else 'Medium'
        else:
            trend_risk = 'Low'
        
        momentum_risk = 'High' if momentum_5d < -5 else 'Medium' if momentum_5d < 0 else 'Low'
        
        return {
            'trend_strength': round(trend_strength, 2),
            'momentum_5d': round(momentum_5d, 2),
            'momentum_20d': round(momentum_20d, 2),
            'volume_ratio': round(volume_ratio, 2),
            'trend_risk': trend_risk,
            'momentum_risk': momentum_risk,
            'volume_confirmation': volume_ratio > 1.2,
            'overall_market_risk': 'High' if trend_risk == 'High' or momentum_risk == 'High' else 'Medium'
        }
    
    def _assess_liquidity_risk(self, df: pd.DataFrame) -> Dict:
        """Assess liquidity risk based on volume and spread patterns"""
        
        if len(df) < 10:
            return {'status': 'insufficient_data'}
        
        # Volume analysis
        avg_volume_10d = df['volume'].tail(10).mean()
        min_volume_10d = df['volume'].tail(10).min()
        volume_consistency = min_volume_10d / avg_volume_10d if avg_volume_10d > 0 else 0
        
        # Spread estimation (using high-low as proxy)
        spreads = ((df['high'] - df['low']) / df['close']) * 100
        avg_spread = spreads.tail(10).mean()
        
        # Liquidity score
        if avg_volume_10d > 1000000 and volume_consistency > 0.5:  # High volume, consistent
            liquidity_score = 'High'
            liquidity_risk = 'Low'
        elif avg_volume_10d > 100000 and volume_consistency > 0.3:  # Medium volume
            liquidity_score = 'Medium'
            liquidity_risk = 'Medium'
        else:  # Low volume or inconsistent
            liquidity_score = 'Low'
            liquidity_risk = 'High'
        
        return {
            'avg_volume_10d': round(avg_volume_10d, 0),
            'volume_consistency': round(volume_consistency, 3),
            'avg_spread_percentage': round(avg_spread, 3),
            'liquidity_score': liquidity_score,
            'liquidity_risk': liquidity_risk,
            'slippage_risk': 'High' if avg_spread > 1.0 else 'Medium' if avg_spread > 0.5 else 'Low'
        }
    
    def _assess_technical_risk_factors(self, df: pd.DataFrame, entry_price: float, stop_loss: float) -> Dict:
        """Assess technical factors that affect risk"""
        
        if len(df) < 20:
            return {'status': 'insufficient_data'}
        
        # Support/Resistance proximity
        highs = df['high'].tail(20)
        lows = df['low'].tail(20)
        
        # Find nearby support/resistance levels
        resistance_levels = highs.nlargest(3).tolist()
        support_levels = lows.nsmallest(3).tolist()
        
        # Check proximity to key levels
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - entry_price))
        nearest_support = min(support_levels, key=lambda x: abs(x - entry_price))
        
        resistance_distance = ((nearest_resistance - entry_price) / entry_price) * 100
        support_distance = ((entry_price - nearest_support) / entry_price) * 100
        
        # Stop loss quality
        stop_distance = abs((entry_price - stop_loss) / entry_price) * 100
        
        # RSI for overbought/oversold conditions (simplified)
        price_changes = df['close'].diff()
        gains = price_changes.where(price_changes > 0, 0)
        losses = -price_changes.where(price_changes < 0, 0)
        
        avg_gains = gains.rolling(window=14).mean()
        avg_losses = losses.rolling(window=14).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
        
        return {
            'nearest_resistance': round(nearest_resistance, 6),
            'nearest_support': round(nearest_support, 6),
            'resistance_distance_pct': round(resistance_distance, 2),
            'support_distance_pct': round(support_distance, 2),
            'stop_distance_pct': round(stop_distance, 2),
            'current_rsi': round(current_rsi, 1),
            'rsi_risk': 'High' if current_rsi > 80 or current_rsi < 20 else 'Medium' if current_rsi > 70 or current_rsi < 30 else 'Low',
            'level_proximity_risk': 'High' if min(abs(resistance_distance), abs(support_distance)) < 1 else 'Low',
            'stop_quality': 'Good' if 1 <= stop_distance <= 5 else 'Poor'
        }
    
    def _calculate_overall_risk_score(self, risk_analysis: Dict) -> Dict:
        """Calculate overall risk score from all factors"""
        
        risk_factors = []
        
        # Basic metrics risk
        basic = risk_analysis.get('basic_metrics', {})
        rr_ratio = basic.get('risk_reward_ratio', 0)
        if rr_ratio >= 3:
            risk_factors.append(('risk_reward', 20))  # Low risk
        elif rr_ratio >= 2:
            risk_factors.append(('risk_reward', 50))  # Medium risk
        else:
            risk_factors.append(('risk_reward', 80))  # High risk
        
        # Volatility risk
        volatility = risk_analysis.get('volatility', {})
        vol_risk = volatility.get('risk_level', 'Medium')
        vol_score = 20 if vol_risk == 'Low' else 50 if vol_risk == 'Medium' else 80
        risk_factors.append(('volatility', vol_score))
        
        # Market conditions risk
        market = risk_analysis.get('market_conditions', {})
        market_risk = market.get('overall_market_risk', 'Medium')
        market_score = 30 if market_risk == 'Low' else 60 if market_risk == 'Medium' else 90
        risk_factors.append(('market_conditions', market_score))
        
        # Liquidity risk
        liquidity = risk_analysis.get('liquidity', {})
        liq_risk = liquidity.get('liquidity_risk', 'Medium')
        liq_score = 15 if liq_risk == 'Low' else 40 if liq_risk == 'Medium' else 75
        risk_factors.append(('liquidity', liq_score))
        
        # Technical risk
        technical = risk_analysis.get('technical_factors', {})
        rsi_risk = technical.get('rsi_risk', 'Medium')
        tech_score = 25 if rsi_risk == 'Low' else 50 if rsi_risk == 'Medium' else 75
        risk_factors.append(('technical', tech_score))
        
        # Calculate weighted average
        total_score = sum(score for _, score in risk_factors)
        avg_score = total_score / len(risk_factors) if risk_factors else 50
        
        # Determine overall risk level
        if avg_score <= 30:
            overall_risk = 'Low'
            recommendation = 'Good risk/reward setup'
        elif avg_score <= 50:
            overall_risk = 'Medium'
            recommendation = 'Acceptable risk with proper sizing'
        elif avg_score <= 70:
            overall_risk = 'High'
            recommendation = 'Consider reducing position size'
        else:
            overall_risk = 'Very High'
            recommendation = 'Avoid this trade setup'
        
        return {
            'overall_risk_score': round(avg_score, 1),
            'risk_level': overall_risk,
            'recommendation': recommendation,
            'risk_factor_breakdown': {factor: score for factor, score in risk_factors},
            'trade_quality': 'Excellent' if avg_score <= 25 else 
                           'Good' if avg_score <= 40 else
                           'Fair' if avg_score <= 60 else 'Poor'
        }
    
    def _generate_risk_recommendations(self, risk_analysis: Dict) -> List[str]:
        """Generate specific risk management recommendations"""
        
        recommendations = []
        
        # Position sizing recommendations
        position = risk_analysis.get('position_sizing', {})
        if not position.get('meets_risk_criteria', True):
            recommendations.append("⚠️ Reduce position size to meet 2% portfolio risk limit")
        
        if not position.get('meets_rr_criteria', True):
            recommendations.append("⚠️ Improve risk/reward ratio to at least 2:1 before entering")
        
        # Volatility recommendations
        volatility = risk_analysis.get('volatility', {})
        if volatility.get('risk_level') == 'High':
            recommendations.append("📊 High volatility detected - consider wider stops or smaller position")
        
        # Market condition recommendations  
        market = risk_analysis.get('market_conditions', {})
        if market.get('momentum_risk') == 'High':
            recommendations.append("📈 Negative momentum - wait for trend confirmation")
        
        if not market.get('volume_confirmation', False):
            recommendations.append("📊 Low volume - ensure sufficient liquidity before entry")
        
        # Technical recommendations
        technical = risk_analysis.get('technical_factors', {})
        if technical.get('rsi_risk') == 'High':
            recommendations.append("🎯 Extreme RSI levels - watch for reversal signals")
        
        if technical.get('stop_quality') == 'Poor':
            recommendations.append("🛑 Optimize stop loss placement - current distance suboptimal")
        
        # Liquidity recommendations
        liquidity = risk_analysis.get('liquidity', {})
        if liquidity.get('liquidity_risk') == 'High':
            recommendations.append("💧 Low liquidity - expect higher slippage and wider spreads")
        
        # Overall recommendations
        overall = risk_analysis.get('overall_assessment', {})
        if overall.get('risk_level') == 'Very High':
            recommendations.append("🚨 AVOID: Risk level too high for recommended criteria")
        elif overall.get('risk_level') == 'High':
            recommendations.append("⚠️ HIGH RISK: Only proceed with reduced position size")
        
        return recommendations
    
    def _get_default_risk_analysis(self, symbol: str, entry_price: float, 
                                 stop_loss: float, take_profit: float) -> Dict:
        """Return default risk analysis when calculation fails"""
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'trade_setup': {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            },
            'overall_assessment': {
                'overall_risk_score': 75,
                'risk_level': 'High',
                'recommendation': 'Unable to assess - proceed with caution',
                'trade_quality': 'Unknown'
            },
            'recommendations': ['🚨 Risk analysis failed - manual review required']
        }
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> Dict:
        """Calculate Kelly Criterion for optimal position sizing"""
        
        try:
            if avg_loss == 0:
                return {'kelly_percentage': 0, 'recommendation': 'Avoid (no loss data)'}
            
            win_loss_ratio = avg_win / abs(avg_loss)
            kelly_percentage = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Practical Kelly (usually 25-50% of full Kelly)
            practical_kelly = kelly_percentage * 0.25
            
            return {
                'kelly_percentage': round(kelly_percentage * 100, 2),
                'practical_kelly': round(practical_kelly * 100, 2),
                'win_rate': round(win_rate * 100, 1),
                'win_loss_ratio': round(win_loss_ratio, 2),
                'recommendation': 'Use practical Kelly for position sizing' if practical_kelly > 0 else 'Negative expectancy - avoid'
            }
            
        except Exception as e:
            logger.error(f"Kelly criterion calculation error: {e}")
            return {'kelly_percentage': 0, 'recommendation': 'Calculation failed'}

def main():
    """Test risk management system"""
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.02)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(50) * 0.001),
        'high': prices * (1 + abs(np.random.randn(50)) * 0.005),
        'low': prices * (1 - abs(np.random.randn(50)) * 0.005),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, 50)
    })
    
    # Initialize risk manager
    risk_mgr = RiskManagementSystem()
    
    # Test risk analysis
    entry_price = prices[-1]
    stop_loss = entry_price * 0.97  # 3% stop loss
    take_profit = entry_price * 1.06  # 6% take profit
    
    print("🛡️ RISK MANAGEMENT SYSTEM TEST")
    print("=" * 50)
    
    risk_analysis = risk_mgr.analyze_risk_profile(
        df, 'TEST/USDT', entry_price, stop_loss, take_profit, 10000
    )
    
    # Display results
    print(f"\n📊 Trade Setup:")
    setup = risk_analysis['trade_setup']
    print(f"Entry: ${setup['entry_price']:.2f}")
    print(f"Stop: ${setup['stop_loss']:.2f}")
    print(f"Target: ${setup['take_profit']:.2f}")
    
    basic = risk_analysis.get('basic_metrics', {})
    print(f"\n📈 Risk/Reward: {basic.get('risk_reward_ratio', 'N/A')}:1")
    print(f"Risk %: {basic.get('risk_percentage', 'N/A')}%")
    
    position = risk_analysis.get('position_sizing', {})
    print(f"\n💰 Position Sizing:")
    print(f"Recommended shares: {position.get('recommended_shares', 'N/A')}")
    print(f"Risk amount: ${position.get('risk_amount', 'N/A')}")
    
    overall = risk_analysis.get('overall_assessment', {})
    print(f"\n🎯 Overall Assessment:")
    print(f"Risk Level: {overall.get('risk_level', 'N/A')}")
    print(f"Trade Quality: {overall.get('trade_quality', 'N/A')}")
    print(f"Recommendation: {overall.get('recommendation', 'N/A')}")
    
    # Test Kelly Criterion
    kelly = risk_mgr.calculate_kelly_criterion(0.6, 0.06, 0.03)
    print(f"\n📊 Kelly Criterion:")
    print(f"Optimal size: {kelly.get('kelly_percentage', 'N/A')}%")
    print(f"Practical size: {kelly.get('practical_kelly', 'N/A')}%")

if __name__ == "__main__":
    main()