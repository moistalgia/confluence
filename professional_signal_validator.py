#!/usr/bin/env python3
"""
Professional Signal Validator
============================

Replaces the broken "wait 30+ minutes and hope" system with instant
multi-factor professional validation in <1 second.

Key improvements:
- Multi-indicator confluence validation (30% weight)
- Multi-timeframe alignment check (25% weight)  
- Volume confirmation analysis (20% weight)
- Market structure validation (15% weight)
- Risk/reward quality assessment (10% weight)

Time to validation: <1 second (not 30+ minutes)
Validation threshold: 60%+ for execution

Author: Professional Trading Team
Date: October 13, 2025
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Signal validation results"""
    EXECUTE = "EXECUTE"      # 75%+ validation score - execute immediately
    APPROVED = "APPROVED"    # 60-74% validation score - execute with normal size
    MARGINAL = "MARGINAL"    # 45-59% validation score - execute with reduced size  
    REJECTED = "REJECTED"    # <45% validation score - skip trade

@dataclass
class MarketData:
    """Complete market data for validation"""
    symbol: str
    current_price: float
    timestamp: datetime
    
    # Technical indicators
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    bb_middle: float
    stoch: float
    
    # Moving averages
    sma_20: float
    sma_50: float
    sma_200: float
    ema_9: float
    ema_21: float
    
    # Volume data
    current_volume: float
    avg_volume_20: float
    volume_ratio: float
    
    # Multi-timeframe data
    timeframe_data: Dict[str, Dict] = None
    
    def get_bb_position(self) -> float:
        """Get position within Bollinger Bands (0=lower, 1=upper)"""
        if self.bb_upper == self.bb_lower:
            return 0.5
        return (self.current_price - self.bb_lower) / (self.bb_upper - self.bb_lower)
    
    def get_ma_alignment_bullish(self) -> bool:
        """Check if moving averages are in bullish alignment"""
        return (self.ema_9 > self.ema_21 > self.sma_50 > self.sma_200 and
                self.current_price > self.ema_9)
    
    def get_ma_alignment_bearish(self) -> bool:
        """Check if moving averages are in bearish alignment"""  
        return (self.ema_9 < self.ema_21 < self.sma_50 < self.sma_200 and
                self.current_price < self.ema_9)

@dataclass
class TradingSignal:
    """Enhanced trading signal for validation"""
    symbol: str
    action: str  # BUY, SELL
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    source: str
    reason: str
    timestamp: datetime
    
    # Additional validation fields
    timeframe: str = "1h"
    risk_reward_ratio: float = 0.0
    expected_hold_time_hours: float = 24.0
    
    def __post_init__(self):
        """Calculate risk/reward ratio"""
        if self.action == "BUY":
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:  # SELL
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        
        if risk > 0:
            self.risk_reward_ratio = reward / risk
        else:
            self.risk_reward_ratio = 0.0

class ProfessionalSignalValidator:
    """
    Professional-grade signal validation system
    
    Validates signals instantly using multi-factor analysis:
    - Indicator confluence
    - Timeframe alignment  
    - Volume confirmation
    - Market structure
    - Risk/reward quality
    """
    
    def __init__(self):
        self.validation_history: List[Dict] = []
        self.performance_stats = {
            'total_validated': 0,
            'executed': 0,
            'approved': 0,
            'marginal': 0,
            'rejected': 0
        }
        
        logger.info("🔍 Professional Signal Validator initialized")
    
    def validate_signal_instantly(self, signal: TradingSignal, 
                                market_data: MarketData) -> Dict[str, Any]:
        """
        Validate signal instantly using multi-factor analysis
        
        Returns: {
            'result': ValidationResult,
            'score': float (0-1),
            'confidence_adjustment': int (-10 to +10),
            'reasons': List[str],
            'execution_size_multiplier': float (0.5-1.5)
        }
        """
        
        start_time = datetime.now()
        
        validation_scores = []
        all_reasons = []
        
        # 1. Multi-Indicator Confluence (30% weight)
        indicator_score, indicator_reasons = self._check_indicator_confluence(
            signal, market_data
        )
        validation_scores.append(indicator_score * 0.30)
        all_reasons.extend([f"Indicators: {r}" for r in indicator_reasons])
        
        # 2. Multi-Timeframe Alignment (25% weight)
        timeframe_score, timeframe_reasons = self._check_timeframe_alignment(
            signal, market_data
        )
        validation_scores.append(timeframe_score * 0.25)
        all_reasons.extend([f"Timeframes: {r}" for r in timeframe_reasons])
        
        # 3. Volume Confirmation (20% weight)
        volume_score, volume_reasons = self._check_volume_confirmation(
            signal, market_data
        )
        validation_scores.append(volume_score * 0.20)
        all_reasons.extend([f"Volume: {r}" for r in volume_reasons])
        
        # 4. Market Structure (15% weight)
        structure_score, structure_reasons = self._check_market_structure(
            signal, market_data
        )
        validation_scores.append(structure_score * 0.15)
        all_reasons.extend([f"Structure: {r}" for r in structure_reasons])
        
        # 5. Risk/Reward Quality (10% weight)
        rr_score, rr_reasons = self._check_risk_reward_quality(signal)
        validation_scores.append(rr_score * 0.10)
        all_reasons.extend([f"Risk/Reward: {r}" for r in rr_reasons])
        
        # Calculate total validation score
        total_score = sum(validation_scores)
        
        # Determine validation result
        if total_score >= 0.75:
            result = ValidationResult.EXECUTE
            confidence_adjustment = +5
            size_multiplier = 1.2  # Increase size for high-quality signals
        elif total_score >= 0.60:
            result = ValidationResult.APPROVED  
            confidence_adjustment = 0
            size_multiplier = 1.0
        elif total_score >= 0.45:
            result = ValidationResult.MARGINAL
            confidence_adjustment = -3
            size_multiplier = 0.7  # Reduce size for marginal signals
        else:
            result = ValidationResult.REJECTED
            confidence_adjustment = -10
            size_multiplier = 0.0
        
        # Calculate validation time
        validation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Record validation with individual factor scores
        validation_record = {
            'timestamp': start_time,
            'symbol': signal.symbol,
            'action': signal.action,
            'original_confidence': signal.confidence,
            'validation_score': total_score,
            'result': result.value,
            'validation_time_ms': validation_time_ms,
            'reasons': all_reasons,
            # Individual factor scores (in percentages for dashboard display)
            'indicator_confluence_score': indicator_score * 100,
            'timeframe_alignment_score': timeframe_score * 100,
            'volume_confirmation_score': volume_score * 100,
            'market_structure_score': structure_score * 100,
            'risk_reward_score': rr_score * 100
        }
        
        self.validation_history.append(validation_record)
        self.performance_stats['total_validated'] += 1
        self.performance_stats[result.value.lower()] += 1
        
        logger.info(f"⚡ INSTANT VALIDATION: {signal.action} {signal.symbol} = "
                   f"{result.value} ({total_score:.1%} score, {validation_time_ms:.1f}ms)")
        
        return {
            'result': result,
            'score': total_score,
            'confidence_adjustment': confidence_adjustment,
            'reasons': all_reasons,
            'execution_size_multiplier': size_multiplier,
            'validation_time_ms': validation_time_ms,
            # Individual factor scores for dashboard display
            'indicator_confluence_score': indicator_score * 100,
            'timeframe_alignment_score': timeframe_score * 100,
            'volume_confirmation_score': volume_score * 100,
            'market_structure_score': structure_score * 100,
            'risk_reward_score': rr_score * 100,
            # Detailed validation breakdown for dashboard
            'detailed_breakdown': {
                'indicator_details': indicator_reasons,
                'timeframe_details': timeframe_reasons,
                'volume_details': volume_reasons,
                'structure_details': structure_reasons,
                'risk_reward_details': rr_reasons,
                'raw_scores': {
                    'indicator_raw': indicator_score,
                    'timeframe_raw': timeframe_score,
                    'volume_raw': volume_score,
                    'structure_raw': structure_score,
                    'risk_reward_raw': rr_score
                },
                'weighted_scores': {
                    'indicator_weighted': indicator_score * 0.30,
                    'timeframe_weighted': timeframe_score * 0.25,
                    'volume_weighted': volume_score * 0.20,
                    'structure_weighted': structure_score * 0.15,
                    'risk_reward_weighted': rr_score * 0.10
                }
            }
        }
    
    def _check_indicator_confluence(self, signal: TradingSignal, 
                                  market_data: MarketData) -> Tuple[float, List[str]]:
        """Check if multiple indicators agree with the signal"""
        
        indicators_aligned = 0
        total_indicators = 0
        reasons = []
        
        # Add current indicator values for transparency
        reasons.append(f"RSI: {market_data.rsi:.1f} | MACD: {market_data.macd:.4f}")
        reasons.append(f"Stoch: {market_data.stoch:.1f} | BB Position: {market_data.get_bb_position():.2f}")
        
        if signal.action == "BUY":
            # RSI oversold check
            if market_data.rsi < 35:
                indicators_aligned += 1
                reasons.append(f"✓ RSI oversold at {market_data.rsi:.1f} (<35)")
            else:
                reasons.append(f"✗ RSI neutral at {market_data.rsi:.1f} (≥35)")
            total_indicators += 1
            
            # MACD bullish check
            if market_data.macd > market_data.macd_signal:
                indicators_aligned += 1
                reasons.append(f"✓ MACD bullish ({market_data.macd:.4f} > {market_data.macd_signal:.4f})")
            else:
                reasons.append(f"✗ MACD bearish ({market_data.macd:.4f} ≤ {market_data.macd_signal:.4f})")
            total_indicators += 1
            
            # Bollinger Bands lower check
            bb_pos = market_data.get_bb_position()
            if bb_pos < 0.25:
                indicators_aligned += 1
                reasons.append(f"✓ BB lower bounce at {bb_pos:.2f} (<0.25)")
            else:
                reasons.append(f"✗ BB position at {bb_pos:.2f} (≥0.25)")
            total_indicators += 1
            
            # Stochastic oversold check
            if market_data.stoch < 25:
                indicators_aligned += 1
                reasons.append(f"✓ Stoch oversold at {market_data.stoch:.1f} (<25)")
            else:
                reasons.append(f"✗ Stoch neutral at {market_data.stoch:.1f} (≥25)")
            total_indicators += 1
            
        else:  # SELL
            # RSI overbought check  
            if market_data.rsi > 65:
                indicators_aligned += 1
                reasons.append(f"✓ RSI overbought at {market_data.rsi:.1f} (>65)")
            else:
                reasons.append(f"✗ RSI neutral at {market_data.rsi:.1f} (≤65)")
            total_indicators += 1
            
            # MACD bearish check
            if market_data.macd < market_data.macd_signal:
                indicators_aligned += 1
                reasons.append(f"✓ MACD bearish ({market_data.macd:.4f} < {market_data.macd_signal:.4f})")
            else:
                reasons.append(f"✗ MACD bullish ({market_data.macd:.4f} ≥ {market_data.macd_signal:.4f})")
            total_indicators += 1
            
            # Bollinger Bands upper check
            bb_pos = market_data.get_bb_position()
            if bb_pos > 0.75:
                indicators_aligned += 1
                reasons.append(f"✓ BB upper rejection at {bb_pos:.2f} (>0.75)")
            else:
                reasons.append(f"✗ BB position at {bb_pos:.2f} (≤0.75)")
            total_indicators += 1
            
            # Stochastic overbought check
            if market_data.stoch > 75:
                indicators_aligned += 1
                reasons.append(f"✓ Stoch overbought at {market_data.stoch:.1f} (>75)")
            else:
                reasons.append(f"✗ Stoch neutral at {market_data.stoch:.1f} (≤75)")
            total_indicators += 1
        
        confluence_score = indicators_aligned / total_indicators
        reasons.append(f"Confluence: {indicators_aligned}/{total_indicators} indicators aligned")
        
        return confluence_score, reasons
    
    def _check_timeframe_alignment(self, signal: TradingSignal,
                                 market_data: MarketData) -> Tuple[float, List[str]]:
        """Check if multiple timeframes agree with the signal"""
        
        reasons = []
        
        if not market_data.timeframe_data:
            # Fallback: Use moving average alignment on current timeframe
            reasons.append("Using MA alignment (no multi-timeframe data)")
            reasons.append(f"EMA9: {market_data.ema_9:.6f} | EMA21: {market_data.ema_21:.6f}")
            reasons.append(f"SMA50: {market_data.sma_50:.6f} | SMA200: {market_data.sma_200:.6f}")
            
            if signal.action == "BUY":
                aligned = market_data.get_ma_alignment_bullish()
                if aligned:
                    reasons.append("✓ Bullish MA alignment: EMA9>EMA21>SMA50>SMA200")
                else:
                    reasons.append("✗ No bullish MA alignment")
            else:
                aligned = market_data.get_ma_alignment_bearish()
                if aligned:
                    reasons.append("✓ Bearish MA alignment: EMA9<EMA21<SMA50<SMA200")
                else:
                    reasons.append("✗ No bearish MA alignment")
            
            return (0.8 if aligned else 0.3), reasons
        
        # Full multi-timeframe analysis
        timeframes = ['1h', '4h', '1d']
        aligned_timeframes = 0
        
        for tf in timeframes:
            if tf in market_data.timeframe_data:
                tf_data = market_data.timeframe_data[tf]
                tf_rsi = tf_data.get('rsi', 50)
                
                if signal.action == "BUY":
                    # Look for oversold conditions
                    if tf_rsi < 45:  # More lenient for multi-timeframe
                        aligned_timeframes += 1
                        reasons.append(f"✓ {tf}: RSI {tf_rsi:.1f} oversold (<45)")
                    else:
                        reasons.append(f"✗ {tf}: RSI {tf_rsi:.1f} neutral (≥45)")
                else:  # SELL
                    if tf_rsi > 55:  # More lenient for multi-timeframe
                        aligned_timeframes += 1
                        reasons.append(f"✓ {tf}: RSI {tf_rsi:.1f} overbought (>55)")
                    else:
                        reasons.append(f"✗ {tf}: RSI {tf_rsi:.1f} neutral (≤55)")
            else:
                reasons.append(f"✗ {tf}: No data available")
        
        alignment_score = aligned_timeframes / len(timeframes)
        reasons.append(f"Alignment: {aligned_timeframes}/{len(timeframes)} timeframes")
        
        return alignment_score, reasons
    
    def _check_volume_confirmation(self, signal: TradingSignal,
                                 market_data: MarketData) -> Tuple[float, List[str]]:
        """Check if volume supports the signal"""
        
        volume_ratio = market_data.volume_ratio
        current_volume = market_data.current_volume
        avg_volume = market_data.avg_volume_20
        reasons = []
        
        # Add detailed quantitative information
        reasons.append(f"Current: {current_volume:,.0f} | Avg(20): {avg_volume:,.0f}")
        
        if volume_ratio >= 2.5:
            score = 1.0
            reasons.append(f"Exceptional volume {volume_ratio:.1f}x average")
        elif volume_ratio >= 2.0:
            score = 0.9
            reasons.append(f"Very high volume {volume_ratio:.1f}x average")
        elif volume_ratio >= 1.5:
            score = 0.7
            reasons.append(f"High volume {volume_ratio:.1f}x average")
        elif volume_ratio >= 1.0:
            score = 0.4
            reasons.append(f"Average volume {volume_ratio:.1f}x (20-period avg)")
        else:
            score = 0.1
            reasons.append(f"Low volume {volume_ratio:.1f}x average")
        
        return score, reasons
    
    def _check_market_structure(self, signal: TradingSignal,
                              market_data: MarketData) -> Tuple[float, List[str]]:
        """Check if market structure supports the signal"""
        
        reasons = []
        structure_score = 0.5  # Default neutral
        
        # Check price relative to key moving averages
        price = market_data.current_price
        
        # Add quantitative price levels
        reasons.append(f"Price: ${price:.6f}")
        reasons.append(f"SMA20: ${market_data.sma_20:.6f} | SMA50: ${market_data.sma_50:.6f}")
        reasons.append(f"BB: ${market_data.bb_lower:.6f} - ${market_data.bb_upper:.6f}")
        
        if signal.action == "BUY":
            # For BUY: Want price near support levels
            distance_to_sma20 = ((price - market_data.sma_20) / market_data.sma_20) * 100
            if price <= market_data.sma_20:
                structure_score += 0.2
                reasons.append(f"Below SMA20 by {abs(distance_to_sma20):.1f}% (support)")
            
            distance_to_bb_lower = ((price - market_data.bb_lower) / market_data.bb_lower) * 100
            if price <= market_data.bb_lower * 1.01:  # Within 1% of lower band
                structure_score += 0.3
                reasons.append(f"Near BB lower ({distance_to_bb_lower:+.1f}%)")
                
        else:  # SELL
            # For SELL: Want price near resistance levels
            distance_to_sma20 = ((price - market_data.sma_20) / market_data.sma_20) * 100
            if price >= market_data.sma_20:
                structure_score += 0.2
                reasons.append(f"Above SMA20 by {distance_to_sma20:.1f}% (resistance)")
            
            distance_to_bb_upper = ((price - market_data.bb_upper) / market_data.bb_upper) * 100
            if price >= market_data.bb_upper * 0.99:  # Within 1% of upper band
                structure_score += 0.3
                reasons.append(f"Near BB upper ({distance_to_bb_upper:+.1f}%)")
        
        # Ensure score stays in bounds
        structure_score = max(0.0, min(1.0, structure_score))
        
        return structure_score, reasons
    
    def _check_risk_reward_quality(self, signal: TradingSignal) -> Tuple[float, List[str]]:
        """Check if risk/reward ratio is acceptable"""
        
        rr_ratio = signal.risk_reward_ratio
        reasons = []
        
        # Show actual calculation details
        entry = signal.entry_price
        stop = signal.stop_loss
        target = signal.take_profit
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        reasons.append(f"Entry: ${entry:.6f} | Stop: ${stop:.6f} | Target: ${target:.6f}")
        reasons.append(f"Risk: ${risk:.6f} | Reward: ${reward:.6f}")
        
        if rr_ratio >= 3.0:
            score = 1.0
            reasons.append(f"✓ Excellent R:R {rr_ratio:.2f}:1 (≥3.0)")
        elif rr_ratio >= 2.0:
            score = 0.8
            reasons.append(f"✓ Good R:R {rr_ratio:.2f}:1 (≥2.0)")
        elif rr_ratio >= 1.5:
            score = 0.6
            reasons.append(f"⚠ Acceptable R:R {rr_ratio:.2f}:1 (≥1.5)")
        elif rr_ratio >= 1.0:
            score = 0.3
            reasons.append(f"⚠ Marginal R:R {rr_ratio:.2f}:1 (≥1.0)")
        else:
            score = 0.0
            reasons.append(f"✗ Poor R:R {rr_ratio:.2f}:1 (<1.0)")
        
        return score, reasons
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics"""
        
        total = self.performance_stats['total_validated']
        if total == 0:
            return {'total_validated': 0}
        
        return {
            'total_validated': total,
            'execution_rate': (self.performance_stats['executed'] / total) * 100,
            'approval_rate': (self.performance_stats['approved'] / total) * 100,
            'marginal_rate': (self.performance_stats['marginal'] / total) * 100,
            'rejection_rate': (self.performance_stats['rejected'] / total) * 100,
            'avg_validation_time_ms': sum(v['validation_time_ms'] for v in self.validation_history[-100:]) / min(len(self.validation_history), 100)
        }
    
    def check_signal_confirmations(self, symbol: str, market_data: Dict[str, Any] = None) -> bool:
        """
        Legacy compatibility method for existing code
        
        In professional mode, this always returns True since we use instant validation
        instead of time-based confirmation delays.
        """
        logger.debug(f"🔄 Legacy signal confirmation check for {symbol} - professional mode always returns True")
        return True

# Test the professional validator
def test_professional_validation():
    """Test the professional validation system"""
    
    print("🧪 TESTING PROFESSIONAL SIGNAL VALIDATION")
    print("=" * 60)
    
    validator = ProfessionalSignalValidator()
    
    # Test Case 1: High-quality BUY signal
    print("Test 1: High-quality BUY signal")
    
    signal1 = TradingSignal(
        symbol="BTC/USDT",
        action="BUY",
        confidence=75.0,
        entry_price=67000.0,
        stop_loss=65500.0,  # 2.2% risk
        take_profit=70000.0,  # 4.5% reward = 2:1 R/R
        source="test",
        reason="Multi-indicator oversold",
        timestamp=datetime.now()
    )
    
    market_data1 = MarketData(
        symbol="BTC/USDT",
        current_price=67000.0,
        timestamp=datetime.now(),
        rsi=28.0,  # Oversold
        macd=150.0,
        macd_signal=100.0,  # Bullish crossover
        bb_upper=70000.0,
        bb_lower=65000.0,  # At lower band
        bb_middle=67500.0,
        stoch=18.0,  # Oversold
        sma_20=67200.0,
        sma_50=68000.0,
        sma_200=69000.0,
        ema_9=66800.0,
        ema_21=67100.0,
        current_volume=1500000.0,
        avg_volume_20=800000.0,
        volume_ratio=1.875  # High volume
    )
    
    result1 = validator.validate_signal_instantly(signal1, market_data1)
    print(f"   Result: {result1['result'].value}")
    print(f"   Score: {result1['score']:.1%}")
    print(f"   Confidence adjustment: {result1['confidence_adjustment']:+d}")
    print(f"   Validation time: {result1['validation_time_ms']:.1f}ms")
    print(f"   Reasons: {len(result1['reasons'])} factors")
    
    # Test Case 2: Low-quality SELL signal
    print("\\nTest 2: Low-quality SELL signal")
    
    signal2 = TradingSignal(
        symbol="ETH/USDT", 
        action="SELL",
        confidence=60.0,
        entry_price=4200.0,
        stop_loss=4250.0,
        take_profit=4150.0,
        source="test",
        reason="Weak bearish signal",
        timestamp=datetime.now()
    )
    
    market_data2 = MarketData(
        symbol="ETH/USDT",
        current_price=4200.0,
        timestamp=datetime.now(),
        rsi=55.0,  # Neutral
        macd=-10.0,
        macd_signal=-5.0,  # Weak bearish
        bb_upper=4300.0,
        bb_lower=4100.0,
        bb_middle=4200.0,  # Middle of range
        stoch=45.0,  # Neutral
        sma_20=4180.0,
        sma_50=4160.0,
        sma_200=4140.0,
        ema_9=4195.0,
        ema_21=4190.0,
        current_volume=600000.0,
        avg_volume_20=800000.0,
        volume_ratio=0.75  # Low volume
    )
    
    result2 = validator.validate_signal_instantly(signal2, market_data2)
    print(f"   Result: {result2['result'].value}")
    print(f"   Score: {result2['score']:.1%}")
    print(f"   Validation time: {result2['validation_time_ms']:.1f}ms")
    
    # Show validation statistics
    print("\\n📊 VALIDATION STATISTICS:")
    stats = validator.get_validation_stats()
    print(f"   Total validated: {stats['total_validated']}")
    print(f"   Average validation time: {stats['avg_validation_time_ms']:.1f}ms")
    print(f"   Execution rate: {stats.get('execution_rate', 0):.1f}%")
    print(f"   Rejection rate: {stats.get('rejection_rate', 0):.1f}%")
    
    # Performance comparison
    print("\\n⚡ PERFORMANCE COMPARISON:")
    print(f"   OLD SYSTEM: 30-120 minutes per validation")
    print(f"   NEW SYSTEM: {stats['avg_validation_time_ms']:.1f}ms per validation")
    print(f"   SPEED IMPROVEMENT: {(30*60*1000)/stats['avg_validation_time_ms']:.0f}x FASTER!")
    
    if result1['result'] != ValidationResult.REJECTED and result2['result'] == ValidationResult.REJECTED:
        print("\\n✅ PROFESSIONAL VALIDATION WORKING!")
        print("   - Quality signals approved instantly")
        print("   - Poor signals rejected properly")
        print("   - Validation time under 1 second")
    else:
        print("\\n❌ Validation needs adjustment")

if __name__ == "__main__":
    test_professional_validation()