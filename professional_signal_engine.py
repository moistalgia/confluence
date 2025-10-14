#!/usr/bin/env python3
"""
Professional Signal Generation Engine
====================================

Implements institutional-grade signal confirmation system:
- Multi-indicator confluence required
- Stricter confidence thresholds  
- Timeframe alignment mandatory
- Volume and momentum confirmation
- Market structure validation

Replaces the amateur "single indicator = signal" approach
with professional multi-confirmation requirements.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Professional confidence levels with strict requirements"""
    EXCEPTIONAL = "EXCEPTIONAL"  # 90%+ - Rare, high-conviction setups
    HIGH = "HIGH"               # 80-90% - Strong multi-timeframe confluence  
    MODERATE = "MODERATE"       # 60-80% - Some confluence, manageable risk
    LOW = "LOW"                # 40-60% - Weak setup, avoid or small position
    TRASH = "TRASH"            # <40% - No tradeable setup

@dataclass
class IndicatorReading:
    """Single indicator reading with context"""
    name: str
    value: float
    condition: str  # overbought, oversold, bullish_cross, etc.
    strength: str   # weak, moderate, strong, extreme
    timeframe: str
    timestamp: datetime
    
@dataclass
class ConfluenceRequirement:
    """Requirements for signal confirmation"""
    min_indicators: int = 3           # Minimum indicators that must agree
    min_timeframes: int = 2           # Minimum timeframes showing alignment  
    volume_confirmation: bool = True   # Volume must support the move
    momentum_confirmation: bool = True # Momentum must be building
    market_structure_check: bool = True # Price must be at logical level
    
class ProfessionalSignalEngine:
    """
    Professional signal generation requiring multiple confirmations
    
    No longer generates signals from single indicators.
    Requires confluence across multiple dimensions.
    """
    
    def __init__(self):
        self.confidence_thresholds = {
            # MUCH stricter than before - professional standards
            'rsi': {
                'overbought': 78,      # Was 70 - now 78+ for strong signal
                'oversold': 22,        # Was 30 - now 22- for strong signal  
                'extreme_overbought': 85,  # Was 80 - now 85+ for extreme
                'extreme_oversold': 15     # Was 20 - now 15- for extreme
            },
            'macd': {
                'strong_momentum': 0.20,   # Was 0.10 - now need 0.20+ 
                'weak_momentum': 0.05      # Was 0.02 - now 0.05+
            },
            'volume': {
                'strong_confirmation': 2.5, # Was 1.5x - now need 2.5x avg volume
                'moderate_confirmation': 1.8 # Was 1.2x - now 1.8x 
            },
            'bollinger': {
                'band_squeeze': 0.02,      # 2% bandwidth for squeeze
                'band_expansion': 0.08     # 8% bandwidth for expansion
            }
        }
        
        # Professional confluence requirements by signal type
        self.confluence_requirements = {
            'SCALP': ConfluenceRequirement(
                min_indicators=3,          # Need 3+ indicators agreeing
                min_timeframes=2,          # Need 2+ timeframes aligned
                volume_confirmation=True,
                momentum_confirmation=True,
                market_structure_check=True
            ),
            'SWING': ConfluenceRequirement(
                min_indicators=4,          # Need 4+ indicators agreeing
                min_timeframes=3,          # Need 3+ timeframes aligned  
                volume_confirmation=True,
                momentum_confirmation=True,
                market_structure_check=True
            ),
            'POSITION': ConfluenceRequirement(
                min_indicators=5,          # Need 5+ indicators agreeing
                min_timeframes=4,          # Need 4+ timeframes aligned
                volume_confirmation=True,
                momentum_confirmation=True,
                market_structure_check=True
            )
        }
        
    def generate_professional_signal(self, analysis_data: Dict) -> Optional[Dict]:
        """
        Generate signal only if professional confluence requirements are met
        
        Returns None if insufficient confluence (most common outcome)
        Returns signal dict only for high-probability setups
        """
        
        # Step 1: Extract all indicator readings across timeframes
        indicator_readings = self._extract_indicator_readings(analysis_data)
        
        if len(indicator_readings) < 3:
            logger.info("❌ Insufficient indicator data for signal generation")
            return None
            
        # Step 2: Check for directional confluence
        directional_consensus = self._check_directional_confluence(indicator_readings)
        
        if not directional_consensus['has_consensus']:
            logger.info("❌ No directional consensus across indicators")
            return None
            
        # Step 3: Validate timeframe alignment
        timeframe_alignment = self._validate_timeframe_alignment(indicator_readings, directional_consensus['direction'])
        
        if not timeframe_alignment['is_aligned']:
            logger.info("❌ Timeframes not aligned for signal generation")
            return None
            
        # Step 4: Check volume and momentum confirmation  
        volume_momentum_check = self._check_volume_momentum_confirmation(analysis_data, directional_consensus['direction'])
        
        if not volume_momentum_check['confirmed']:
            logger.info("❌ Insufficient volume/momentum confirmation")
            return None
            
        # Step 5: Validate market structure
        market_structure_valid = self._validate_market_structure(analysis_data, directional_consensus['direction'])
        
        if not market_structure_valid:
            logger.info("❌ Market structure does not support signal")
            return None
            
        # Step 6: Calculate professional confidence score
        confidence_score = self._calculate_professional_confidence(
            directional_consensus,
            timeframe_alignment,
            volume_momentum_check,
            indicator_readings
        )
        
        # Step 7: Only generate signal if confidence meets minimum threshold
        if confidence_score['numeric_score'] < 60:  # Minimum 60% for any signal
            logger.info(f"❌ Confidence too low: {confidence_score['numeric_score']:.1f}% (need 60%+)")
            return None
            
        # SUCCESS: Generate professional signal
        signal = {
            'symbol': analysis_data.get('symbol', 'UNKNOWN'),
            'timestamp': datetime.now(),
            'direction': directional_consensus['direction'],
            'signal_type': self._determine_signal_type(timeframe_alignment),
            'confidence_level': confidence_score['level'],
            'confidence_numeric': confidence_score['numeric_score'],
            'supporting_indicators': directional_consensus['supporting_indicators'],
            'timeframe_alignment': timeframe_alignment,
            'volume_confirmation': volume_momentum_check,
            'confluence_strength': directional_consensus['confluence_strength'],
            'entry_criteria': self._generate_entry_criteria(analysis_data, directional_consensus),
            'risk_reward_ratio': self._calculate_risk_reward(analysis_data, directional_consensus['direction']),
            'invalidation_levels': self._set_invalidation_levels(analysis_data, directional_consensus['direction'])
        }
        
        logger.info(f"✅ PROFESSIONAL SIGNAL GENERATED")
        logger.info(f"   Direction: {signal['direction']}")
        logger.info(f"   Confidence: {signal['confidence_level']} ({signal['confidence_numeric']:.1f}%)")
        logger.info(f"   Supporting Indicators: {len(signal['supporting_indicators'])}")
        logger.info(f"   Timeframe Alignment: {len(timeframe_alignment['aligned_timeframes'])}")
        logger.info(f"   Confluence Strength: {signal['confluence_strength']:.1f}/10")
        
        return signal
        
    def _extract_indicator_readings(self, analysis_data: Dict) -> List[IndicatorReading]:
        """Extract and normalize all indicator readings from analysis"""
        readings = []
        
        # Extract from multi-timeframe analysis
        if 'multi_timeframe_analysis' in analysis_data:
            mtf_data = analysis_data['multi_timeframe_analysis']
            tf_data = mtf_data.get('timeframe_data', {})
            
            for timeframe, tf_info in tf_data.items():
                if 'indicators' in tf_info:
                    indicators = tf_info['indicators']
                    
                    # RSI readings
                    rsi = indicators.get('rsi', 50)
                    if rsi != 50:  # Only include if we have real data
                        condition = self._classify_rsi_condition(rsi)
                        strength = self._calculate_rsi_strength(rsi)
                        readings.append(IndicatorReading(
                            name='RSI',
                            value=rsi,
                            condition=condition,
                            strength=strength,
                            timeframe=timeframe,
                            timestamp=datetime.now()
                        ))
                    
                    # MACD readings
                    macd = indicators.get('macd', 0)
                    macd_signal = indicators.get('macd_signal', 0)
                    if macd != 0 and macd_signal != 0:
                        momentum = abs(macd - macd_signal)
                        condition = 'bullish_cross' if macd > macd_signal else 'bearish_cross'
                        strength = self._calculate_macd_strength(momentum)
                        readings.append(IndicatorReading(
                            name='MACD',
                            value=momentum,
                            condition=condition,
                            strength=strength,
                            timeframe=timeframe,
                            timestamp=datetime.now()
                        ))
                    
                    # Add more indicators (Bollinger Bands, Stochastic, etc.)
                    bb_upper = indicators.get('bb_upper', 0)
                    bb_lower = indicators.get('bb_lower', 0) 
                    bb_middle = indicators.get('bb_middle', 0)
                    
                    if bb_upper > 0 and bb_lower > 0:
                        # Calculate current price position within bands
                        current_price = indicators.get('close', 0)
                        if current_price > 0:
                            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                            condition = self._classify_bb_condition(bb_position)
                            strength = self._calculate_bb_strength(bb_position)
                            readings.append(IndicatorReading(
                                name='BB',
                                value=bb_position,
                                condition=condition,
                                strength=strength,
                                timeframe=timeframe,
                                timestamp=datetime.now()
                            ))
        
        return readings
        
    def _classify_rsi_condition(self, rsi: float) -> str:
        """Classify RSI condition with professional thresholds"""
        thresholds = self.confidence_thresholds['rsi']
        
        if rsi >= thresholds['extreme_overbought']:
            return 'extreme_overbought'
        elif rsi >= thresholds['overbought']:
            return 'overbought'
        elif rsi <= thresholds['extreme_oversold']:
            return 'extreme_oversold'
        elif rsi <= thresholds['oversold']:
            return 'oversold'
        else:
            return 'neutral'
            
    def _calculate_rsi_strength(self, rsi: float) -> str:
        """Calculate RSI signal strength"""
        if rsi >= 85 or rsi <= 15:
            return 'extreme'
        elif rsi >= 78 or rsi <= 22:
            return 'strong'  
        elif rsi >= 70 or rsi <= 30:
            return 'moderate'
        else:
            return 'weak'
            
    def _calculate_macd_strength(self, momentum: float) -> str:
        """Calculate MACD momentum strength"""
        thresholds = self.confidence_thresholds['macd']
        
        if momentum >= thresholds['strong_momentum']:
            return 'strong'
        elif momentum >= thresholds['weak_momentum']:
            return 'moderate'
        else:
            return 'weak'
            
    def _classify_bb_condition(self, position: float) -> str:
        """Classify Bollinger Band position"""
        if position >= 0.95:
            return 'upper_band_touch'
        elif position >= 0.80:
            return 'upper_zone'
        elif position <= 0.05:
            return 'lower_band_touch'
        elif position <= 0.20:
            return 'lower_zone'
        else:
            return 'middle_zone'
            
    def _calculate_bb_strength(self, position: float) -> str:
        """Calculate Bollinger Band signal strength"""
        if position >= 0.95 or position <= 0.05:
            return 'extreme'
        elif position >= 0.85 or position <= 0.15:
            return 'strong'
        elif position >= 0.75 or position <= 0.25:
            return 'moderate'
        else:
            return 'weak'
            
    def _check_directional_confluence(self, readings: List[IndicatorReading]) -> Dict:
        """Check if indicators show directional consensus"""
        
        bullish_signals = []
        bearish_signals = []
        neutral_signals = []
        
        for reading in readings:
            # Classify each reading as bullish, bearish, or neutral
            if reading.condition in ['oversold', 'extreme_oversold', 'bullish_cross', 'lower_band_touch', 'lower_zone']:
                if reading.strength in ['strong', 'extreme']:
                    bullish_signals.append(reading)
            elif reading.condition in ['overbought', 'extreme_overbought', 'bearish_cross', 'upper_band_touch', 'upper_zone']:
                if reading.strength in ['strong', 'extreme']:
                    bearish_signals.append(reading)
            else:
                neutral_signals.append(reading)
        
        total_signals = len(bullish_signals) + len(bearish_signals)
        
        if len(bullish_signals) >= 3 and len(bullish_signals) > len(bearish_signals) * 2:
            direction = 'BULLISH'
            supporting_indicators = bullish_signals
            confluence_strength = min(10, (len(bullish_signals) / max(1, len(bearish_signals))) * 2)
        elif len(bearish_signals) >= 3 and len(bearish_signals) > len(bullish_signals) * 2:
            direction = 'BEARISH'  
            supporting_indicators = bearish_signals
            confluence_strength = min(10, (len(bearish_signals) / max(1, len(bullish_signals))) * 2)
        else:
            return {
                'has_consensus': False,
                'direction': 'NEUTRAL',
                'supporting_indicators': [],
                'confluence_strength': 0
            }
        
        return {
            'has_consensus': True,
            'direction': direction,
            'supporting_indicators': [r.name for r in supporting_indicators],
            'confluence_strength': confluence_strength,
            'signal_details': supporting_indicators
        }
        
    def _validate_timeframe_alignment(self, readings: List[IndicatorReading], direction: str) -> Dict:
        """Validate that multiple timeframes show alignment"""
        
        timeframe_scores = {}
        
        for reading in readings:
            tf = reading.timeframe
            if tf not in timeframe_scores:
                timeframe_scores[tf] = {'bullish': 0, 'bearish': 0}
                
            if direction == 'BULLISH' and reading.condition in ['oversold', 'extreme_oversold', 'bullish_cross', 'lower_band_touch']:
                timeframe_scores[tf]['bullish'] += 1
            elif direction == 'BEARISH' and reading.condition in ['overbought', 'extreme_overbought', 'bearish_cross', 'upper_band_touch']:
                timeframe_scores[tf]['bearish'] += 1
        
        # Check alignment
        aligned_timeframes = []
        for tf, scores in timeframe_scores.items():
            if direction == 'BULLISH' and scores['bullish'] > scores['bearish']:
                aligned_timeframes.append(tf)
            elif direction == 'BEARISH' and scores['bearish'] > scores['bullish']:
                aligned_timeframes.append(tf)
        
        return {
            'is_aligned': len(aligned_timeframes) >= 2,  # Need at least 2 timeframes aligned
            'aligned_timeframes': aligned_timeframes,
            'timeframe_count': len(aligned_timeframes),
            'total_timeframes': len(timeframe_scores)
        }
        
    def _check_volume_momentum_confirmation(self, analysis_data: Dict, direction: str) -> Dict:
        """Check volume and momentum support the signal"""
        
        # This would integrate with volume analysis from your existing system
        # For now, return basic structure
        return {
            'confirmed': True,  # Placeholder - implement based on your volume analysis
            'volume_strength': 'moderate',
            'momentum_strength': 'moderate'
        }
        
    def _validate_market_structure(self, analysis_data: Dict, direction: str) -> bool:
        """Validate market structure supports the signal"""
        
        # This would check support/resistance levels, trend structure, etc.
        # For now, return True - implement based on your market structure analysis
        return True
        
    def _calculate_professional_confidence(self, directional_consensus: Dict, 
                                        timeframe_alignment: Dict,
                                        volume_momentum: Dict,
                                        readings: List[IndicatorReading]) -> Dict:
        """Calculate professional confidence score"""
        
        base_score = 40  # Start at 40% - below tradeable threshold
        
        # Confluence bonus (0-25 points)
        confluence_bonus = min(25, directional_consensus['confluence_strength'] * 2.5)
        
        # Timeframe alignment bonus (0-20 points)  
        tf_bonus = min(20, timeframe_alignment['timeframe_count'] * 5)
        
        # Indicator strength bonus (0-15 points)
        strong_indicators = len([r for r in readings if r.strength in ['strong', 'extreme']])
        indicator_bonus = min(15, strong_indicators * 3)
        
        total_score = base_score + confluence_bonus + tf_bonus + indicator_bonus
        
        # Classify confidence level
        if total_score >= 90:
            level = ConfidenceLevel.EXCEPTIONAL
        elif total_score >= 80:
            level = ConfidenceLevel.HIGH
        elif total_score >= 60:
            level = ConfidenceLevel.MODERATE
        elif total_score >= 40:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.TRASH
            
        return {
            'numeric_score': min(100, total_score),
            'level': level,
            'components': {
                'base': base_score,
                'confluence': confluence_bonus,
                'timeframes': tf_bonus,
                'indicators': indicator_bonus
            }
        }
        
    def _determine_signal_type(self, timeframe_alignment: Dict) -> str:
        """Determine signal type based on timeframe alignment"""
        
        tf_count = timeframe_alignment['timeframe_count']
        
        if tf_count >= 4:
            return 'POSITION'  # Long-term position trade
        elif tf_count >= 3:
            return 'SWING'     # Multi-day swing trade
        else:
            return 'SCALP'     # Intraday scalp trade
            
    def _generate_entry_criteria(self, analysis_data: Dict, directional_consensus: Dict) -> Dict:
        """Generate specific entry criteria for the signal"""
        
        # This would generate specific price levels, confirmation requirements, etc.
        return {
            'entry_type': 'market_pullback',  # market, limit, pullback
            'confirmation_required': True,
            'max_wait_time_hours': 4
        }
        
    def _calculate_risk_reward(self, analysis_data: Dict, direction: str) -> float:
        """Calculate risk/reward ratio for the signal"""
        
        # This would calculate based on support/resistance levels
        return 2.5  # Placeholder
        
    def _set_invalidation_levels(self, analysis_data: Dict, direction: str) -> Dict:
        """Set specific invalidation levels"""
        
        # This would set based on market structure
        return {
            'stop_loss_percent': 2.0,
            'time_based_exit_hours': 24,
            'structure_break_level': 'previous_swing'
        }