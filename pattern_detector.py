#!/usr/bin/env python3
"""
Pattern Detector - Advanced Chart Pattern Recognition
Implements professional-grade pattern detection algorithms
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy.signal import argrelextrema
from scipy.stats import linregress

logger = logging.getLogger(__name__)

class AdvancedPatternDetector:
    """Professional chart pattern detection system"""
    
    def __init__(self):
        self.min_pattern_length = 10
        self.confidence_threshold = 0.6
        
    def detect_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect all available patterns"""
        patterns = []
        
        if len(df) < self.min_pattern_length:
            return patterns
        
        try:
            # Reversal patterns
            patterns.extend(self._detect_double_top_bottom(df))
            patterns.extend(self._detect_head_shoulders(df))
            patterns.extend(self._detect_wedge_patterns(df))
            
            # Continuation patterns
            patterns.extend(self._detect_flag_pennant(df))
            patterns.extend(self._detect_triangle_patterns(df))
            
            # Trend patterns
            patterns.extend(self._detect_channel_patterns(df))
            patterns.extend(self._detect_breakout_patterns(df))
            
            # Candlestick patterns
            patterns.extend(self._detect_candlestick_patterns(df))
            
        except Exception as e:
            logger.warning(f"Pattern detection error: {e}")
            
        return sorted(patterns, key=lambda x: x.get('confidence_score', 0), reverse=True)
    
    def _detect_double_top_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double top/bottom patterns"""
        patterns = []
        
        # Find local extrema
        high_indices = argrelextrema(df['high'].values, np.greater, order=5)[0]
        low_indices = argrelextrema(df['low'].values, np.less, order=5)[0]
        
        # Double top detection
        if len(high_indices) >= 2:
            for i in range(len(high_indices) - 1):
                idx1, idx2 = high_indices[i], high_indices[i + 1]
                height1, height2 = df['high'].iloc[idx1], df['high'].iloc[idx2]
                
                # Check if heights are similar (within 2%)
                if abs(height1 - height2) / height1 < 0.02:
                    # Check for valley between peaks
                    valley_low = df['low'].iloc[idx1:idx2+1].min()
                    if valley_low < min(height1, height2) * 0.95:
                        patterns.append({
                            'type': 'double_top',
                            'confidence': 'HIGH' if abs(height1 - height2) / height1 < 0.01 else 'MEDIUM',
                            'confidence_score': 85 if abs(height1 - height2) / height1 < 0.01 else 70,
                            'resistance_level': max(height1, height2),
                            'support_level': valley_low,
                            'pattern_start': idx1,
                            'pattern_end': idx2,
                            'target': valley_low - (max(height1, height2) - valley_low)
                        })
        
        # Double bottom detection  
        if len(low_indices) >= 2:
            for i in range(len(low_indices) - 1):
                idx1, idx2 = low_indices[i], low_indices[i + 1]
                low1, low2 = df['low'].iloc[idx1], df['low'].iloc[idx2]
                
                # Check if lows are similar (within 2%)
                if abs(low1 - low2) / low1 < 0.02:
                    # Check for peak between lows
                    peak_high = df['high'].iloc[idx1:idx2+1].max()
                    if peak_high > max(low1, low2) * 1.05:
                        patterns.append({
                            'type': 'double_bottom',
                            'confidence': 'HIGH' if abs(low1 - low2) / low1 < 0.01 else 'MEDIUM',
                            'confidence_score': 85 if abs(low1 - low2) / low1 < 0.01 else 70,
                            'support_level': min(low1, low2),
                            'resistance_level': peak_high,
                            'pattern_start': idx1,
                            'pattern_end': idx2,
                            'target': peak_high + (peak_high - min(low1, low2))
                        })
        
        return patterns
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """Detect head and shoulders patterns"""
        patterns = []
        
        # Find prominent peaks
        high_indices = argrelextrema(df['high'].values, np.greater, order=7)[0]
        
        if len(high_indices) >= 3:
            for i in range(len(high_indices) - 2):
                left_shoulder = high_indices[i]
                head = high_indices[i + 1] 
                right_shoulder = high_indices[i + 2]
                
                left_height = df['high'].iloc[left_shoulder]
                head_height = df['high'].iloc[head]
                right_height = df['high'].iloc[right_shoulder]
                
                # Head should be higher than shoulders
                if (head_height > left_height and head_height > right_height and 
                    abs(left_height - right_height) / left_height < 0.05):
                    
                    # Find neckline (lows between shoulders and head)
                    left_valley = df['low'].iloc[left_shoulder:head].min()
                    right_valley = df['low'].iloc[head:right_shoulder+1].min()
                    neckline = (left_valley + right_valley) / 2
                    
                    patterns.append({
                        'type': 'head_and_shoulders',
                        'confidence': 'HIGH' if abs(left_height - right_height) / left_height < 0.02 else 'MEDIUM',
                        'confidence_score': 80,
                        'neckline': neckline,
                        'head_level': head_height,
                        'pattern_start': left_shoulder,
                        'pattern_end': right_shoulder,
                        'target': neckline - (head_height - neckline)
                    })
        
        return patterns
    
    def _detect_wedge_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect rising/falling wedge patterns"""
        patterns = []
        
        if len(df) < 20:
            return patterns
            
        # Find trend lines
        recent_data = df.tail(20)
        
        # Get highs and lows for trend lines
        high_indices = argrelextrema(recent_data['high'].values, np.greater, order=2)[0]
        low_indices = argrelextrema(recent_data['low'].values, np.less, order=2)[0]
        
        if len(high_indices) >= 2 and len(low_indices) >= 2:
            # Calculate trend lines
            high_slope, _, high_r, _, _ = linregress(high_indices, recent_data['high'].iloc[high_indices])
            low_slope, _, low_r, _, _ = linregress(low_indices, recent_data['low'].iloc[low_indices])
            
            # Rising wedge: both lines rising, converging
            if high_slope > 0 and low_slope > 0 and high_slope < low_slope * 1.5:
                patterns.append({
                    'type': 'rising_wedge',
                    'confidence': 'HIGH' if abs(high_r) > 0.8 and abs(low_r) > 0.8 else 'MEDIUM',
                    'confidence_score': 75,
                    'direction': 'bearish',
                    'upper_trendline_slope': high_slope,
                    'lower_trendline_slope': low_slope
                })
            
            # Falling wedge: both lines falling, converging  
            elif high_slope < 0 and low_slope < 0 and abs(low_slope) > abs(high_slope) * 1.5:
                patterns.append({
                    'type': 'falling_wedge',
                    'confidence': 'HIGH' if abs(high_r) > 0.8 and abs(low_r) > 0.8 else 'MEDIUM',
                    'confidence_score': 75,
                    'direction': 'bullish',
                    'upper_trendline_slope': high_slope,
                    'lower_trendline_slope': low_slope
                })
        
        return patterns
    
    def _detect_flag_pennant(self, df: pd.DataFrame) -> List[Dict]:
        """Detect flag and pennant continuation patterns"""
        patterns = []
        
        if len(df) < 15:
            return patterns
            
        # Look for strong move followed by consolidation
        recent = df.tail(15)
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # Strong initial move (>5%)
        if abs(price_change) > 0.05:
            # Check for consolidation in latter half
            consolidation_data = recent.tail(8)
            high_low_range = (consolidation_data['high'].max() - consolidation_data['low'].min()) / consolidation_data['close'].mean()
            
            # Tight consolidation (<3% range)
            if high_low_range < 0.03:
                pattern_type = 'bull_flag' if price_change > 0 else 'bear_flag'
                patterns.append({
                    'type': pattern_type,
                    'confidence': 'MEDIUM',
                    'confidence_score': 65,
                    'direction': 'bullish' if price_change > 0 else 'bearish',
                    'flag_range': high_low_range,
                    'initial_move': price_change
                })
        
        return patterns
    
    def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []
        
        if len(df) < 20:
            return patterns
            
        recent = df.tail(20)
        
        # Find trend lines
        high_indices = argrelextrema(recent['high'].values, np.greater, order=2)[0]
        low_indices = argrelextrema(recent['low'].values, np.less, order=2)[0]
        
        if len(high_indices) >= 3 and len(low_indices) >= 3:
            # Calculate trend lines
            high_slope, _, high_r, _, _ = linregress(high_indices, recent['high'].iloc[high_indices])
            low_slope, _, low_r, _, _ = linregress(low_indices, recent['low'].iloc[low_indices])
            
            # Ascending triangle: flat resistance, rising support
            if abs(high_slope) < 0.001 and low_slope > 0.01:
                patterns.append({
                    'type': 'ascending_triangle',
                    'confidence': 'HIGH' if abs(high_r) > 0.7 and abs(low_r) > 0.7 else 'MEDIUM',
                    'confidence_score': 70,
                    'direction': 'bullish',
                    'resistance_level': recent['high'].iloc[high_indices].mean()
                })
            
            # Descending triangle: falling resistance, flat support
            elif high_slope < -0.01 and abs(low_slope) < 0.001:
                patterns.append({
                    'type': 'descending_triangle',
                    'confidence': 'HIGH' if abs(high_r) > 0.7 and abs(low_r) > 0.7 else 'MEDIUM',
                    'confidence_score': 70,
                    'direction': 'bearish',
                    'support_level': recent['low'].iloc[low_indices].mean()
                })
            
            # Symmetrical triangle: converging trend lines
            elif high_slope < 0 and low_slope > 0:
                patterns.append({
                    'type': 'symmetrical_triangle',
                    'confidence': 'MEDIUM',
                    'confidence_score': 60,
                    'direction': 'neutral',
                    'convergence': True
                })
        
        return patterns
    
    def _detect_channel_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect channel patterns"""
        patterns = []
        
        if len(df) < 25:
            return patterns
            
        # Parallel channel detection
        recent = df.tail(25)
        
        # Find trend lines
        high_indices = argrelextrema(recent['high'].values, np.greater, order=3)[0]
        low_indices = argrelextrema(recent['low'].values, np.less, order=3)[0]
        
        if len(high_indices) >= 2 and len(low_indices) >= 2:
            high_slope, _, high_r, _, _ = linregress(high_indices, recent['high'].iloc[high_indices])
            low_slope, _, low_r, _, _ = linregress(low_indices, recent['low'].iloc[low_indices])
            
            # Parallel lines (similar slopes)
            if abs(high_slope - low_slope) < 0.01 and abs(high_r) > 0.6 and abs(low_r) > 0.6:
                if high_slope > 0.01:
                    pattern_type = 'ascending_channel'
                    direction = 'bullish'
                elif high_slope < -0.01:
                    pattern_type = 'descending_channel'
                    direction = 'bearish'
                else:
                    pattern_type = 'horizontal_channel'
                    direction = 'neutral'
                
                patterns.append({
                    'type': pattern_type,
                    'confidence': 'HIGH' if abs(high_r) > 0.8 and abs(low_r) > 0.8 else 'MEDIUM',
                    'confidence_score': 75,
                    'direction': direction,
                    'channel_width': recent['high'].iloc[high_indices].mean() - recent['low'].iloc[low_indices].mean()
                })
        
        return patterns
    
    def _detect_breakout_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect breakout patterns"""
        patterns = []
        
        if len(df) < 10:
            return patterns
            
        current_price = df['close'].iloc[-1]
        
        # Recent high/low breakouts
        recent_20 = df.tail(20)
        recent_high = recent_20['high'].max()
        recent_low = recent_20['low'].min()
        
        # Volume confirmation
        avg_volume = df['volume'].tail(20).mean()
        recent_volume = df['volume'].iloc[-1]
        
        # Upside breakout
        if current_price > recent_high * 1.002:  # 0.2% above recent high
            patterns.append({
                'type': 'upside_breakout',
                'confidence': 'HIGH' if recent_volume > avg_volume * 1.5 else 'MEDIUM',
                'confidence_score': 80 if recent_volume > avg_volume * 1.5 else 60,
                'direction': 'bullish',
                'breakout_level': recent_high,
                'volume_confirmation': recent_volume > avg_volume * 1.2
            })
        
        # Downside breakdown
        elif current_price < recent_low * 0.998:  # 0.2% below recent low
            patterns.append({
                'type': 'downside_breakdown',
                'confidence': 'HIGH' if recent_volume > avg_volume * 1.5 else 'MEDIUM',
                'confidence_score': 80 if recent_volume > avg_volume * 1.5 else 60,
                'direction': 'bearish',
                'breakdown_level': recent_low,
                'volume_confirmation': recent_volume > avg_volume * 1.2
            })
        
        return patterns
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect key candlestick patterns"""
        patterns = []
        
        if len(df) < 3:
            return patterns
            
        # Get recent candles
        last3 = df.tail(3)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None
        
        # Doji pattern
        body_size = abs(last['close'] - last['open'])
        candle_range = last['high'] - last['low']
        
        if candle_range > 0 and body_size / candle_range < 0.1:
            patterns.append({
                'type': 'doji',
                'confidence': 'MEDIUM',
                'confidence_score': 55,
                'direction': 'neutral',
                'reversal_signal': True
            })
        
        # Hammer/Shooting star
        if prev is not None:
            lower_shadow = min(last['close'], last['open']) - last['low']
            upper_shadow = last['high'] - max(last['close'], last['open'])
            
            # Hammer (bullish reversal)
            if (lower_shadow > 2 * body_size and upper_shadow < body_size and 
                last['close'] > last['open'] and prev['close'] < prev['open']):
                patterns.append({
                    'type': 'hammer',
                    'confidence': 'HIGH',
                    'confidence_score': 70,
                    'direction': 'bullish',
                    'reversal_signal': True
                })
            
            # Shooting star (bearish reversal)
            elif (upper_shadow > 2 * body_size and lower_shadow < body_size and 
                  last['close'] < last['open'] and prev['close'] > prev['open']):
                patterns.append({
                    'type': 'shooting_star',
                    'confidence': 'HIGH',
                    'confidence_score': 70,
                    'direction': 'bearish',
                    'reversal_signal': True
                })
        
        # Engulfing patterns
        if prev is not None:
            # Bullish engulfing
            if (last['close'] > last['open'] and prev['close'] < prev['open'] and
                last['close'] > prev['open'] and last['open'] < prev['close']):
                patterns.append({
                    'type': 'bullish_engulfing',
                    'confidence': 'HIGH',
                    'confidence_score': 75,
                    'direction': 'bullish',
                    'reversal_signal': True
                })
            
            # Bearish engulfing
            elif (last['close'] < last['open'] and prev['close'] > prev['open'] and
                  last['close'] < prev['open'] and last['open'] > prev['close']):
                patterns.append({
                    'type': 'bearish_engulfing',
                    'confidence': 'HIGH', 
                    'confidence_score': 75,
                    'direction': 'bearish',
                    'reversal_signal': True
                })
        
        return patterns

def main():
    """Test pattern detection"""
    # This would normally be called with real OHLCV data
    print("Pattern Detector initialized")
    print("Available patterns: Double Top/Bottom, Head & Shoulders, Wedges, Triangles, Channels, Breakouts, Candlesticks")

if __name__ == "__main__":
    main()