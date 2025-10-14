"""
Filter 2: Setup Scanner
🎯 Identifies pullback-to-support setups in trending markets

Looking for:
- Clear trend (up/down)
- Pullback to key level (SMA, previous resistance/support)
- Bounce/rejection from that level
- Volume confirmation

Quality setups only - not every bounce is tradeable
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from core.data_provider import DataProvider, TechnicalIndicators

logger = logging.getLogger(__name__)

class SetupScanner:
    """
    Scans for high-quality pullback-to-support/resistance setups
    Only identifies setups with clear trend + pullback + bounce pattern
    """
    
    def __init__(self, config: Dict):
        """
        Initialize setup scanner with pattern parameters
        
        Args:
            config: Scanner configuration
        """
        # Trend requirements
        self.min_trend_bars = config.get('min_trend_bars', 10)          # Min bars in trend
        self.trend_angle_min = config.get('trend_angle_min', 0.01)      # Min SMA slope
        
        # Pullback requirements  
        self.pullback_depth_min = config.get('pullback_depth_min', 0.02) # 2% min pullback
        self.pullback_depth_max = config.get('pullback_depth_max', 0.08) # 8% max pullback
        self.max_pullback_bars = config.get('max_pullback_bars', 15)     # Max bars in pullback
        
        # Support/resistance levels
        self.support_touch_tolerance = config.get('support_tolerance', 0.005) # 0.5% tolerance
        self.min_level_strength = config.get('min_level_strength', 2)          # Min touches needed
        
        # Volume confirmation
        self.volume_surge_ratio = config.get('volume_surge_ratio', 1.5)  # 1.5x average volume
        
        logger.info("🎯 Setup Scanner initialized")
        logger.info(f"   Trend: {self.min_trend_bars} bars, {self.trend_angle_min:.1%} slope")
        logger.info(f"   Pullback: {self.pullback_depth_min:.1%} to {self.pullback_depth_max:.1%} depth")
        
    def scan_for_setup(self, symbol: str, data_provider: DataProvider) -> Dict:
        """
        Scan symbol for pullback-to-support setup
        
        Args:
            symbol: Trading pair to scan
            data_provider: Data source
            
        Returns:
            Dict with setup analysis and trade recommendation
        """
        try:
            # Get 4H data for setup scanning (better than 1H noise, more responsive than 1D)
            df = data_provider.get_ohlcv(symbol, '4h', days=30)
            
            if len(df) < 50:
                logger.warning(f"⚠️ {symbol}: Insufficient data for setup scan ({len(df)} bars)")
                return self._create_result(symbol, False, 0, ["Insufficient data"])
            
            # Add technical indicators
            df = self._add_indicators(df)
            
            # Step 1: Identify trend direction and strength
            trend_analysis = self._analyze_trend(df)
            
            if not trend_analysis['has_trend']:
                return self._create_result(
                    symbol, False, 1, 
                    ["No clear trend identified"],
                    trend_analysis
                )
            
            # Step 2: Look for pullback pattern
            pullback_analysis = self._analyze_pullback(df, trend_analysis)
            
            if not pullback_analysis['has_pullback']:
                return self._create_result(
                    symbol, False, 2,
                    ["No valid pullback found"], 
                    trend_analysis, pullback_analysis
                )
            
            # Step 3: Check for support/resistance level interaction
            level_analysis = self._analyze_level_interaction(df, trend_analysis, pullback_analysis)
            
            if not level_analysis['at_key_level']:
                return self._create_result(
                    symbol, False, 3,
                    ["Not at key support/resistance level"],
                    trend_analysis, pullback_analysis, level_analysis
                )
            
            # Step 4: Look for bounce/rejection signal
            bounce_analysis = self._analyze_bounce_signal(df, trend_analysis)
            
            if not bounce_analysis['has_bounce']:
                return self._create_result(
                    symbol, False, 4,
                    ["No bounce/rejection signal yet"],
                    trend_analysis, pullback_analysis, level_analysis, bounce_analysis
                )
            
            # Step 5: Volume confirmation
            volume_analysis = self._analyze_volume_confirmation(df)
            
            # Calculate setup quality score
            quality_score = self._calculate_quality_score(
                trend_analysis, pullback_analysis, level_analysis, 
                bounce_analysis, volume_analysis
            )
            
            # Determine if setup is tradeable (need high quality)
            is_tradeable = quality_score >= 7  # Out of 10
            
            result = self._create_result(
                symbol, is_tradeable, quality_score,
                [] if is_tradeable else ["Setup quality too low"],
                trend_analysis, pullback_analysis, level_analysis,
                bounce_analysis, volume_analysis
            )
            
            # Log result
            setup_type = f"{trend_analysis['direction']} trend pullback"
            status = "🎯 SETUP FOUND" if is_tradeable else "⚠️ SETUP WEAK"
            logger.info(f"{status} {symbol}: {setup_type} (quality: {quality_score}/10)")
            
            return result
            
        except Exception as e:
            logger.error(f"🚨 Setup scan failed for {symbol}: {str(e)}")
            return self._create_result(symbol, False, 0, [f"Scan error: {str(e)}"])
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators needed for setup analysis"""
        df = df.copy()
        
        # Moving averages for trend
        df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        
        # RSI for momentum
        df['rsi'] = TechnicalIndicators.rsi(df['close'], 14)
        
        # Volume average
        df['volume_sma'] = TechnicalIndicators.sma(df['volume'], 20)
        
        return df
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        Analyze trend direction and strength
        
        Returns:
            Dict with trend analysis
        """
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        # Check SMA alignment and slope
        sma_20_slope = (df['sma_20'].iloc[-1] - df['sma_20'].iloc[-10]) / df['sma_20'].iloc[-10]
        
        # Determine trend direction
        if current_price > sma_20 > sma_50 and sma_20_slope > self.trend_angle_min:
            direction = "UP"
            strength = min(abs(sma_20_slope) * 100, 10)  # Cap at 10
        elif current_price < sma_20 < sma_50 and sma_20_slope < -self.trend_angle_min:
            direction = "DOWN" 
            strength = min(abs(sma_20_slope) * 100, 10)  # Cap at 10
        else:
            direction = "SIDEWAYS"
            strength = 0
        
        has_trend = direction in ["UP", "DOWN"] and strength >= 3
        
        return {
            'has_trend': has_trend,
            'direction': direction,
            'strength': strength,
            'sma_20_slope': sma_20_slope,
            'price_vs_sma20': (current_price - sma_20) / sma_20
        }
    
    def _analyze_pullback(self, df: pd.DataFrame, trend: Dict) -> Dict:
        """
        Look for pullback pattern in the trend
        
        Returns:
            Dict with pullback analysis
        """
        if not trend['has_trend']:
            return {'has_pullback': False}
        
        recent_df = df.tail(self.max_pullback_bars + 5)  # Look at recent bars
        
        if trend['direction'] == "UP":
            # For uptrend, look for pullback from recent high
            high_idx = recent_df['high'].idxmax()
            high_price = recent_df.loc[high_idx, 'high']
            current_price = recent_df['close'].iloc[-1]
            
            pullback_depth = (high_price - current_price) / high_price
            
        else:  # DOWN trend
            # For downtrend, look for pullback from recent low  
            low_idx = recent_df['low'].idxmin()
            low_price = recent_df.loc[low_idx, 'low']
            current_price = recent_df['close'].iloc[-1]
            
            pullback_depth = (current_price - low_price) / low_price
        
        # Check if pullback is in valid range
        has_pullback = (self.pullback_depth_min <= pullback_depth <= self.pullback_depth_max)
        
        return {
            'has_pullback': has_pullback,
            'depth': pullback_depth,
            'direction': "UP" if trend['direction'] == "DOWN" else "DOWN"
        }
    
    def _analyze_level_interaction(self, df: pd.DataFrame, trend: Dict, pullback: Dict) -> Dict:
        """
        Check if price is at key support/resistance level
        
        Returns:
            Dict with level analysis
        """
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        
        # Key levels to check
        levels = []
        
        # SMA20 as dynamic support/resistance
        levels.append(('SMA20', sma_20))
        
        # Recent swing highs/lows as static levels
        recent_highs = df['high'].rolling(10, center=True).max().dropna()
        recent_lows = df['low'].rolling(10, center=True).min().dropna()
        
        for high in recent_highs.tail(5):
            levels.append(('Resistance', high))
            
        for low in recent_lows.tail(5):
            levels.append(('Support', low))
        
        # Check if current price is near any key level
        at_key_level = False
        closest_level = None
        min_distance = float('inf')
        
        for level_type, level_price in levels:
            distance = abs(current_price - level_price) / current_price
            
            if distance <= self.support_touch_tolerance:
                at_key_level = True
                if distance < min_distance:
                    min_distance = distance
                    closest_level = (level_type, level_price, distance)
        
        return {
            'at_key_level': at_key_level,
            'closest_level': closest_level,
            'all_levels': levels
        }
    
    def _analyze_bounce_signal(self, df: pd.DataFrame, trend: Dict) -> Dict:
        """
        Look for bounce/rejection signal from support/resistance
        
        Returns:
            Dict with bounce analysis
        """
        recent_bars = df.tail(5)  # Last 5 bars for bounce signal
        
        if trend['direction'] == "UP":
            # For uptrend pullback, look for bullish bounce
            # Green candle with higher low
            last_bar = recent_bars.iloc[-1]
            prev_bar = recent_bars.iloc[-2] if len(recent_bars) > 1 else last_bar
            
            is_green = last_bar['close'] > last_bar['open']
            higher_low = last_bar['low'] > prev_bar['low']
            rsi_oversold_bounce = df['rsi'].iloc[-1] > df['rsi'].iloc[-2] and df['rsi'].iloc[-2] < 40
            
            has_bounce = is_green and (higher_low or rsi_oversold_bounce)
            signal_type = "BULLISH_BOUNCE"
            
        else:  # DOWN trend
            # For downtrend pullback, look for bearish rejection
            # Red candle with lower high
            last_bar = recent_bars.iloc[-1]
            prev_bar = recent_bars.iloc[-2] if len(recent_bars) > 1 else last_bar
            
            is_red = last_bar['close'] < last_bar['open']
            lower_high = last_bar['high'] < prev_bar['high']
            rsi_overbought_rejection = df['rsi'].iloc[-1] < df['rsi'].iloc[-2] and df['rsi'].iloc[-2] > 60
            
            has_bounce = is_red and (lower_high or rsi_overbought_rejection)
            signal_type = "BEARISH_REJECTION"
        
        return {
            'has_bounce': has_bounce,
            'signal_type': signal_type,
            'rsi_current': df['rsi'].iloc[-1]
        }
    
    def _analyze_volume_confirmation(self, df: pd.DataFrame) -> Dict:
        """
        Check for volume confirmation of the setup
        
        Returns:
            Dict with volume analysis
        """
        recent_volume = df['volume'].iloc[-1]
        avg_volume = df['volume_sma'].iloc[-1]
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
        has_volume_surge = volume_ratio >= self.volume_surge_ratio
        
        return {
            'has_volume_confirmation': has_volume_surge,
            'volume_ratio': volume_ratio,
            'required_ratio': self.volume_surge_ratio
        }
    
    def _calculate_quality_score(self, trend: Dict, pullback: Dict, level: Dict, 
                                bounce: Dict, volume: Dict) -> int:
        """
        Calculate overall setup quality score (0-10)
        
        Returns:
            Integer quality score
        """
        score = 0
        
        # Trend strength (0-3 points)
        if trend['has_trend']:
            score += min(int(trend['strength']), 3)
        
        # Pullback quality (0-2 points) 
        if pullback['has_pullback']:
            # Better score for moderate pullbacks
            depth = pullback['depth']
            if 0.03 <= depth <= 0.06:  # Sweet spot
                score += 2
            else:
                score += 1
        
        # Level interaction (0-2 points)
        if level['at_key_level']:
            score += 2
        
        # Bounce signal (0-2 points)
        if bounce['has_bounce']:
            score += 2
        
        # Volume confirmation (0-1 point)
        if volume['has_volume_confirmation']:
            score += 1
        
        return min(score, 10)  # Cap at 10
    
    def _create_result(self, symbol: str, tradeable: bool, score: int, 
                      reasons: list, *analysis_dicts) -> Dict:
        """Create standardized result dictionary"""
        result = {
            'symbol': symbol,
            'tradeable': tradeable,
            'quality_score': score,
            'max_score': 10,
            'failed_reasons': reasons,
            'filter_name': 'SetupScanner'
        }
        
        # Add all analysis dictionaries
        analysis_keys = ['trend', 'pullback', 'level', 'bounce', 'volume']
        for i, analysis in enumerate(analysis_dicts):
            if i < len(analysis_keys) and analysis:
                result[analysis_keys[i]] = analysis
        
        return result


def scan_watchlist_for_setups(watchlist: list, data_provider: DataProvider, config: Dict) -> Dict:
    """
    Scan entire watchlist for pullback setups
    
    Args:
        watchlist: List of symbols to scan
        data_provider: Data source  
        config: Scanner configuration
        
    Returns:
        Dict with setup results
    """
    scanner = SetupScanner(config)
    
    setups_found = []
    no_setups = []
    scan_results = {}
    
    logger.info(f"🎯 SETUP SCANNER: Checking {len(watchlist)} symbols for pullback setups")
    logger.info("=" * 60)
    
    for symbol in watchlist:
        try:
            result = scanner.scan_for_setup(symbol, data_provider)
            scan_results[symbol] = result
            
            if result['tradeable']:
                setups_found.append(symbol)
                trend_dir = result.get('trend', {}).get('direction', 'Unknown')
                quality = result['quality_score']
                logger.info(f"🎯 {symbol}: {trend_dir} pullback setup (quality: {quality}/10)")
            else:
                no_setups.append(symbol)
                logger.debug(f"❌ {symbol}: {', '.join(result['failed_reasons'])}")
                
        except Exception as e:
            logger.error(f"🚨 {symbol}: Setup scan failed - {str(e)}")
            no_setups.append(symbol)
            scan_results[symbol] = scanner._create_result(
                symbol, False, 0, [f"Error: {str(e)}"]
            )
    
    logger.info("=" * 60)
    logger.info(f"📊 SETUP SCANNER RESULTS:")
    logger.info(f"   Setups found: {len(setups_found)}/{len(watchlist)} symbols")
    
    if len(setups_found) > 0:
        # Sort by quality score
        sorted_setups = sorted(setups_found, 
                             key=lambda s: scan_results[s]['quality_score'], 
                             reverse=True)
        logger.info(f"   Best setup: {sorted_setups[0]} (quality: {scan_results[sorted_setups[0]]['quality_score']}/10)")
    else:
        logger.warning("⚠️ NO SETUPS found - no pullback patterns ready")
    
    return {
        'setup_symbols': setups_found,
        'no_setup_symbols': no_setups,
        'results': scan_results,
        'summary': {
            'total_scanned': len(watchlist),
            'setups_count': len(setups_found),
            'best_setup': setups_found[0] if setups_found else None
        }
    }