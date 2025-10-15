"""
Filter 3: Confluence Checker
🎯 Validates multi-timeframe alignment before entry

Checks alignment across timeframes:
- 1H: Entry timing and momentum
- 4H: Setup confirmation 
- 1D: Overall trend context
- Optional 1W: Major trend direction

Must have confluence on at least 2/3 timeframes
"""

import logging
from typing import Dict, List
import pandas as pd
from core.data_provider import DataProvider, TechnicalIndicators

logger = logging.getLogger(__name__)

class ConfluenceChecker:
    """
    Validates multi-timeframe alignment for high-confidence entries
    Prevents trading against higher timeframe momentum
    """
    
    def __init__(self, config: Dict):
        """
        Initialize confluence checker with timeframe parameters
        
        Args:
            config: Checker configuration
        """
        # Timeframes to check (in order of importance)
        self.timeframes = config.get('timeframes', ['1h', '4h', '1d'])
        self.min_confluence_count = config.get('min_confluence_count', 2)
        
        # Trend alignment thresholds
        self.sma_alignment_buffer = config.get('sma_alignment_buffer', 0.01)  # 1%
        self.rsi_neutral_zone = config.get('rsi_neutral_zone', [40, 60])      # RSI 40-60 neutral
        self.momentum_threshold = config.get('momentum_threshold', 0.02)      # 2% momentum
        
        # Confluence scoring weights
        self.timeframe_weights = config.get('timeframe_weights', {
            '1h': 1.0,   # Entry timing
            '4h': 2.0,   # Setup confirmation (most important)  
            '1d': 1.5,   # Trend context
            '1w': 1.0    # Major trend (if used)
        })
        
        logger.info("🔄 Confluence Checker initialized")
        logger.info(f"   Timeframes: {', '.join(self.timeframes)}")
        logger.info(f"   Required confluence: {self.min_confluence_count}/{len(self.timeframes)} timeframes")
        
    def check_confluence(self, symbol: str, trade_direction: str, data_provider: DataProvider) -> Dict:
        """
        Check multi-timeframe confluence for trade direction
        
        Args:
            symbol: Trading pair to check
            trade_direction: 'LONG' or 'SHORT' 
            data_provider: Data source
            
        Returns:
            Dict with confluence analysis and approval
        """
        try:
            # Analyze each timeframe
            timeframe_results = {}
            confluence_scores = {}
            
            logger.info(f"🔄 Checking {trade_direction} confluence for {symbol}")
            
            for tf in self.timeframes:
                try:
                    tf_result = self._analyze_timeframe(symbol, tf, trade_direction, data_provider)
                    timeframe_results[tf] = tf_result
                    confluence_scores[tf] = tf_result['confluence_score']
                    
                    status = "✅" if tf_result['supports_direction'] else "❌"
                    logger.info(f"   {status} {tf.upper()}: {tf_result['trend_direction']} "
                               f"(score: {tf_result['confluence_score']:.1f})")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to analyze {tf} for {symbol}: {str(e)}")
                    timeframe_results[tf] = {
                        'supports_direction': False,
                        'confluence_score': 0,
                        'trend_direction': 'ERROR',
                        'error': str(e)
                    }
                    confluence_scores[tf] = 0
            
            # Calculate weighted confluence score
            weighted_score = self._calculate_weighted_score(confluence_scores)
            
            # Count supporting timeframes
            supporting_tfs = [tf for tf, result in timeframe_results.items() 
                            if result.get('supports_direction', False)]
            
            # Determine if confluence is sufficient
            has_confluence = len(supporting_tfs) >= self.min_confluence_count
            
            # Check for major conflicts (higher TF against trade)
            conflicts = self._check_major_conflicts(timeframe_results, trade_direction)
            
            # Final approval (need confluence AND no major conflicts)
            approved = has_confluence and len(conflicts) == 0
            
            result = {
                'symbol': symbol,
                'trade_direction': trade_direction,
                'approved': approved,
                'has_confluence': has_confluence,
                'supporting_timeframes': supporting_tfs,
                'confluence_count': len(supporting_tfs),
                'required_count': self.min_confluence_count,
                'weighted_score': weighted_score,
                'max_weighted_score': 10.0,
                'timeframe_results': timeframe_results,
                'conflicts': conflicts,
                'filter_name': 'ConfluenceChecker'
            }
            
            # Log final result
            if approved:
                logger.info(f"✅ CONFLUENCE APPROVED {symbol} {trade_direction}: "
                           f"{len(supporting_tfs)}/{len(self.timeframes)} TFs support "
                           f"(score: {weighted_score:.1f}/10)")
            else:
                reasons = []
                if not has_confluence:
                    reasons.append(f"Only {len(supporting_tfs)}/{self.min_confluence_count} TFs support")
                if conflicts:
                    reasons.append(f"Conflicts: {', '.join(conflicts)}")
                    
                logger.info(f"❌ CONFLUENCE REJECTED {symbol} {trade_direction}: {', '.join(reasons)}")
            
            return result
            
        except Exception as e:
            logger.error(f"🚨 Confluence check failed for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'trade_direction': trade_direction,
                'approved': False,
                'error': str(e),
                'filter_name': 'ConfluenceChecker'
            }
    
    def _analyze_timeframe(self, symbol: str, timeframe: str, trade_direction: str, 
                          data_provider: DataProvider) -> Dict:
        """
        Analyze single timeframe for confluence
        
        Returns:
            Dict with timeframe analysis
        """
        # Get appropriate number of periods for timeframe
        periods_map = {'1h': 7, '4h': 14, '1d': 30, '1w': 52}
        days = periods_map.get(timeframe, 14)
        
        df = data_provider.get_ohlcv(symbol, timeframe, days=days)
        
        if len(df) < 20:
            raise ValueError(f"Insufficient data for {timeframe} analysis")
        
        # Add indicators
        df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['close'], 50) 
        df['rsi'] = TechnicalIndicators.rsi(df['close'], 14)
        df['macd'], df['macd_signal'], _ = TechnicalIndicators.macd(df['close'])
        
        # Current values
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        
        # Analyze trend direction
        trend_signals = self._get_trend_signals(
            current_price, sma_20, sma_50, rsi, macd, macd_signal
        )
        
        # Determine overall trend direction
        bullish_signals = sum([
            trend_signals['price_above_sma20'],
            trend_signals['sma20_above_sma50'],
            trend_signals['rsi_bullish'],
            trend_signals['macd_bullish']
        ])
        
        bearish_signals = sum([
            trend_signals['price_below_sma20'],
            trend_signals['sma20_below_sma50'], 
            trend_signals['rsi_bearish'],
            trend_signals['macd_bearish']
        ])
        
        if bullish_signals >= 3:
            tf_direction = "BULLISH"
            momentum_score = bullish_signals
        elif bearish_signals >= 3:
            tf_direction = "BEARISH"
            momentum_score = bearish_signals
        else:
            tf_direction = "NEUTRAL"
            momentum_score = 0
        
        # Check if timeframe supports trade direction
        supports_direction = self._check_direction_support(tf_direction, trade_direction)
        
        # Calculate base confluence score (0-10)
        base_confluence_score = self._calculate_timeframe_score(
            momentum_score, tf_direction, trade_direction, trend_signals
        )
        
        # Perform enhanced confluence analysis (Fibonacci, volume profile, bounces)
        enhanced_analysis = self._enhanced_confluence_analysis(df, trade_direction)
        
        # Combine base score with enhanced confluence factors
        # Enhanced factors can add up to 50 points, scaled to 0-5 range for final score
        enhanced_score = enhanced_analysis['total_confluence_score'] / 10.0  # Scale 50 -> 5
        final_confluence_score = min(10.0, base_confluence_score + enhanced_score)
        
        result = {
            'timeframe': timeframe,
            'trend_direction': tf_direction,
            'supports_direction': supports_direction,
            'confluence_score': final_confluence_score,
            'base_confluence_score': base_confluence_score,
            'enhanced_confluence_score': enhanced_score,
            'momentum_score': momentum_score,
            'trend_signals': trend_signals,
            'current_rsi': rsi,
            'sma_alignment': sma_20 > sma_50 if pd.notna(sma_20) and pd.notna(sma_50) else None,
            'enhanced_analysis': enhanced_analysis
        }
        
        # Log enhanced confluence factors if any found
        if enhanced_analysis['confluence_factors']:
            logger.info(f"   Enhanced confluence factors for {timeframe}: {', '.join(enhanced_analysis['confluence_factors'])}")
        
        return result
    
    def _get_trend_signals(self, price: float, sma_20: float, sma_50: float, 
                          rsi: float, macd: float, macd_signal: float) -> Dict:
        """
        Get individual trend signals for confluence analysis
        
        Returns:
            Dict of boolean trend signals
        """
        return {
            # Price vs SMAs
            'price_above_sma20': price > sma_20 if pd.notna(sma_20) else False,
            'price_below_sma20': price < sma_20 if pd.notna(sma_20) else False,
            
            # SMA alignment
            'sma20_above_sma50': sma_20 > sma_50 if pd.notna(sma_20) and pd.notna(sma_50) else False,
            'sma20_below_sma50': sma_20 < sma_50 if pd.notna(sma_20) and pd.notna(sma_50) else False,
            
            # RSI momentum
            'rsi_bullish': rsi > self.rsi_neutral_zone[1] if pd.notna(rsi) else False,
            'rsi_bearish': rsi < self.rsi_neutral_zone[0] if pd.notna(rsi) else False,
            'rsi_neutral': self.rsi_neutral_zone[0] <= rsi <= self.rsi_neutral_zone[1] if pd.notna(rsi) else True,
            
            # MACD momentum
            'macd_bullish': macd > macd_signal if pd.notna(macd) and pd.notna(macd_signal) else False,
            'macd_bearish': macd < macd_signal if pd.notna(macd) and pd.notna(macd_signal) else False,
        }
    
    def _check_direction_support(self, tf_direction: str, trade_direction: str) -> bool:
        """
        Check if timeframe direction supports trade direction
        
        Returns:
            True if timeframe supports the trade
        """
        if trade_direction == "LONG":
            return tf_direction in ["BULLISH", "NEUTRAL"]
        elif trade_direction == "SHORT":
            return tf_direction in ["BEARISH", "NEUTRAL"]
        else:
            return False
    
    def _calculate_timeframe_score(self, momentum_score: int, tf_direction: str, 
                                  trade_direction: str, signals: Dict) -> float:
        """
        Calculate confluence score for single timeframe (0-10)
        
        Returns:
            Float score from 0-10
        """
        base_score = 0
        
        # Strong directional alignment (0-6 points)
        if tf_direction == trade_direction.replace("LONG", "BULLISH").replace("SHORT", "BEARISH"):
            base_score += momentum_score * 1.5  # 1.5 points per signal
        elif tf_direction == "NEUTRAL":
            base_score += 3  # Neutral doesn't hurt but doesn't help much
        else:
            base_score = 0  # Opposing direction = no points
        
        # Bonus points for strong momentum indicators (0-2 points)
        if signals.get('rsi_bullish') and trade_direction == "LONG":
            base_score += 1
        elif signals.get('rsi_bearish') and trade_direction == "SHORT":
            base_score += 1
            
        if signals.get('macd_bullish') and trade_direction == "LONG":
            base_score += 1
        elif signals.get('macd_bearish') and trade_direction == "SHORT":
            base_score += 1
        
        # Penalty for opposing momentum (0-2 points deduction)
        if signals.get('rsi_bearish') and trade_direction == "LONG":
            base_score -= 1
        elif signals.get('rsi_bullish') and trade_direction == "SHORT":
            base_score -= 1
        
        return max(0, min(10, base_score))  # Clamp between 0-10
    
    def _calculate_weighted_score(self, confluence_scores: Dict) -> float:
        """
        Calculate weighted confluence score across all timeframes
        
        Returns:
            Weighted score from 0-10
        """
        total_weighted_score = 0
        total_weight = 0
        
        for tf, score in confluence_scores.items():
            weight = self.timeframe_weights.get(tf, 1.0)
            total_weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return total_weighted_score / total_weight
    
    def _check_major_conflicts(self, timeframe_results: Dict, trade_direction: str) -> List[str]:
        """
        Check for major conflicts that should block the trade
        
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        
        # Check daily timeframe opposition (major conflict)
        if '1d' in timeframe_results:
            daily_result = timeframe_results['1d']
            daily_trend = daily_result.get('trend_direction', 'NEUTRAL')
            
            if trade_direction == "LONG" and daily_trend == "BEARISH":
                if daily_result.get('confluence_score', 0) >= 6:  # Strong bearish daily
                    conflicts.append("Strong daily bearish trend opposes LONG")
                    
            elif trade_direction == "SHORT" and daily_trend == "BULLISH":
                if daily_result.get('confluence_score', 0) >= 6:  # Strong bullish daily
                    conflicts.append("Strong daily bullish trend opposes SHORT")
        
        # Check weekly timeframe opposition (major conflict)
        if '1w' in timeframe_results:
            weekly_result = timeframe_results['1w']
            weekly_trend = weekly_result.get('trend_direction', 'NEUTRAL')
            
            if trade_direction == "LONG" and weekly_trend == "BEARISH":
                if weekly_result.get('confluence_score', 0) >= 7:  # Very strong bearish weekly
                    conflicts.append("Strong weekly bearish trend opposes LONG")
                    
            elif trade_direction == "SHORT" and weekly_trend == "BULLISH":
                if weekly_result.get('confluence_score', 0) >= 7:  # Very strong bullish weekly
                    conflicts.append("Strong weekly bullish trend opposes SHORT")
        
        return conflicts

    def _calculate_fibonacci_levels(self, df: pd.DataFrame, lookback_bars: int = 50) -> Dict:
        """
        Calculate Fibonacci retracement levels from recent swing high/low
        
        Args:
            df: OHLCV dataframe
            lookback_bars: Number of bars to look back for swing points
            
        Returns:
            Dict with fib levels and current price proximity
        """
        if len(df) < lookback_bars:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        # Find swing high and low in lookback period
        recent_data = df.iloc[-lookback_bars:]
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        # Calculate Fibonacci levels
        price_range = swing_high - swing_low
        
        # If no price range (flat data), no meaningful Fibonacci levels
        if price_range == 0:
            return {'valid': False, 'reason': 'No price movement for Fibonacci calculation'}
        fib_levels = {
            '0%': swing_low,
            '23.6%': swing_low + (price_range * 0.236),
            '38.2%': swing_low + (price_range * 0.382),
            '50%': swing_low + (price_range * 0.5),
            '61.8%': swing_low + (price_range * 0.618),
            '78.6%': swing_low + (price_range * 0.786),
            '100%': swing_high
        }
        
        current_price = df['close'].iloc[-1]
        
        # Find closest Fibonacci level
        closest_level = None
        min_distance = float('inf')
        
        for level_name, level_price in fib_levels.items():
            distance = abs(current_price - level_price) / current_price
            if distance < min_distance:
                min_distance = distance
                closest_level = level_name
        
        # Check if price is near any significant fib level (within 1.5%)
        tolerance = 0.015  # 1.5%
        near_fib = min_distance < tolerance
        
        return {
            'valid': True,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'levels': fib_levels,
            'current_price': current_price,
            'closest_level': closest_level,
            'distance_pct': min_distance * 100,
            'near_fibonacci': near_fib,
            'confluence_score': 15 if near_fib else 0  # 15 points for fib confluence
        }

    def _calculate_volume_profile(self, df: pd.DataFrame, price_tolerance: float = 0.02) -> Dict:
        """
        Analyze volume distribution at current price level
        
        Args:
            df: OHLCV dataframe with volume
            price_tolerance: Price range tolerance (2% = 0.02)
            
        Returns:
            Dict with volume profile analysis
        """
        if len(df) < 30:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        current_price = df['close'].iloc[-1]
        
        # Define price zone around current level
        price_min = current_price * (1 - price_tolerance)
        price_max = current_price * (1 + price_tolerance)
        
        # Find bars where price traded in current zone
        in_zone = df[
            (df['low'] <= price_max) & (df['high'] >= price_min)
        ]
        
        if len(in_zone) == 0:
            return {
                'valid': True,
                'volume_in_zone': 0,
                'avg_volume': df['volume'].mean(),
                'zone_strength': 0,
                'confluence_score': 0
            }
        
        # Calculate volume metrics
        volume_in_zone = in_zone['volume'].sum()
        avg_volume_per_bar = df['volume'].mean()
        expected_zone_volume = avg_volume_per_bar * len(in_zone)
        
        # Zone strength = actual volume vs expected volume
        zone_strength = volume_in_zone / expected_zone_volume if expected_zone_volume > 0 else 0
        
        # Score based on volume strength (high volume = strong support/resistance)
        confluence_score = 0
        if zone_strength > 1.5:  # 50% above average
            confluence_score = 15
        elif zone_strength > 1.2:  # 20% above average
            confluence_score = 10
        elif zone_strength > 1.0:  # Above average
            confluence_score = 5
        
        return {
            'valid': True,
            'current_price': current_price,
            'price_zone': (price_min, price_max),
            'volume_in_zone': volume_in_zone,
            'avg_volume': avg_volume_per_bar,
            'zone_strength': zone_strength,
            'bars_in_zone': len(in_zone),
            'confluence_score': confluence_score
        }

    def _detect_previous_bounces(self, df: pd.DataFrame, price_tolerance: float = 0.015) -> Dict:
        """
        Detect previous bounces/rejections at current price level
        
        Args:
            df: OHLCV dataframe
            price_tolerance: Price tolerance for bounce detection (1.5% = 0.015)
            
        Returns:
            Dict with bounce analysis
        """
        if len(df) < 50:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        current_price = df['close'].iloc[-1]
        
        # Define price zone
        price_min = current_price * (1 - price_tolerance) 
        price_max = current_price * (1 + price_tolerance)
        
        bounces = 0
        bounce_details = []
        
        # Look for bounces in historical data (skip recent 5 bars)
        historical_data = df.iloc[:-5]  
        
        for i in range(2, len(historical_data) - 2):
            bar = historical_data.iloc[i]
            prev_bar = historical_data.iloc[i-1]
            next_bar = historical_data.iloc[i+1]
            
            # Check if this bar touched our price zone
            if bar['low'] <= price_max and bar['high'] >= price_min:
                
                # Check for bounce pattern (low followed by higher close)
                if (bar['low'] < prev_bar['low'] and  # Made a lower low
                    bar['close'] > bar['low'] * 1.005 and  # Closed above low (0.5% bounce)
                    next_bar['close'] > bar['close']):  # Next bar continued higher
                    
                    bounces += 1
                    bounce_details.append({
                        'date': bar.name if hasattr(bar, 'name') else i,
                        'price': bar['low'],
                        'bounce_strength': (bar['close'] - bar['low']) / bar['low']
                    })
                
                # Check for rejection pattern (high followed by lower close)  
                elif (bar['high'] > prev_bar['high'] and  # Made a higher high
                      bar['close'] < bar['high'] * 0.995 and  # Closed below high (0.5% rejection)
                      next_bar['close'] < bar['close']):  # Next bar continued lower
                    
                    bounces += 1
                    bounce_details.append({
                        'date': bar.name if hasattr(bar, 'name') else i,
                        'price': bar['high'],
                        'bounce_strength': (bar['high'] - bar['close']) / bar['high']
                    })
        
        # Score based on number of previous reactions
        confluence_score = 0
        if bounces >= 3:
            confluence_score = 20  # Very strong level
        elif bounces >= 2:
            confluence_score = 15  # Strong level
        elif bounces >= 1:
            confluence_score = 10  # Moderate level
        
        return {
            'valid': True,
            'current_price': current_price,
            'price_zone': (price_min, price_max),
            'bounce_count': bounces,
            'bounce_details': bounce_details,
            'confluence_score': confluence_score,
            'level_strength': 'Very Strong' if bounces >= 3 else 
                            'Strong' if bounces >= 2 else
                            'Moderate' if bounces >= 1 else 'Weak'
        }

    def _enhanced_confluence_analysis(self, df: pd.DataFrame, trade_direction: str) -> Dict:
        """
        Perform enhanced confluence analysis including Fibonacci, volume profile, and bounce detection
        
        Args:
            df: OHLCV dataframe
            trade_direction: 'LONG' or 'SHORT'
            
        Returns:
            Dict with all confluence factors and combined score
        """
        # Calculate all confluence factors
        fib_analysis = self._calculate_fibonacci_levels(df)
        volume_analysis = self._calculate_volume_profile(df)
        bounce_analysis = self._detect_previous_bounces(df)
        
        # Combine confluence scores
        total_confluence_score = 0
        confluence_factors = []
        
        if fib_analysis.get('valid', False) and fib_analysis.get('near_fibonacci', False):
            total_confluence_score += fib_analysis['confluence_score']
            confluence_factors.append(f"Fibonacci {fib_analysis['closest_level']} level")
        
        if volume_analysis.get('valid', False) and volume_analysis['confluence_score'] > 0:
            total_confluence_score += volume_analysis['confluence_score']
            strength = volume_analysis['zone_strength']
            confluence_factors.append(f"Volume support ({strength:.1f}x average)")
        
        if bounce_analysis.get('valid', False) and bounce_analysis['bounce_count'] > 0:
            total_confluence_score += bounce_analysis['confluence_score']
            count = bounce_analysis['bounce_count']
            confluence_factors.append(f"{count} previous bounce(s) at level")
        
        return {
            'fibonacci': fib_analysis,
            'volume_profile': volume_analysis,
            'previous_bounces': bounce_analysis,
            'total_confluence_score': total_confluence_score,
            'max_possible_score': 50,  # 15 + 15 + 20
            'confluence_factors': confluence_factors,
            'has_enhanced_confluence': total_confluence_score >= 10  # Minimum threshold (lowered)
        }


def check_setup_confluence(setup_symbols: List[str], data_provider: DataProvider, 
                          setup_results: Dict, config: Dict) -> Dict:
    """
    Check confluence for all symbols with valid setups
    
    Args:
        setup_symbols: List of symbols with valid setups
        data_provider: Data source
        setup_results: Results from setup scanner
        config: Confluence configuration
        
    Returns:
        Dict with confluence results
    """
    confluence_checker = ConfluenceChecker(config)
    
    approved_symbols = []
    rejected_symbols = []
    confluence_results = {}
    
    logger.info(f"🔄 CONFLUENCE CHECKER: Validating {len(setup_symbols)} setups")
    logger.info("=" * 60)
    
    for symbol in setup_symbols:
        try:
            # Determine trade direction from setup
            setup_result = setup_results.get(symbol, {})
            trend_direction = setup_result.get('trend', {}).get('direction', 'UNKNOWN')
            
            if trend_direction == 'UP':
                trade_direction = 'LONG'
            elif trend_direction == 'DOWN':
                trade_direction = 'SHORT'  
            else:
                logger.warning(f"⚠️ {symbol}: Cannot determine trade direction from setup")
                rejected_symbols.append(symbol)
                confluence_results[symbol] = {
                    'symbol': symbol,
                    'approved': False,
                    'error': 'Unknown trade direction from setup',
                    'filter_name': 'ConfluenceChecker'
                }
                continue
            
            # Check confluence
            result = confluence_checker.check_confluence(symbol, trade_direction, data_provider)
            confluence_results[symbol] = result
            
            if result['approved']:
                approved_symbols.append(symbol)
                score = result.get('weighted_score', 0)
                supporting = result.get('confluence_count', 0)
                total_tfs = len(confluence_checker.timeframes)
                logger.info(f"✅ {symbol}: {trade_direction} confluence approved "
                           f"({supporting}/{total_tfs} TFs, score: {score:.1f}/10)")
            else:
                rejected_symbols.append(symbol)
                conflicts = result.get('conflicts', [])
                if conflicts:
                    logger.info(f"❌ {symbol}: Confluence rejected - {', '.join(conflicts)}")
                else:
                    supporting = result.get('confluence_count', 0)
                    required = result.get('required_count', confluence_checker.min_confluence_count)
                    logger.info(f"❌ {symbol}: Insufficient confluence ({supporting}/{required})")
                    
        except Exception as e:
            logger.error(f"🚨 {symbol}: Confluence check failed - {str(e)}")
            rejected_symbols.append(symbol)
            confluence_results[symbol] = {
                'symbol': symbol,
                'approved': False,
                'error': str(e),
                'filter_name': 'ConfluenceChecker'
            }
    
    logger.info("=" * 60)
    logger.info(f"📊 CONFLUENCE RESULTS:")
    logger.info(f"   Approved: {len(approved_symbols)}/{len(setup_symbols)} symbols")
    
    if len(approved_symbols) > 0:
        # Sort by weighted confluence score
        sorted_approved = sorted(approved_symbols,
                               key=lambda s: confluence_results[s].get('weighted_score', 0),
                               reverse=True)
        best_symbol = sorted_approved[0]
        best_score = confluence_results[best_symbol].get('weighted_score', 0)
        logger.info(f"   Best confluence: {best_symbol} (score: {best_score:.1f}/10)")
    else:
        logger.warning("⚠️ NO SYMBOLS passed confluence check - multi-timeframe alignment poor")
    
    return {
        'approved_symbols': approved_symbols,
        'rejected_symbols': rejected_symbols,
        'results': confluence_results,
        'summary': {
            'total_checked': len(setup_symbols),
            'approved_count': len(approved_symbols),
            'best_symbol': approved_symbols[0] if approved_symbols else None,
            'approval_rate': len(approved_symbols) / len(setup_symbols) if setup_symbols else 0
        }
    }