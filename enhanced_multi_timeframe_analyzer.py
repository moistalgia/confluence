#!/usr/bin/env python3
"""
Enhanced Multi-Timeframe Analyzer - Tier 1 AI Feedback Implementation
Expected improvement: +25% confidence boost according to AI analysis
"""

import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMultiTimeframeAnalyzer:
    """Enhanced multi-timeframe analysis system with AI feedback integration"""
    
    def __init__(self, exchange_id='kraken'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000
        })
        
        # Timeframe configuration with AI-optimized weights
        self.timeframes = {
            '1w': {'weight': 0.35, 'periods': 100},  # Weekly - highest weight for trend
            '1d': {'weight': 0.40, 'periods': 200},  # Daily - primary analysis
            '4h': {'weight': 0.20, 'periods': 300},  # 4-hour - momentum
            '1h': {'weight': 0.05, 'periods': 168}   # Hourly - entry timing
        }
        
        self.output_dir = Path("output/ultimate_analysis/raw_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_multi_timeframe(self, symbol: str) -> Dict:
        """
        Enhanced multi-timeframe analysis
        Returns comprehensive analysis across all timeframes
        """
        logger.info(f"Starting enhanced multi-timeframe analysis for {symbol}")
        
        try:
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'timeframe_data': {},
                'confluence_analysis': {},
                'enhanced_signals': {},
                'data_quality': {}
            }
            
            # Collect data for all timeframes
            timeframe_data = {}
            for tf, config in self.timeframes.items():
                logger.info(f"Fetching {tf} data for {symbol}")
                
                try:
                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, 
                        tf, 
                        limit=config['periods']
                    )
                    
                    if not ohlcv:
                        logger.warning(f"No data received for {symbol} {tf}")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    
                    # Calculate technical indicators for this timeframe
                    indicators = self._calculate_enhanced_indicators(df, tf)
                    
                    # Store timeframe data
                    timeframe_data[tf] = {
                        'ohlcv': df.to_dict('records'),
                        'indicators': indicators,
                        'data_points': len(df),
                        'date_range': {
                            'start': df.index[0].isoformat(),
                            'end': df.index[-1].isoformat()
                        }
                    }
                    
                    logger.info(f"Successfully analyzed {tf} data: {len(df)} candles")
                    
                except Exception as e:
                    logger.error(f"Error fetching {tf} data for {symbol}: {e}")
                    timeframe_data[tf] = {'error': str(e)}
                    continue
            
            analysis_result['timeframe_data'] = timeframe_data
            
            # Perform confluence analysis
            confluence = self._analyze_timeframe_confluence(timeframe_data)
            analysis_result['confluence_analysis'] = confluence
            
            # Generate enhanced trading signals
            signals = self._generate_enhanced_signals(timeframe_data, confluence)
            analysis_result['enhanced_signals'] = signals
            
            # Assess data quality
            quality = self._assess_data_quality(timeframe_data)
            analysis_result['data_quality'] = quality
            
            # Save raw data
            self._save_timeframe_data(symbol, timeframe_data)
            
            logger.info(f"Enhanced multi-timeframe analysis complete for {symbol}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Enhanced multi-timeframe analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
    
    def _calculate_enhanced_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Calculate enhanced technical indicators for a specific timeframe"""
        
        if df.empty or len(df) < 20:
            return {'error': 'insufficient_data', 'periods': len(df)}
        
        indicators = {}
        
        try:
            # Trend Indicators
            indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20).iloc[-1]
            indicators['sma_50'] = ta.trend.sma_indicator(df['close'], window=50).iloc[-1] if len(df) >= 50 else None
            indicators['sma_200'] = ta.trend.sma_indicator(df['close'], window=200).iloc[-1] if len(df) >= 200 else None
            
            indicators['ema_12'] = ta.trend.ema_indicator(df['close'], window=12).iloc[-1]
            indicators['ema_26'] = ta.trend.ema_indicator(df['close'], window=26).iloc[-1]
            indicators['ema_50'] = ta.trend.ema_indicator(df['close'], window=50).iloc[-1] if len(df) >= 50 else None
            
            # MACD
            macd_line = ta.trend.macd(df['close'])
            macd_signal = ta.trend.macd_signal(df['close'])
            macd_histogram = ta.trend.macd_diff(df['close'])
            
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = macd_signal.iloc[-1]
            indicators['macd_histogram'] = macd_histogram.iloc[-1]
            
            # Momentum Indicators
            indicators['rsi'] = ta.momentum.rsi(df['close'], window=14).iloc[-1]
            indicators['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close']).iloc[-1]
            indicators['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close']).iloc[-1]
            
            # Enhanced Bollinger Band Analysis with Squeeze Detection
            bb_analysis = self._calculate_bollinger_analysis(df)
            indicators.update(bb_analysis)
            
            # ATR
            indicators['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
            indicators['atr_percent'] = (indicators['atr'] / df['close'].iloc[-1]) * 100
            
            # Volume Indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma']
            
            if len(df) >= 20:
                indicators['obv'] = ta.volume.on_balance_volume(df['close'], df['volume']).iloc[-1]
            
            # VWAP Calculations (Volume Weighted Average Price)
            # Standard VWAP calculation
            df_copy = df.copy()
            df_copy['typical_price'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
            df_copy['volume_price'] = df_copy['typical_price'] * df_copy['volume']
            
            # Calculate cumulative VWAP for the entire period
            cumulative_volume = df_copy['volume'].cumsum()
            cumulative_volume_price = df_copy['volume_price'].cumsum()
            vwap_series = cumulative_volume_price / cumulative_volume
            
            current_vwap = vwap_series.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            indicators['vwap'] = current_vwap
            indicators['vwap_distance'] = current_price - current_vwap
            indicators['vwap_distance_percent'] = ((current_price - current_vwap) / current_vwap) * 100
            
            # VWAP trend analysis
            if current_price > current_vwap * 1.02:  # 2% above VWAP
                indicators['vwap_position'] = 'STRONG_ABOVE'
                indicators['vwap_signal'] = 'BULLISH'
            elif current_price > current_vwap:
                indicators['vwap_position'] = 'ABOVE'
                indicators['vwap_signal'] = 'MILDLY_BULLISH'
            elif current_price < current_vwap * 0.98:  # 2% below VWAP
                indicators['vwap_position'] = 'STRONG_BELOW'
                indicators['vwap_signal'] = 'BEARISH'
            else:
                indicators['vwap_position'] = 'BELOW'
                indicators['vwap_signal'] = 'MILDLY_BEARISH'
            
            # Moving VWAP (anchored VWAP for different periods)
            if len(df) >= 20:
                # 20-period anchored VWAP
                recent_df = df_copy.tail(20).copy()
                recent_df['cum_volume'] = recent_df['volume'].cumsum()
                recent_df['cum_volume_price'] = recent_df['volume_price'].cumsum()
                moving_vwap_20 = (recent_df['cum_volume_price'] / recent_df['cum_volume']).iloc[-1]
                indicators['vwap_20'] = moving_vwap_20
                indicators['vwap_20_distance_percent'] = ((current_price - moving_vwap_20) / moving_vwap_20) * 100
            
            if len(df) >= 50:
                # 50-period anchored VWAP
                recent_df = df_copy.tail(50).copy()
                recent_df['cum_volume'] = recent_df['volume'].cumsum()
                recent_df['cum_volume_price'] = recent_df['volume_price'].cumsum()
                moving_vwap_50 = (recent_df['cum_volume_price'] / recent_df['cum_volume']).iloc[-1]
                indicators['vwap_50'] = moving_vwap_50
                indicators['vwap_50_distance_percent'] = ((current_price - moving_vwap_50) / moving_vwap_50) * 100
            
            # VWAP bands (standard deviation bands around VWAP)
            if len(df) >= 20:
                # Calculate VWAP standard deviation
                price_variance = ((df_copy['typical_price'] - current_vwap) ** 2 * df_copy['volume']).sum() / df_copy['volume'].sum()
                vwap_std = price_variance ** 0.5
                
                indicators['vwap_upper_1'] = current_vwap + vwap_std
                indicators['vwap_lower_1'] = current_vwap - vwap_std
                indicators['vwap_upper_2'] = current_vwap + (2 * vwap_std)
                indicators['vwap_lower_2'] = current_vwap - (2 * vwap_std)
                
                # Determine position within VWAP bands
                if current_price > indicators['vwap_upper_2']:
                    indicators['vwap_band_position'] = 'ABOVE_2STD'
                elif current_price > indicators['vwap_upper_1']:
                    indicators['vwap_band_position'] = 'ABOVE_1STD'
                elif current_price < indicators['vwap_lower_2']:
                    indicators['vwap_band_position'] = 'BELOW_2STD'
                elif current_price < indicators['vwap_lower_1']:
                    indicators['vwap_band_position'] = 'BELOW_1STD'
                else:
                    indicators['vwap_band_position'] = 'WITHIN_BANDS'
            
            # Price Action
            current_price = df['close'].iloc[-1]
            indicators['price'] = current_price
            
            # Support/Resistance levels
            if len(df) >= 20:
                recent_high = df['high'].tail(20).max()
                recent_low = df['low'].tail(20).min()
                indicators['resistance_level'] = recent_high
                indicators['support_level'] = recent_low
                indicators['position_in_range'] = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            # Trend strength
            if indicators['sma_50'] and indicators['sma_200']:
                if current_price > indicators['sma_200']:
                    indicators['trend'] = 'BULLISH'
                    indicators['trend_strength'] = min(((current_price - indicators['sma_200']) / indicators['sma_200']) * 100, 100)
                else:
                    indicators['trend'] = 'BEARISH'
                    indicators['trend_strength'] = min(((indicators['sma_200'] - current_price) / indicators['sma_200']) * 100, 100)
            else:
                indicators['trend'] = 'NEUTRAL'
                indicators['trend_strength'] = 0
            
            # Momentum analysis
            if indicators['rsi'] > 70:
                indicators['momentum'] = 'OVERBOUGHT'
            elif indicators['rsi'] < 30:
                indicators['momentum'] = 'OVERSOLD'
            elif indicators['rsi'] > 60:
                indicators['momentum'] = 'STRONG_BULLISH'
            elif indicators['rsi'] < 40:
                indicators['momentum'] = 'WEAK_BEARISH'
            else:
                indicators['momentum'] = 'NEUTRAL'
            
            # Volatility assessment with regime calculation
            if indicators['atr_percent'] > 5:
                indicators['volatility'] = 'HIGH'
            elif indicators['atr_percent'] > 2:
                indicators['volatility'] = 'MEDIUM'
            else:
                indicators['volatility'] = 'LOW'
            
            # Volatility regime based on historical percentile
            if len(df) >= 50:
                atr_series = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
                current_atr = atr_series.iloc[-1]
                historical_atrs = atr_series.tail(50)
                atr_percentile = (historical_atrs < current_atr).sum() / len(historical_atrs) * 100
                
                if atr_percentile > 75:
                    indicators['volatility_regime'] = 'HIGH'
                elif atr_percentile > 25:
                    indicators['volatility_regime'] = 'NORMAL'  
                else:
                    indicators['volatility_regime'] = 'LOW'
            else:
                indicators['volatility_regime'] = 'UNKNOWN'
            
            logger.info(f"Calculated {len(indicators)} indicators for {timeframe}")
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {timeframe}: {e}")
            indicators['error'] = str(e)
        
        return indicators
    
    def _analyze_timeframe_confluence(self, timeframe_data: Dict) -> Dict:
        """Analyze confluence across multiple timeframes"""
        
        confluence = {
            'trend_alignment': {},
            'momentum_confluence': {},
            'support_resistance': {},
            'volume_confirmation': {},
            'overall_score': 0
        }
        
        try:
            valid_timeframes = [tf for tf, data in timeframe_data.items() 
                              if 'indicators' in data and 'error' not in data['indicators']]
            
            if not valid_timeframes:
                confluence['error'] = 'No valid timeframe data'
                return confluence
            
            # Trend alignment analysis
            trends = {}
            for tf in valid_timeframes:
                indicators = timeframe_data[tf]['indicators']
                trends[tf] = indicators.get('trend', 'NEUTRAL')
            
            # Count trend agreements
            bullish_count = sum(1 for trend in trends.values() if trend == 'BULLISH')
            bearish_count = sum(1 for trend in trends.values() if trend == 'BEARISH')
            
            confluence['trend_alignment'] = {
                'trends': trends,
                'bullish_timeframes': bullish_count,
                'bearish_timeframes': bearish_count,
                'alignment_strength': max(bullish_count, bearish_count) / len(valid_timeframes),
                'dominant_trend': 'BULLISH' if bullish_count > bearish_count else 'BEARISH' if bearish_count > bullish_count else 'MIXED'
            }
            
            # Momentum confluence
            momentum_scores = {}
            for tf in valid_timeframes:
                indicators = timeframe_data[tf]['indicators']
                rsi = indicators.get('rsi', 50)
                momentum_scores[tf] = rsi
            
            avg_momentum = sum(momentum_scores.values()) / len(momentum_scores)
            momentum_alignment = 1 - (np.std(list(momentum_scores.values())) / 50)  # Normalize by max RSI range
            
            confluence['momentum_confluence'] = {
                'individual_rsi': momentum_scores,
                'average_rsi': avg_momentum,
                'alignment_score': momentum_alignment,
                'momentum_bias': 'BULLISH' if avg_momentum > 55 else 'BEARISH' if avg_momentum < 45 else 'NEUTRAL'
            }
            
            # Support/Resistance confluence
            key_levels = {}
            for tf in valid_timeframes:
                indicators = timeframe_data[tf]['indicators']
                if 'support_level' in indicators and 'resistance_level' in indicators:
                    key_levels[tf] = {
                        'support': indicators['support_level'],
                        'resistance': indicators['resistance_level'],
                        'current_price': indicators['price']
                    }
            
            confluence['support_resistance'] = {
                'levels_by_timeframe': key_levels,
                'confluence_zones': self._find_confluence_zones(key_levels)
            }
            
            # Volume confirmation
            volume_ratios = {}
            for tf in valid_timeframes:
                indicators = timeframe_data[tf]['indicators']
                if 'volume_ratio' in indicators:
                    volume_ratios[tf] = indicators['volume_ratio']
            
            avg_volume_ratio = sum(volume_ratios.values()) / len(volume_ratios) if volume_ratios else 1
            
            confluence['volume_confirmation'] = {
                'volume_ratios': volume_ratios,
                'average_volume_ratio': avg_volume_ratio,
                'volume_supporting': avg_volume_ratio > 1.2,
                'volume_strength': 'HIGH' if avg_volume_ratio > 2 else 'MEDIUM' if avg_volume_ratio > 1.2 else 'LOW'
            }
            
            # Calculate overall confluence score
            trend_score = confluence['trend_alignment']['alignment_strength'] * 30
            momentum_score = confluence['momentum_confluence']['alignment_score'] * 25
            volume_score = min(avg_volume_ratio / 2, 1) * 20  # Cap at 2x average volume
            
            # Additional scoring factors
            timeframe_coverage = len(valid_timeframes) / len(self.timeframes) * 25
            
            confluence['overall_score'] = int(trend_score + momentum_score + volume_score + timeframe_coverage)
            
            logger.info(f"Confluence analysis complete: Score {confluence['overall_score']}/100")
            
        except Exception as e:
            logger.error(f"Confluence analysis error: {e}")
            confluence['error'] = str(e)
        
        return confluence
    
    def _find_confluence_zones(self, key_levels: Dict) -> Dict:
        """Find price zones where multiple timeframes agree on support/resistance"""
        
        confluence_zones = {
            'support_zones': [],
            'resistance_zones': []
        }
        
        if not key_levels:
            return confluence_zones
        
        # Extract all support and resistance levels
        all_supports = []
        all_resistances = []
        
        for tf, levels in key_levels.items():
            all_supports.append(levels['support'])
            all_resistances.append(levels['resistance'])
        
        # Find clusters (levels within 2% of each other)
        def find_clusters(levels, tolerance=0.02):
            clusters = []
            sorted_levels = sorted(levels)
            
            for level in sorted_levels:
                # Check if this level belongs to an existing cluster
                added_to_cluster = False
                for cluster in clusters:
                    if any(abs(level - existing) / existing < tolerance for existing in cluster):
                        cluster.append(level)
                        added_to_cluster = True
                        break
                
                if not added_to_cluster:
                    clusters.append([level])
            
            return clusters
        
        # Find support and resistance clusters
        support_clusters = find_clusters(all_supports)
        resistance_clusters = find_clusters(all_resistances)
        
        # Convert clusters to zones
        for cluster in support_clusters:
            if len(cluster) >= 2:  # At least 2 timeframes agree
                confluence_zones['support_zones'].append({
                    'level': sum(cluster) / len(cluster),  # Average level
                    'strength': len(cluster),
                    'range': [min(cluster), max(cluster)]
                })
        
        for cluster in resistance_clusters:
            if len(cluster) >= 2:
                confluence_zones['resistance_zones'].append({
                    'level': sum(cluster) / len(cluster),
                    'strength': len(cluster),
                    'range': [min(cluster), max(cluster)]
                })
        
        return confluence_zones
    
    def _generate_enhanced_signals(self, timeframe_data: Dict, confluence: Dict) -> Dict:
        """Generate enhanced trading signals based on multi-timeframe analysis"""
        
        signals = {
            'primary_signal': 'NEUTRAL',
            'signal_strength': 0,
            'entry_conditions': [],
            'risk_factors': [],
            'timeframe_signals': {}
        }
        
        try:
            valid_timeframes = [tf for tf, data in timeframe_data.items() 
                              if 'indicators' in data and 'error' not in data['indicators']]
            
            if not valid_timeframes:
                signals['error'] = 'No valid data for signal generation'
                return signals
            
            # Collect signals from each timeframe
            bullish_signals = 0
            bearish_signals = 0
            signal_weights = 0
            
            for tf in valid_timeframes:
                indicators = timeframe_data[tf]['indicators']
                tf_weight = self.timeframes[tf]['weight']
                
                # Determine timeframe signal
                tf_signal = 'NEUTRAL'
                tf_strength = 0
                
                # Trend-based signals
                trend = indicators.get('trend', 'NEUTRAL')
                rsi = indicators.get('rsi', 50)
                macd = indicators.get('macd', 0)
                macd_signal = indicators.get('macd_signal', 0)
                
                if trend == 'BULLISH' and rsi > 50 and macd > macd_signal:
                    tf_signal = 'BULLISH'
                    tf_strength = 70 + min((rsi - 50) / 50 * 30, 30)
                elif trend == 'BEARISH' and rsi < 50 and macd < macd_signal:
                    tf_signal = 'BEARISH'
                    tf_strength = 70 + min((50 - rsi) / 50 * 30, 30)
                else:
                    # Check for reversal signals
                    if rsi < 30 and trend != 'BEARISH':
                        tf_signal = 'BULLISH_REVERSAL'
                        tf_strength = 60
                    elif rsi > 70 and trend != 'BULLISH':
                        tf_signal = 'BEARISH_REVERSAL'
                        tf_strength = 60
                
                signals['timeframe_signals'][tf] = {
                    'signal': tf_signal,
                    'strength': tf_strength,
                    'weight': tf_weight
                }
                
                # Accumulate weighted signals
                if 'BULLISH' in tf_signal:
                    bullish_signals += tf_strength * tf_weight
                elif 'BEARISH' in tf_signal:
                    bearish_signals += tf_strength * tf_weight
                
                signal_weights += tf_weight
            
            # Determine primary signal
            if bullish_signals > bearish_signals and bullish_signals > 40:
                signals['primary_signal'] = 'BULLISH'
                signals['signal_strength'] = int(bullish_signals)
            elif bearish_signals > bullish_signals and bearish_signals > 40:
                signals['primary_signal'] = 'BEARISH'  
                signals['signal_strength'] = int(bearish_signals)
            else:
                signals['primary_signal'] = 'NEUTRAL'
                signals['signal_strength'] = max(bullish_signals, bearish_signals)
            
            # Add confluence boost
            confluence_score = confluence.get('overall_score', 0)
            if confluence_score > 70:
                signals['signal_strength'] = min(signals['signal_strength'] + 15, 100)
                signals['entry_conditions'].append(f"High confluence support ({confluence_score}/100)")
            
            # Generate entry conditions and risk factors
            signals['entry_conditions'], signals['risk_factors'] = self._analyze_entry_conditions(
                timeframe_data, confluence, signals['primary_signal']
            )
            
            logger.info(f"Generated {signals['primary_signal']} signal with {signals['signal_strength']} strength")
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            signals['error'] = str(e)
        
        return signals
    
    def _analyze_entry_conditions(self, timeframe_data: Dict, confluence: Dict, primary_signal: str) -> Tuple[List[str], List[str]]:
        """Analyze specific entry conditions and risk factors"""
        
        entry_conditions = []
        risk_factors = []
        
        try:
            # Get daily timeframe data (primary)
            daily_data = timeframe_data.get('1d', {}).get('indicators', {})
            weekly_data = timeframe_data.get('1w', {}).get('indicators', {})
            
            if not daily_data:
                return ['No daily data available'], ['Data quality insufficient']
            
            # Volume conditions
            volume_ratio = daily_data.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                entry_conditions.append(f"High volume confirmation ({volume_ratio:.1f}x average)")
            elif volume_ratio < 0.8:
                risk_factors.append(f"Low volume ({volume_ratio:.1f}x average)")
            
            # Volatility conditions
            volatility = daily_data.get('volatility', 'UNKNOWN')
            if volatility == 'HIGH':
                risk_factors.append("High volatility environment")
            elif volatility == 'LOW':
                entry_conditions.append("Low volatility - potential breakout setup")
            
            # RSI conditions
            rsi = daily_data.get('rsi', 50)
            if primary_signal == 'BULLISH':
                if 40 <= rsi <= 60:
                    entry_conditions.append(f"Healthy RSI for bullish entry ({rsi:.1f})")
                elif rsi > 70:
                    risk_factors.append(f"Overbought RSI ({rsi:.1f}) - potential pullback risk")
            elif primary_signal == 'BEARISH':
                if 40 <= rsi <= 60:
                    entry_conditions.append(f"Neutral RSI for bearish entry ({rsi:.1f})")
                elif rsi < 30:
                    risk_factors.append(f"Oversold RSI ({rsi:.1f}) - potential bounce risk")
            
            # Bollinger Band position
            bb_position = daily_data.get('position_in_range', 0.5)
            if bb_position > 0.8:
                risk_factors.append("Price near resistance level")
            elif bb_position < 0.2:
                risk_factors.append("Price near support level")
            
            # Weekly trend alignment
            if weekly_data:
                weekly_trend = weekly_data.get('trend', 'NEUTRAL')
                if primary_signal == 'BULLISH' and weekly_trend == 'BULLISH':
                    entry_conditions.append("Weekly trend alignment")
                elif primary_signal == 'BEARISH' and weekly_trend == 'BEARISH':
                    entry_conditions.append("Weekly trend alignment")
                elif weekly_trend != 'NEUTRAL' and weekly_trend != primary_signal.replace('_REVERSAL', ''):
                    risk_factors.append(f"Weekly trend divergence ({weekly_trend})")
            
            # Confluence zones
            support_zones = confluence.get('support_resistance', {}).get('confluence_zones', {}).get('support_zones', [])
            resistance_zones = confluence.get('support_resistance', {}).get('confluence_zones', {}).get('resistance_zones', [])
            
            if support_zones and primary_signal == 'BULLISH':
                entry_conditions.append(f"Multiple timeframe support confluence ({len(support_zones)} zones)")
            if resistance_zones and primary_signal == 'BEARISH':
                entry_conditions.append(f"Multiple timeframe resistance confluence ({len(resistance_zones)} zones)")
            
        except Exception as e:
            logger.error(f"Entry conditions analysis error: {e}")
            risk_factors.append(f"Analysis error: {str(e)}")
        
        return entry_conditions, risk_factors
    
    def _assess_data_quality(self, timeframe_data: Dict) -> Dict:
        """Assess the quality of the collected data"""
        
        quality = {
            'overall_score': 0,
            'timeframe_coverage': 0,
            'data_completeness': {},
            'issues': []
        }
        
        try:
            total_timeframes = len(self.timeframes)
            valid_timeframes = 0
            
            for tf, data in timeframe_data.items():
                if 'error' in data:
                    quality['issues'].append(f"{tf}: {data['error']}")
                    quality['data_completeness'][tf] = 0
                elif 'indicators' in data and 'error' not in data['indicators']:
                    valid_timeframes += 1
                    # Check data completeness
                    expected_indicators = 20  # Expected number of key indicators
                    actual_indicators = len([k for k, v in data['indicators'].items() 
                                           if v is not None and k != 'error'])
                    completeness = min(actual_indicators / expected_indicators, 1.0)
                    quality['data_completeness'][tf] = completeness
                    
                    if completeness < 0.8:
                        quality['issues'].append(f"{tf}: Incomplete indicators ({actual_indicators}/{expected_indicators})")
                else:
                    quality['issues'].append(f"{tf}: No indicators calculated")
                    quality['data_completeness'][tf] = 0
            
            quality['timeframe_coverage'] = valid_timeframes / total_timeframes
            
            # Calculate overall score
            avg_completeness = sum(quality['data_completeness'].values()) / len(quality['data_completeness']) if quality['data_completeness'] else 0
            quality['overall_score'] = int((quality['timeframe_coverage'] * 0.6 + avg_completeness * 0.4) * 100)
            
            logger.info(f"Data quality assessment: {quality['overall_score']}/100")
            
        except Exception as e:
            logger.error(f"Data quality assessment error: {e}")
            quality['issues'].append(f"Assessment error: {str(e)}")
        
        return quality
    
    def _save_timeframe_data(self, symbol: str, timeframe_data: Dict):
        """Save raw timeframe data to files"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol.replace('/', '_')}_timeframe_data_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Prepare data for JSON serialization
            serializable_data = {}
            for tf, data in timeframe_data.items():
                if 'ohlcv' in data:
                    # Convert any numpy types to native Python types
                    serializable_data[tf] = {
                        'data_points': data.get('data_points', 0),
                        'date_range': data.get('date_range', {}),
                        'indicators': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                     for k, v in data.get('indicators', {}).items()},
                        'has_ohlcv_data': True
                    }
                else:
                    serializable_data[tf] = data
            
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            logger.info(f"Timeframe data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving timeframe data: {e}")

def main():
    """Test the Enhanced Multi-Timeframe Analyzer"""
    
    print("🚀 ENHANCED MULTI-TIMEFRAME ANALYZER")
    print("AI Feedback Integrated - Expected +25% Confidence Boost")
    print("=" * 70)
    
    analyzer = EnhancedMultiTimeframeAnalyzer()
    
    # Test symbols
    test_symbols = ['BTC/USDT', 'ETH/USDT']
    
    for symbol in test_symbols:
        print(f"\n🎯 Analyzing {symbol}...")
        
        result = analyzer.analyze_multi_timeframe(symbol)
        
        if 'error' in result:
            print(f"❌ Analysis failed: {result['error']}")
            continue
        
        # Display key results
        confluence = result.get('confluence_analysis', {})
        signals = result.get('enhanced_signals', {})
        quality = result.get('data_quality', {})
        
        print(f"✅ Analysis Complete:")
        print(f"   Data Quality: {quality.get('overall_score', 'N/A')}/100")
        print(f"   Confluence Score: {confluence.get('overall_score', 'N/A')}/100")
        print(f"   Primary Signal: {signals.get('primary_signal', 'N/A')} ({signals.get('signal_strength', 'N/A')}%)")
        
        if signals.get('entry_conditions'):
            print(f"   Entry Conditions: {len(signals['entry_conditions'])}")
        if signals.get('risk_factors'):
            print(f"   Risk Factors: {len(signals['risk_factors'])}")

if __name__ == "__main__":
    main()