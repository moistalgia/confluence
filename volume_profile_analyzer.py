#!/usr/bin/env python3
"""
Volume Profile Analyzer - Tier 1 AI Feedback Implementation
Addresses the #2 highest impact missing data: Volume-at-Price (VAP)
Expected improvement: +20% confidence boost according to AI analysis
"""

import pandas as pd
import numpy as np
import ccxt
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VolumeProfileAnalyzer:
    """
    Volume Profile Analysis - Game Changer #2 from AI feedback
    
    Implements:
    - Volume-at-Price (VAP): Where most trading occurred
    - Point of Control (POC): Price level with highest volume
    - Value Area: Where 70% of volume traded (institutional zone)
    - High/Low Volume Nodes (HVN/LVN): Support/resistance zones
    """
    
    def __init__(self):
        self.exchanges = self._initialize_exchanges()
        
    def _initialize_exchanges(self) -> List[Dict]:
        """Initialize exchanges for volume data"""
        exchange_configs = [
            {'class': ccxt.kraken, 'name': 'Kraken', 'priority': 1},
            {'class': ccxt.kucoin, 'name': 'KuCoin', 'priority': 2},
        ]
        
        available_exchanges = []
        
        for config in exchange_configs:
            try:
                exchange = config['class']({
                    'enableRateLimit': True,
                    'rateLimit': 1500,
                    'timeout': 30000,
                })
                
                exchange.load_markets()
                available_exchanges.append({
                    'exchange': exchange,
                    'name': config['name'],
                    'priority': config['priority']
                })
                
                logger.info(f"✅ {config['name']}: Connected for volume data")
                
            except Exception as e:
                logger.warning(f"❌ {config['name']}: {str(e)[:100]}...")
        
        return sorted(available_exchanges, key=lambda x: x['priority'])
    
    def fetch_volume_data(self, symbol: str, timeframe: str = '1h', days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for volume profile analysis
        """
        logger.info(f"Fetching volume data for {symbol} ({timeframe}, {days} days)...")
        
        limit = days * 24 if timeframe == '1h' else days * 24 * 4 if timeframe == '15m' else days
        
        for exchange_info in self.exchanges:
            exchange = exchange_info['exchange']
            exchange_name = exchange_info['name']
            
            try:
                if symbol not in exchange.markets:
                    continue
                
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv or len(ohlcv) < 100:
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"✅ Volume data from {exchange_name}: {len(df)} periods")
                logger.info(f"Volume range: {df['volume'].min():.4f} to {df['volume'].max():.4f}")
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching volume data from {exchange_name}: {e}")
                continue
        
        logger.error(f"❌ Could not fetch volume data for {symbol}")
        return None
    
    def calculate_volume_profile(self, df: pd.DataFrame, price_bins: int = 100) -> Dict:
        """
        Calculate Volume Profile - Core AI feedback implementation
        
        Returns:
        - Volume-at-Price distribution
        - Point of Control (POC)
        - Value Area High/Low (70% volume zone)
        - High/Low Volume Nodes
        """
        logger.info("Calculating Volume Profile (VAP)...")
        
        try:
            # Create price range
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_step = (price_max - price_min) / price_bins
            
            # Create price levels
            price_levels = np.arange(price_min, price_max + price_step, price_step)
            volume_at_price = np.zeros(len(price_levels) - 1)
            
            # Calculate volume at each price level
            for _, row in df.iterrows():
                # Distribute volume across the price range for this candle
                candle_low = row['low']
                candle_high = row['high']
                candle_volume = row['volume']
                
                # Find which price bins this candle spans
                low_bin = np.digitize(candle_low, price_levels) - 1
                high_bin = np.digitize(candle_high, price_levels) - 1
                
                # Distribute volume evenly across the bins (simple approximation)
                bins_spanned = max(1, high_bin - low_bin + 1)
                volume_per_bin = candle_volume / bins_spanned
                
                for bin_idx in range(max(0, low_bin), min(len(volume_at_price), high_bin + 1)):
                    volume_at_price[bin_idx] += volume_per_bin
            
            # Calculate price centers for each bin
            price_centers = (price_levels[:-1] + price_levels[1:]) / 2
            
            # Find Point of Control (POC) - price with highest volume
            poc_idx = np.argmax(volume_at_price)
            poc_price = price_centers[poc_idx]
            poc_volume = volume_at_price[poc_idx]
            
            # Calculate Value Area (70% of total volume)
            total_volume = np.sum(volume_at_price)
            value_area_volume = total_volume * 0.70
            
            # Find Value Area boundaries
            sorted_indices = np.argsort(volume_at_price)[::-1]  # Descending order
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                cumulative_volume += volume_at_price[idx]
                value_area_indices.append(idx)
                if cumulative_volume >= value_area_volume:
                    break
            
            # Value Area High and Low
            va_high = price_centers[max(value_area_indices)]
            va_low = price_centers[min(value_area_indices)]
            
            # Identify High Volume Nodes (HVN) and Low Volume Nodes (LVN)
            volume_threshold_high = np.percentile(volume_at_price, 80)
            volume_threshold_low = np.percentile(volume_at_price, 20)
            
            hvn_levels = []
            lvn_levels = []
            
            for i, (price, volume) in enumerate(zip(price_centers, volume_at_price)):
                if volume >= volume_threshold_high:
                    hvn_levels.append(float(price))
                elif volume <= volume_threshold_low:
                    lvn_levels.append(float(price))
            
            volume_profile = {
                'poc': {
                    'price': float(poc_price),
                    'volume': float(poc_volume),
                    'description': 'Point of Control - Highest traded volume price'
                },
                'value_area': {
                    'high': float(va_high),
                    'low': float(va_low),
                    'volume_percentage': 70.0,
                    'description': 'Institutional pricing zone where 70% of volume traded'
                },
                'hvn_levels': hvn_levels,
                'lvn_levels': lvn_levels,
                'total_volume': float(total_volume),
                'price_range': {
                    'min': float(price_min),
                    'max': float(price_max)
                },
                'volume_distribution': {
                    'prices': [float(p) for p in price_centers],
                    'volumes': [float(v) for v in volume_at_price]
                }
            }
            
            logger.info(f"✅ Volume Profile calculated")
            logger.info(f"POC: ${poc_price:.2f} (Volume: {poc_volume:.4f})")
            logger.info(f"Value Area: ${va_low:.2f} - ${va_high:.2f}")
            logger.info(f"HVN Levels: {len(hvn_levels)} identified")
            logger.info(f"LVN Levels: {len(lvn_levels)} identified")
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return {}
    
    def analyze_current_price_context(self, current_price: float, volume_profile: Dict) -> Dict:
        """
        Analyze current price relative to volume profile
        This is the key insight for trading decisions
        """
        logger.info(f"Analyzing price context for ${current_price:.2f}...")
        
        try:
            poc_price = volume_profile['poc']['price']
            va_high = volume_profile['value_area']['high']
            va_low = volume_profile['value_area']['low']
            hvn_levels = volume_profile['hvn_levels']
            lvn_levels = volume_profile['lvn_levels']
            
            # Determine price position
            if va_low <= current_price <= va_high:
                position = "INSIDE_VALUE_AREA"
                significance = "Institutional pricing zone - expect choppy price action"
            elif current_price > va_high:
                position = "ABOVE_VALUE_AREA"
                significance = "Above institutional fair value - potential selling pressure"
            else:
                position = "BELOW_VALUE_AREA" 
                significance = "Below institutional fair value - potential buying interest"
            
            # Find nearest HVN/LVN levels
            nearest_hvn = min(hvn_levels, key=lambda x: abs(x - current_price)) if hvn_levels else None
            nearest_lvn = min(lvn_levels, key=lambda x: abs(x - current_price)) if lvn_levels else None
            
            # Distance to POC
            poc_distance_pct = ((current_price - poc_price) / poc_price) * 100
            
            # Support/Resistance analysis based on volume
            support_levels = []
            resistance_levels = []
            
            for hvn_price in hvn_levels:
                if hvn_price < current_price:
                    support_levels.append(hvn_price)
                else:
                    resistance_levels.append(hvn_price)
            
            # Sort levels by distance to current price
            support_levels.sort(reverse=True)  # Closest support first
            resistance_levels.sort()           # Closest resistance first
            
            context_analysis = {
                'current_price': current_price,
                'position': position,
                'significance': significance,
                'poc_distance_percent': round(poc_distance_pct, 2),
                'nearest_hvn': nearest_hvn,
                'nearest_lvn': nearest_lvn,
                'volume_based_levels': {
                    'support': support_levels[:3],  # Top 3 support levels
                    'resistance': resistance_levels[:3]  # Top 3 resistance levels
                },
                'trading_implications': self._get_trading_implications(position, poc_distance_pct, current_price, va_high, va_low)
            }
            
            logger.info(f"✅ Price context analysis complete")
            logger.info(f"Position: {position}")
            logger.info(f"POC Distance: {poc_distance_pct:.2f}%")
            
            return context_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing price context: {e}")
            return {}
    
    def _get_trading_implications(self, position: str, poc_distance: float, current_price: float, va_high: float, va_low: float) -> List[str]:
        """Generate trading implications based on volume profile analysis"""
        
        implications = []
        
        if position == "INSIDE_VALUE_AREA":
            implications.extend([
                "Price in institutional fair value zone",
                "Expect choppy, range-bound trading",
                "Look for breakout above VA High or breakdown below VA Low",
                "Low probability directional moves within VA"
            ])
        
        elif position == "ABOVE_VALUE_AREA":
            implications.extend([
                "Price above institutional fair value",
                "Potential selling pressure from profit-taking",
                f"Key support at VA High: ${va_high:.2f}",
                "Watch for rejection or continuation above VA"
            ])
        
        elif position == "BELOW_VALUE_AREA":
            implications.extend([
                "Price below institutional fair value", 
                "Potential buying interest from value seekers",
                f"Key resistance at VA Low: ${va_low:.2f}",
                "Watch for bounce or further breakdown"
            ])
        
        # POC distance implications
        if abs(poc_distance) > 5:
            implications.append(f"Significant deviation from POC ({poc_distance:.1f}%) - expect reversion pressure")
        elif abs(poc_distance) < 2:
            implications.append("Close to POC - balanced market, watch for directional break")
        
        return implications
    
    def _generate_trading_signals(self, current_price: float, volume_profile: Dict, price_context: Dict) -> Dict:
        """Generate trading signals based on volume profile analysis"""
        signals = {
            'signal_strength': 'NEUTRAL',
            'direction': 'SIDEWAYS',
            'confidence': 0.0,
            'entry_signals': [],
            'exit_signals': [],
            'risk_factors': []
        }
        
        try:
            poc_price = volume_profile.get('point_of_control', {}).get('price', current_price)
            value_area_high = volume_profile.get('value_area', {}).get('high', current_price)
            value_area_low = volume_profile.get('value_area', {}).get('low', current_price)
            
            # Price position analysis
            distance_from_poc = abs(current_price - poc_price) / poc_price * 100
            
            # Generate signals based on position relative to volume zones
            if current_price > value_area_high:
                signals['direction'] = 'BEARISH'
                signals['entry_signals'].append('Price above value area - potential rejection zone')
                signals['signal_strength'] = 'STRONG' if distance_from_poc > 2 else 'MODERATE'
                signals['confidence'] = min(0.8, distance_from_poc / 3)
                
            elif current_price < value_area_low:
                signals['direction'] = 'BULLISH'
                signals['entry_signals'].append('Price below value area - potential support zone')
                signals['signal_strength'] = 'STRONG' if distance_from_poc > 2 else 'MODERATE'
                signals['confidence'] = min(0.8, distance_from_poc / 3)
                
            elif abs(current_price - poc_price) / poc_price < 0.01:  # Within 1% of POC
                signals['direction'] = 'NEUTRAL'
                signals['entry_signals'].append('Price near Point of Control - equilibrium zone')
                signals['signal_strength'] = 'WEAK'
                signals['confidence'] = 0.3
                
            # Add volume-based risk factors
            hvn_count = len(volume_profile.get('high_volume_nodes', []))
            if hvn_count < 3:
                signals['risk_factors'].append('Limited volume support levels')
            
        except Exception as e:
            logger.warning(f"Error generating trading signals: {e}")
            signals['error'] = str(e)
            
        return signals
    
    def _generate_market_context(self, symbol: str, timeframe: str, days: int, 
                               volume_profile: Dict, price_context: Dict) -> Dict:
        """Generate comprehensive market context analysis"""
        context = {
            'market_structure': 'UNKNOWN',
            'volume_trend': 'NEUTRAL',
            'institutional_zones': [],
            'support_resistance': [],
            'market_sentiment': 'NEUTRAL',
            'liquidity_analysis': {}
        }
        
        try:
            # Analyze market structure based on volume distribution
            total_volume = volume_profile.get('total_volume', 0)
            value_area = volume_profile.get('value_area', {})
            
            if total_volume > 0 and value_area:
                value_area_volume = value_area.get('volume_percentage', 0)
                
                if value_area_volume > 75:
                    context['market_structure'] = 'CONSOLIDATION'
                    context['market_sentiment'] = 'ACCUMULATION'
                elif value_area_volume < 50:
                    context['market_structure'] = 'TRENDING'
                    context['market_sentiment'] = 'DISTRIBUTION'
                else:
                    context['market_structure'] = 'BALANCED'
                    context['market_sentiment'] = 'NEUTRAL'
            
            # Extract institutional zones (high volume nodes)
            hvn_nodes = volume_profile.get('high_volume_nodes', [])
            for i, node in enumerate(hvn_nodes[:5]):  # Top 5 HVN zones
                context['institutional_zones'].append({
                    'price_level': node.get('price', 0),
                    'volume': node.get('volume', 0),
                    'strength': 'HIGH' if i < 2 else 'MODERATE',
                    'type': 'INSTITUTIONAL_ZONE'
                })
            
            # Support/Resistance from volume profile
            poc = volume_profile.get('point_of_control', {})
            if poc:
                context['support_resistance'].append({
                    'level': poc.get('price', 0),
                    'type': 'POC_SUPPORT_RESISTANCE',
                    'strength': 'VERY_HIGH',
                    'volume': poc.get('volume', 0)
                })
            
            # Liquidity analysis
            context['liquidity_analysis'] = {
                'total_volume_analyzed': total_volume,
                'analysis_period': f"{days} days",
                'timeframe': timeframe,
                'volume_concentration': value_area.get('volume_percentage', 0),
                'liquidity_rating': 'HIGH' if total_volume > 1000000 else 'MODERATE' if total_volume > 100000 else 'LOW'
            }
            
        except Exception as e:
            logger.warning(f"Error generating market context: {e}")
            context['error'] = str(e)
            
        return context
    
    def generate_volume_based_signals(self, symbol: str, timeframe: str = '1h', days: int = 30) -> Dict:
        """
        Generate comprehensive volume-based trading signals
        Main function that combines all volume profile analysis
        """
        logger.info(f"Generating volume-based signals for {symbol}...")
        
        # Fetch volume data
        df = self.fetch_volume_data(symbol, timeframe, days)
        if df is None:
            return {'error': 'No volume data available'}
        
        # Calculate volume profile
        volume_profile = self.calculate_volume_profile(df)
        if not volume_profile:
            return {'error': 'Volume profile calculation failed'}
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Analyze price context
        price_context = self.analyze_current_price_context(current_price, volume_profile)
        
        # Generate trading signals based on volume analysis
        trading_signals = self._generate_trading_signals(current_price, volume_profile, price_context)
        
        # Generate market context analysis
        market_context = self._generate_market_context(symbol, timeframe, days, volume_profile, price_context)
        
        # Combine all analysis with expected field structure for Ultimate Crypto Analyzer
        comprehensive_analysis = {
            'volume_profile': volume_profile,
            'price_analysis': price_context,  # Renamed from price_context to match expectations
            'trading_signals': trading_signals,
            'market_context': market_context,
            # Additional metadata (not required by Ultimate system but useful)
            'metadata': {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'analysis_period_days': days,
                'current_price': float(current_price),
                'ai_feedback_implementation': {
                    'missing_data_addressed': 'Volume-at-Price (VAP)',
                    'confidence_boost_expected': '+20%',
                    'tier': 'Tier 1 - Game Changer'
                }
            }
        }
        
        return comprehensive_analysis

def main():
    """Test the Volume Profile Analyzer"""
    
    print("📊 Volume Profile Analyzer - AI Feedback Implementation")
    print("Tier 1 Enhancement: +20% confidence boost expected")
    print("=" * 60)
    
    try:
        analyzer = VolumeProfileAnalyzer()
        
        if not analyzer.exchanges:
            print("❌ No exchanges available")
            return
        
        # Test with BTC
        symbol = 'BTC/USDT'
        
        # Generate volume-based signals
        analysis = analyzer.generate_volume_based_signals(symbol, timeframe='1h', days=30)
        
        if 'error' in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            return
        
        # Display results
        print(f"\n✅ Volume Profile Analysis Complete for {symbol}")
        print(f"Analysis Period: {analysis['analysis_period_days']} days")
        print(f"Current Price: ${analysis['current_price']:.2f}")
        
        # Volume Profile Results
        vp = analysis['volume_profile']
        print(f"\n📈 Volume Profile:")
        print(f"  POC (Point of Control): ${vp['poc']['price']:.2f}")
        print(f"  Value Area High: ${vp['value_area']['high']:.2f}")
        print(f"  Value Area Low: ${vp['value_area']['low']:.2f}")
        print(f"  HVN Levels (Support/Resistance): {len(vp['hvn_levels'])}")
        
        # Price Context
        context = analysis['price_context']
        print(f"\n🎯 Current Price Analysis:")
        print(f"  Position: {context['position']}")
        print(f"  Distance from POC: {context['poc_distance_percent']}%")
        print(f"  Significance: {context['significance']}")
        
        # Trading Implications
        print(f"\n💡 Trading Implications:")
        for implication in context['trading_implications']:
            print(f"  • {implication}")
        
        # Volume-Based Levels
        levels = context['volume_based_levels']
        if levels['support']:
            print(f"\n📉 Volume-Based Support Levels:")
            for i, level in enumerate(levels['support'][:3], 1):
                print(f"  {i}. ${level:.2f}")
        
        if levels['resistance']:
            print(f"\n📈 Volume-Based Resistance Levels:")
            for i, level in enumerate(levels['resistance'][:3], 1):
                print(f"  {i}. ${level:.2f}")
        
        # Save analysis
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"volume_profile_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Volume Profile analysis saved to: {output_file}")
        
        print(f"\n🎯 AI Feedback Implementation Status:")
        print(f"  ✅ Volume-at-Price (VAP) calculation")
        print(f"  ✅ Point of Control (POC) identification")  
        print(f"  ✅ Value Area (70% volume zone) analysis")
        print(f"  ✅ High/Low Volume Nodes detection")
        print(f"  ✅ Volume-based Support/Resistance levels")
        print(f"  ✅ Expected confidence boost: +20%")
        
    except Exception as e:
        logger.error(f"Error in volume profile analysis: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()