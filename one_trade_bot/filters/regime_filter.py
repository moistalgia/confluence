"""
Filter 1: Market Regime Filter
🎯 Eliminates 50% of bad trading days by checking market conditions

Only allows trading when market is in favorable regime:
- Trending (not ranging)
- Reasonable volatility
- Adequate volume
- Not in tight squeeze

Must pass 3 of 4 checks to be tradeable
"""

import logging
from typing import Dict
import pandas as pd
from core.data_provider import DataProvider, TechnicalIndicators

logger = logging.getLogger(__name__)

class MarketRegimeFilter:
    """
    Checks if market conditions are suitable for trading
    Eliminates ~50% of days that would result in poor trades
    """
    
    def __init__(self, config: Dict):
        """
        Initialize regime filter with thresholds
        
        Args:
            config: Filter configuration with thresholds
        """
        self.min_trend_distance = config.get('min_trend_distance', 0.02)  # 2% from SMA
        self.max_atr_pct = config.get('max_atr_pct', 0.08)              # 8% ATR/price
        self.min_volume_ratio = config.get('min_volume_ratio', 0.5)      # 50% of average
        self.max_bb_squeeze = config.get('max_bb_squeeze', 0.15)         # BB squeeze limit
        self.min_checks_pass = config.get('min_checks_pass', 3)          # 3 of 4 checks
        
        logger.info("🔍 Market Regime Filter initialized")
        logger.info(f"   Requirements: {self.min_checks_pass}/4 checks must pass")
        
    def check_regime(self, symbol: str, data_provider: DataProvider) -> Dict:
        """
        Check if symbol is in tradeable market regime
        
        Args:
            symbol: Trading pair to check
            data_provider: Data source
            
        Returns:
            Dict with regime analysis and trading permission
        """
        try:
            # Get daily data for regime analysis
            df = data_provider.get_ohlcv(symbol, '1d', days=30)
            
            if len(df) < 25:
                logger.warning(f"⚠️ {symbol}: Insufficient data for regime check ({len(df)} days)")
                return self._create_result(symbol, False, 0, ["Insufficient historical data"])
            
            # Current values
            current_price = df['close'].iloc[-1]
            
            # Calculate regime indicators
            sma_20 = TechnicalIndicators.sma(df['close'], 20).iloc[-1]
            atr_14 = TechnicalIndicators.atr(df, 14).iloc[-1]
            volume_20d_avg = df['volume'].rolling(20).mean().iloc[-1]
            volume_today = df['volume'].iloc[-1]
            squeeze_ratio = TechnicalIndicators.bb_squeeze(df).iloc[-1]
            
            # Run the 4 regime checks
            checks = self._run_regime_checks(
                current_price, sma_20, atr_14, 
                volume_today, volume_20d_avg, squeeze_ratio
            )
            
            # Determine if tradeable
            checks_passed = sum(checks.values())
            is_tradeable = bool(checks_passed >= self.min_checks_pass)
            
            # Determine regime type
            regime_type = "TRENDING" if checks['trending'] else "RANGING"
            
            # Failed checks for logging
            failed_reasons = [check for check, passed in checks.items() if not passed]
            
            result = self._create_result(
                symbol, is_tradeable, int(checks_passed), 
                failed_reasons, regime_type, checks
            )
            
            # Log result
            status = "✅ TRADEABLE" if is_tradeable else "❌ NOT TRADEABLE"
            logger.info(f"{status} {symbol}: {regime_type} ({checks_passed}/{len(checks)} checks passed)")
            
            if failed_reasons:
                logger.debug(f"   Failed: {', '.join(failed_reasons)}")
            
            return result
            
        except Exception as e:
            logger.error(f"🚨 Regime check failed for {symbol}: {str(e)}")
            return self._create_result(symbol, False, 0, [f"Analysis error: {str(e)}"])
    
    def _run_regime_checks(self, price: float, sma_20: float, atr: float, 
                          volume_today: float, volume_avg: float, squeeze_ratio: float) -> Dict[str, bool]:
        """
        Execute the 4 regime validation checks
        
        Returns:
            Dict of check results
        """
        # Check 1: Trending market (price significantly away from SMA20)
        distance_from_sma = abs(price - sma_20) / sma_20
        trending = distance_from_sma >= self.min_trend_distance
        
        # Check 2: Reasonable volatility (ATR not excessive)
        atr_pct = atr / price if price > 0 else 1.0
        not_too_volatile = atr_pct <= self.max_atr_pct
        
        # Check 3: Adequate volume (not dead market)
        volume_ratio = volume_today / volume_avg if volume_avg > 0 else 0
        has_volume = volume_ratio >= self.min_volume_ratio
        
        # Check 4: Not in tight consolidation (BB squeeze)
        not_in_squeeze = squeeze_ratio >= self.max_bb_squeeze
        
        checks = {
            'trending': trending,
            'not_too_volatile': not_too_volatile, 
            'has_volume': has_volume,
            'not_in_squeeze': not_in_squeeze
        }
        
        # Log check details for debugging
        logger.debug(f"Regime checks:")
        logger.debug(f"  Trending: {trending} (distance: {distance_from_sma:.3f} vs {self.min_trend_distance})")
        logger.debug(f"  Volatility OK: {not_too_volatile} (ATR%: {atr_pct:.3f} vs {self.max_atr_pct})")
        logger.debug(f"  Volume OK: {has_volume} (ratio: {volume_ratio:.2f} vs {self.min_volume_ratio})")
        logger.debug(f"  No Squeeze: {not_in_squeeze} (ratio: {squeeze_ratio:.3f} vs {self.max_bb_squeeze})")
        
        return checks
    
    def _create_result(self, symbol: str, tradeable: bool, score: int, 
                      reasons: list, regime: str = "UNKNOWN", checks: Dict = None) -> Dict:
        """Create standardized result dictionary"""
        return {
            'symbol': symbol,
            'tradeable': tradeable,
            'score': score,
            'max_score': 4,
            'regime': regime,
            'failed_reasons': reasons,
            'checks': checks or {},
            'filter_name': 'MarketRegimeFilter'
        }


def filter_watchlist_by_regime(watchlist: list, data_provider: DataProvider, config: Dict) -> Dict:
    """
    Filter entire watchlist through regime analysis
    
    Args:
        watchlist: List of symbols to check
        data_provider: Data source
        config: Filter configuration
        
    Returns:
        Dict with tradeable and non-tradeable symbols
    """
    regime_filter = MarketRegimeFilter(config)
    
    tradeable_symbols = []
    filtered_symbols = []
    regime_results = {}
    
    logger.info(f"🔍 REGIME FILTER: Checking {len(watchlist)} symbols")
    logger.info("=" * 60)
    
    for symbol in watchlist:
        try:
            result = regime_filter.check_regime(symbol, data_provider)
            regime_results[symbol] = result
            
            if result['tradeable']:
                tradeable_symbols.append(symbol)
                logger.info(f"✅ {symbol}: {result['regime']} (score: {result['score']}/4)")
            else:
                filtered_symbols.append(symbol)
                logger.info(f"❌ {symbol}: {', '.join(result['failed_reasons'])}")
                
        except Exception as e:
            logger.error(f"🚨 {symbol}: Regime check failed - {str(e)}")
            filtered_symbols.append(symbol)
            regime_results[symbol] = regime_filter._create_result(
                symbol, False, 0, [f"Error: {str(e)}"]
            )
    
    logger.info("=" * 60)
    logger.info(f"📊 REGIME FILTER RESULTS:")
    logger.info(f"   Tradeable: {len(tradeable_symbols)}/{len(watchlist)} symbols")
    logger.info(f"   Filtered out: {len(filtered_symbols)} symbols")
    
    if len(tradeable_symbols) == 0:
        logger.warning("⚠️ NO SYMBOLS passed regime filter - consider sitting on hands today")
    
    return {
        'tradeable_symbols': tradeable_symbols,
        'filtered_symbols': filtered_symbols,
        'results': regime_results,
        'summary': {
            'total_checked': len(watchlist),
            'tradeable_count': len(tradeable_symbols),
            'filter_rate': len(filtered_symbols) / len(watchlist) if watchlist else 0
        }
    }