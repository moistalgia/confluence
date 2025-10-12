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
import time
import functools
from typing import Optional, Any, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
import os
from dataclasses import dataclass
from threading import Lock
from config_manager import ConfigManager, CryptoAnalyzerConfig, ConfigFormat
from alert_system import AlertManager
from sentiment_analyzer_enhanced import EnhancedSentimentAnalyzer, SentimentData

# US-accessible exchanges in order of preference
US_ACCESSIBLE_EXCHANGES = [
    'kraken',      # Primary choice - excellent US support
    'coinbase',    # Coinbase Pro/Advanced Trade
    'gemini',      # US-based exchange
    'ftxus',       # FTX US (if available)
    'bittrex',     # US-compliant
    'kucoin'       # Generally accessible
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Enhanced retry decorator with exponential backoff for API calls and data processing
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
            return None
        return wrapper
    return decorator

def safe_calculation(default_value: Any = None, log_errors: bool = True):
    """
    Safe calculation decorator for technical indicators and analysis methods
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                if result is None:
                    logger.warning(f"{func.__name__} returned None, using default value: {default_value}")
                    return default_value
                return result
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(f"Full traceback for {func.__name__}:", exc_info=True)
                return default_value
        return wrapper
    return decorator

@dataclass
class CacheConfig:
    """Configuration for caching system"""
    enabled: bool = True
    cache_dir: str = "cache"
    ohlcv_ttl: int = 300  # 5 minutes for OHLCV data
    indicators_ttl: int = 60  # 1 minute for calculated indicators
    max_cache_size: int = 100  # Maximum number of cached items

class PerformanceCache:
    """
    High-performance caching system with TTL and size limits
    """
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = {}
        self.cache_timestamps = {}
        self.access_times = {}
        self.lock = Lock()
        
        if config.enabled and not os.path.exists(config.cache_dir):
            os.makedirs(config.cache_dir)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments"""
        key_str = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str, ttl: int = 300) -> Optional[Any]:
        """Get item from cache if not expired"""
        if not self.config.enabled:
            return None
            
        with self.lock:
            if key not in self.cache:
                return None
                
            # Check TTL
            if time.time() - self.cache_timestamps[key] > ttl:
                self._remove_key(key)
                return None
                
            # Update access time for LRU
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with size management"""
        if not self.config.enabled:
            return
            
        with self.lock:
            current_time = time.time()
            
            # Remove expired items
            self._cleanup_expired()
            
            # Check size limit
            if len(self.cache) >= self.config.max_cache_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.cache_timestamps[key] = current_time
            self.access_times[key] = current_time
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all cache structures"""
        self.cache.pop(key, None)
        self.cache_timestamps.pop(key, None)
        self.access_times.pop(key, None)
    
    def _cleanup_expired(self) -> None:
        """Remove expired items from cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            # Use default TTL for cleanup
            if current_time - timestamp > 600:  # 10 minutes default
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def clear(self) -> None:
        """Clear all cached data"""
        with self.lock:
            self.cache.clear()
            self.cache_timestamps.clear()
            self.access_times.clear()

def cached_method(ttl: int = 300, cache_key_func: Optional[Callable] = None):
    """
    Decorator for caching method results with TTL
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            if not hasattr(self, '_cache'):
                return func(self, *args, **kwargs)
            
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(self, *args, **kwargs)
            else:
                cache_key = self._cache._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = self._cache.get(cache_key, ttl)
            if result is not None:
                return result
            
            # Calculate and cache result
            result = func(self, *args, **kwargs)
            if result is not None:
                self._cache.set(cache_key, result)
            
            return result
        return wrapper
    return decorator

class EnhancedMultiTimeframeAnalyzer:
    """Enhanced multi-timeframe analysis system with AI feedback integration"""
    
    def __init__(self, config_profile: Optional[str] = None, config: Optional[CryptoAnalyzerConfig] = None):
        """
        Initialize analyzer with configuration management support
        """
        # Load configuration
        if config:
            self.config = config
        else:
            config_manager = ConfigManager()
            self.config = config_manager.load_config(config_profile or "default")
        
        # Initialize US-accessible exchange with fallback support
        exchange_config = self.config.exchange
        self.exchange = self._initialize_us_accessible_exchange(exchange_config)
        
        # Performance configuration
        perf_config = self.config.performance
        self.parallel_processing = perf_config.parallel_processing
        self.max_workers = perf_config.max_workers
        
        # Initialize high-performance caching system
        cache_config = CacheConfig(
            enabled=perf_config.enable_caching,
            cache_dir=perf_config.cache_dir,
            ohlcv_ttl=perf_config.ohlcv_cache_ttl,
            indicators_ttl=perf_config.indicators_cache_ttl,
            max_cache_size=perf_config.max_cache_size
        )
        self._cache = PerformanceCache(cache_config)
        
        # Convert timeframe configurations
        self.timeframes = {}
        for tf_name, tf_config in self.config.timeframes.items():
            if tf_config.enabled:
                self.timeframes[tf_name] = {
                    'weight': tf_config.weight,
                    'periods': tf_config.periods,
                    'priority': tf_config.priority,
                    'indicators_enabled': tf_config.indicators_enabled
                }
        
        # Validate timeframe weights
        total_weight = sum(tf['weight'] for tf in self.timeframes.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Timeframe weights sum to {total_weight:.2f}, normalizing to 1.0")
            for tf_name in self.timeframes:
                self.timeframes[tf_name]['weight'] /= total_weight
        
        # Performance metrics tracking
        self.performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_executions': 0,
            'total_analysis_time': 0,
            'api_calls_saved': 0
        }
        
        # Initialize alert system
        self.alert_manager = AlertManager()
        self.alert_manager.start_alert_processor()
        
        # Setup output directory from config
        self.output_dir = Path(self.config.output.output_dir) / "raw_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging level
        log_level = getattr(logging, self.config.output.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
        
        logger.info(f"Enhanced analyzer initialized with profile: {self.config.profile_name}")
        logger.info(f"Configuration - Exchange: {exchange_config.exchange_id}, Timeframes: {len(self.timeframes)}, "
                   f"Caching: {perf_config.enable_caching}, Parallel: {perf_config.parallel_processing}")
        logger.info(f"Alert system initialized with {len(self.alert_manager.alert_rules)} rules")
    
    def _initialize_us_accessible_exchange(self, exchange_config):
        """
        Initialize exchange with US-accessible options and fallback support
        """
        # If config specifies a non-US exchange, try to use US alternatives
        preferred_exchange = exchange_config.exchange_id
        if preferred_exchange == 'binance':
            logger.warning("Binance may have geographic restrictions - trying US-accessible exchanges")
            exchanges_to_try = US_ACCESSIBLE_EXCHANGES
        else:
            # Use configured exchange first, then fallback to US options
            exchanges_to_try = [preferred_exchange] + [ex for ex in US_ACCESSIBLE_EXCHANGES if ex != preferred_exchange]
        
        for exchange_id in exchanges_to_try:
            try:
                if not hasattr(ccxt, exchange_id):
                    logger.warning(f"Exchange {exchange_id} not available in ccxt")
                    continue
                
                exchange = getattr(ccxt, exchange_id)({
                    'enableRateLimit': exchange_config.enable_rate_limit,
                    'options': {'defaultType': 'spot'},
                    'timeout': exchange_config.timeout,
                    'rateLimit': exchange_config.rate_limit,
                    'sandbox': exchange_config.sandbox_mode
                })
                
                # Test the connection
                exchange.load_markets()
                logger.info(f"✅ Successfully connected to {exchange_id}")
                return exchange
                
            except Exception as e:
                logger.warning(f"Failed to connect to {exchange_id}: {e}")
                continue
        
        # If all fail, create a basic kraken instance as last resort
        logger.error("All exchanges failed, falling back to basic Kraken configuration")
        return ccxt.kraken({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000,
            'rateLimit': 1000
        })
    
    @classmethod
    def create_with_custom_config(cls, **config_overrides):
        """
        Create analyzer with custom configuration overrides
        """
        config_manager = ConfigManager()
        config = config_manager.load_config("default")
        
        # Apply overrides using the config manager's update method
        temp_manager = ConfigManager()
        temp_manager.current_config = config
        
        if temp_manager.update_config(config_overrides):
            updated_config = temp_manager.get_current_config()
            return cls(config=updated_config)
        else:
            logger.warning("Failed to apply config overrides, using default config")
            return cls(config=config)
    
    def analyze_multi_timeframe(self, symbol: str) -> Dict:
        """
        Enhanced multi-timeframe analysis
        Returns comprehensive analysis across all timeframes
        """
        logger.info(f"DEBUG-ENTRY-MULTI: Starting enhanced multi-timeframe analysis for {symbol}")
        logger.info(f"DEBUG-ENTRY-MULTI: parallel_processing={self.parallel_processing}, timeframes_count={len(self.timeframes)}")
        
        try:
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'timeframe_data': {},
                'confluence_analysis': {},
                'enhanced_signals': {},
                'data_quality': {}
            }
            
            # Collect data for all timeframes with parallel processing and enhanced error handling
            timeframe_data = {}
            successful_timeframes = 0
            
            start_time = time.time()
            
            if self.parallel_processing and len(self.timeframes) > 1:
                # Parallel processing for multiple timeframes
                logger.info(f"Starting parallel analysis for {len(self.timeframes)} timeframes using {self.max_workers} workers")
                self.performance_metrics['parallel_executions'] += 1
                
                timeframe_data = self._analyze_timeframes_parallel(symbol)
                successful_timeframes = sum(1 for data in timeframe_data.values() if data.get('status') == 'success')
                
            else:
                # Sequential processing (fallback or single timeframe)
                logger.info("Starting sequential timeframe analysis")
                
                for tf, config in self.timeframes.items():
                    logger.info(f"Fetching {tf} data for {symbol} - requesting {config['periods']} periods")
                    
                    try:
                        timeframe_result = self._fetch_and_analyze_timeframe(symbol, tf, config)
                        
                        if timeframe_result and timeframe_result.get('status') == 'success':
                            timeframe_data[tf] = timeframe_result
                            successful_timeframes += 1
                            
                            # Enhanced success logging with data quality metrics
                            quality = timeframe_result.get('data_quality', {})
                            completeness = quality.get('completeness', 0)
                            data_points = timeframe_result.get('data_points', 0)
                            
                            logger.info(f"✅ {tf} analysis complete: {data_points} candles, {completeness:.1f}% complete")
                            
                            # Log data quality warnings
                            if completeness < 90:
                                logger.warning(f"⚠️  Low data completeness for {tf}: {completeness:.1f}%")
                            if data_points < config['periods'] * 0.8:
                                logger.warning(f"⚠️  Fewer data points than expected for {tf}: {data_points}/{config['periods']}")
                                
                        else:
                            error_msg = 'Failed to fetch or analyze data'
                            if timeframe_result and 'error' in timeframe_result:
                                error_msg = timeframe_result['error']
                                
                            logger.warning(f"❌ {tf} analysis failed: {error_msg}")
                            timeframe_data[tf] = {
                                'error': error_msg,
                                'status': 'failed',
                                'timeframe': tf,
                                'requested_periods': config['periods']
                            }
                            
                    except Exception as e:
                        error_msg = f"Critical error during {tf} analysis: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        timeframe_data[tf] = {
                            'error': error_msg,
                            'status': 'error',
                            'timeframe': tf,
                            'exception_type': type(e).__name__
                        }
            
            # Track performance metrics
            analysis_time = time.time() - start_time
            self.performance_metrics['total_analysis_time'] += analysis_time
            
            # Log overall success rate
            total_timeframes = len(self.timeframes)
            success_rate = (successful_timeframes / total_timeframes) * 100 if total_timeframes > 0 else 0
            logger.info(f"Timeframe analysis complete: {successful_timeframes}/{total_timeframes} successful ({success_rate:.1f}%)")
            
            # Add analysis metadata
            analysis_result['analysis_metadata'] = {
                'successful_timeframes': successful_timeframes,
                'total_timeframes': total_timeframes,
                'success_rate': success_rate,
                'failed_timeframes': [tf for tf, data in timeframe_data.items() if data.get('status') in ['failed', 'error']]
            }
            
            analysis_result['timeframe_data'] = timeframe_data
            
            # Perform confluence analysis with graceful degradation
            # Initialize confluence with empty structure
            confluence = {
                'overall_confluence': {'confluence_score': 0},
                'error': 'Confluence analysis not executed'
            }
            
            try:
                if successful_timeframes > 0:
                    # Filter out failed timeframes for confluence analysis
                    successful_data = {
                        tf: data for tf, data in timeframe_data.items() 
                        if data.get('status') == 'success'
                    }
                    
                    if len(successful_data) >= 1:
                        confluence = self._analyze_timeframe_confluence(successful_data)
                        analysis_result['confluence_analysis'] = confluence
                        
                        if len(successful_data) < total_timeframes:
                            confluence['analysis_note'] = f"Analysis based on {len(successful_data)}/{total_timeframes} successful timeframes"
                            logger.info(f"Confluence analysis using {len(successful_data)} successful timeframes")
                    else:
                        analysis_result['confluence_analysis'] = {
                            'error': 'No successful timeframe data for confluence analysis',
                            'status': 'failed'
                        }
                        logger.error("No successful timeframes available for confluence analysis")
                else:
                    analysis_result['confluence_analysis'] = {
                        'error': 'All timeframe analyses failed',
                        'status': 'failed'
                    }
                    logger.error("All timeframe analyses failed - no confluence analysis possible")
                    
            except Exception as e:
                logger.error(f"Error in confluence analysis: {e}", exc_info=True)
                analysis_result['confluence_analysis'] = {
                    'error': f'Confluence analysis failed: {str(e)}',
                    'status': 'error'
                }
                # Keep the default confluence structure for signal generation
            
            # Generate enhanced trading signals
            signals = self._generate_enhanced_signals(timeframe_data, confluence)
            analysis_result['enhanced_signals'] = signals
            
            # Enhanced Sentiment Analysis
            try:
                logger.info("🔍 Analyzing market sentiment...")
                sentiment_analyzer = EnhancedSentimentAnalyzer()
                sentiment_data = sentiment_analyzer.get_comprehensive_sentiment(symbol)
                
                analysis_result['sentiment_analysis'] = {
                    'fear_greed_index': sentiment_data.fear_greed_index,
                    'fear_greed_classification': sentiment_data.fear_greed_classification,
                    'social_sentiment': sentiment_data.social_sentiment,
                    'funding_rate': sentiment_data.funding_rate,
                    'funding_sentiment': sentiment_data.funding_sentiment,
                    'overall_sentiment': sentiment_data.overall_sentiment,
                    'sentiment_score': sentiment_data.sentiment_score,
                    'data_sources': sentiment_data.data_sources,
                    'timestamp': sentiment_data.timestamp
                }
                
                logger.info(f"✅ Sentiment analysis complete: {sentiment_data.overall_sentiment} ({sentiment_data.sentiment_score}/100)")
                
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                analysis_result['sentiment_analysis'] = {
                    'error': str(e),
                    'status': 'failed',
                    'overall_sentiment': 'NEUTRAL',
                    'sentiment_score': 50
                }
            
            # Assess data quality
            quality = self._assess_data_quality(timeframe_data)
            analysis_result['data_quality'] = quality
            
            # Save raw data
            self._save_timeframe_data(symbol, timeframe_data)
            
            # Add performance metrics to result
            cache_hit_rate = (self.performance_metrics['cache_hits'] / 
                            max(self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'], 1)) * 100
            
            analysis_result['performance_metrics'] = {
                'analysis_time_seconds': analysis_time,
                'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                'api_calls_saved': self.performance_metrics['api_calls_saved'],
                'parallel_processing_used': self.parallel_processing and len(self.timeframes) > 1,
                'successful_timeframes': successful_timeframes,
                'total_timeframes': len(self.timeframes)
            }
            
            # Check for alert conditions
            try:
                self.alert_manager.check_analysis_for_alerts(symbol, analysis_result)
                logger.debug(f"Alert check completed for {symbol}")
            except Exception as e:
                logger.error(f"Alert checking failed for {symbol}: {e}")
            
            logger.info(f"Enhanced analysis complete for {symbol} - {analysis_time:.2f}s, {cache_hit_rate:.1f}% cache hits")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Enhanced multi-timeframe analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
    
    @cached_method(ttl=300)  # 5-minute cache for OHLCV data
    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    def _fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int) -> Optional[List]:
        """
        Cached OHLCV data fetching with performance optimization
        """
        try:
            logger.debug(f"Fetching OHLCV data: {symbol} {timeframe} (limit: {limit})")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv and len(ohlcv) > 0:
                self.performance_metrics['api_calls_saved'] += 1
                return ohlcv
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} {timeframe}: {e}")
            raise
    
    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    def _fetch_and_analyze_timeframe(self, symbol: str, timeframe: str, config: Dict) -> Optional[Dict]:
        """
        Enhanced timeframe data fetching with comprehensive error handling and validation
        """
        logger.info(f"DEBUG-ENTRY: _fetch_and_analyze_timeframe called for {symbol} {timeframe}")
        try:
            # Validate inputs
            if not symbol or not timeframe:
                raise ValueError(f"Invalid symbol ({symbol}) or timeframe ({timeframe})")
            
            if config.get('periods', 0) <= 0:
                raise ValueError(f"Invalid periods configuration: {config.get('periods')}")
            
            # Fetch OHLCV data with caching and timeout protection
            logger.debug(f"Fetching {config['periods']} periods of {timeframe} data for {symbol}")
            
            # Use cached OHLCV method for better performance
            ohlcv = self._fetch_ohlcv_data(symbol, timeframe, config['periods'])
            
            # Track cache performance
            if ohlcv:
                self.performance_metrics['cache_hits'] += 1
            else:
                self.performance_metrics['cache_misses'] += 1
            
            # Validate data response
            if not ohlcv:
                logger.warning(f"Empty data response for {symbol} {timeframe}")
                return None
                
            if len(ohlcv) < 10:  # Minimum viable data points
                logger.warning(f"Insufficient data points ({len(ohlcv)}) for {symbol} {timeframe}")
                return None
            
            # Validate data structure
            for i, candle in enumerate(ohlcv[:3]):  # Check first 3 candles
                if len(candle) != 6:
                    raise ValueError(f"Invalid OHLCV data structure at index {i}: {candle}")
                if not all(isinstance(x, (int, float)) or x is None for x in candle[1:6]):
                    raise ValueError(f"Invalid OHLCV data types at index {i}: {candle}")
            
            # Convert to DataFrame with data validation
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Validate price data
            df = self._validate_and_clean_data(df, symbol, timeframe)
            if df is None or len(df) < 5:
                logger.warning(f"Data validation failed for {symbol} {timeframe}")
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Calculate technical indicators with error handling
            logger.info(f"DEBUG: About to calculate indicators for {symbol} {timeframe}, df shape: {df.shape}")
            indicators = self._calculate_enhanced_indicators(df, timeframe)
            logger.info(f"DEBUG: Indicators result for {symbol} {timeframe}: {type(indicators)}, keys: {list(indicators.keys()) if isinstance(indicators, dict) else 'NOT_DICT'}")
            
            # Validate indicators
            if not indicators or len(indicators) == 0:
                logger.warning(f"No indicators calculated for {symbol} {timeframe}")
                indicators = {'error': 'Failed to calculate indicators'}
            
            # Build comprehensive result with metadata
            result = {
                'ohlcv': df.to_dict('records'),
                'indicators': indicators,
                'data_points': len(df),
                'date_range': {
                    'start': df.index[0].isoformat(),
                    'end': df.index[-1].isoformat()
                },
                'data_quality': {
                    'completeness': len(df) / config['periods'] * 100,
                    'price_range': {
                        'high': float(df['high'].max()),
                        'low': float(df['low'].min()),
                        'spread': float(df['high'].max() - df['low'].min())
                    },
                    'volume_stats': {
                        'avg': float(df['volume'].mean()),
                        'max': float(df['volume'].max()),
                        'min': float(df['volume'].min())
                    }
                },
                'status': 'success'
            }
            
            return result
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {timeframe} data for {symbol}: {e}")
            raise  # Re-raise for retry mechanism
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {timeframe} data for {symbol}: {e}")
            raise  # Re-raise for retry mechanism
        except ValueError as e:
            logger.error(f"Data validation error for {symbol} {timeframe}: {e}")
            return None  # Don't retry validation errors
        except Exception as e:
            logger.error(f"Unexpected error fetching {timeframe} data for {symbol}: {e}")
            logger.debug(f"Full traceback:", exc_info=True)
            return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Comprehensive data validation and cleaning
        """
        try:
            original_length = len(df)
            
            # Remove rows with null or invalid data
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            # Validate price relationships (high >= low, etc.)
            invalid_price_mask = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | (df['low'] > df['open']) | (df['low'] > df['close'])
            if invalid_price_mask.any():
                logger.warning(f"Found {invalid_price_mask.sum()} invalid price relationships in {symbol} {timeframe}")
                df = df[~invalid_price_mask]
            
            # Remove obvious price anomalies (more than 50% price change in one candle)
            df['price_change'] = df['close'].pct_change().abs()
            anomaly_mask = df['price_change'] > 0.5
            if anomaly_mask.any():
                logger.warning(f"Found {anomaly_mask.sum()} price anomalies in {symbol} {timeframe}")
                df = df[~anomaly_mask]
            
            # Remove negative or zero prices/volumes
            negative_mask = (df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0) | (df['volume'] < 0)
            if negative_mask.any():
                logger.warning(f"Found {negative_mask.sum()} negative/zero values in {symbol} {timeframe}")
                df = df[~negative_mask]
            
            # Check data retention rate
            retention_rate = len(df) / original_length if original_length > 0 else 0
            if retention_rate < 0.8:
                logger.warning(f"Low data quality for {symbol} {timeframe}: {retention_rate:.1%} retention rate")
            
            df = df.drop('price_change', axis=1, errors='ignore')
            
            return df if len(df) >= 5 else None
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol} {timeframe}: {e}")
            return None
    
    def _analyze_timeframes_parallel(self, symbol: str) -> Dict:
        """
        Parallel timeframe analysis with priority-based execution
        """
        timeframe_data = {}
        
        # Sort timeframes by priority (1=highest, 3=lowest)
        sorted_timeframes = sorted(
            self.timeframes.items(),
            key=lambda x: x[1]['priority']
        )
        
        def analyze_single_timeframe(tf_config_pair):
            """Worker function for parallel execution"""
            tf, config = tf_config_pair
            try:
                logger.info(f"DEBUG-WORKER: Worker analyzing {tf} for {symbol}")
                result = self._fetch_and_analyze_timeframe(symbol, tf, config)
                
                if result and result.get('status') == 'success':
                    quality = result.get('data_quality', {})
                    completeness = quality.get('completeness', 0)
                    data_points = result.get('data_points', 0)
                    
                    logger.info(f"✅ Parallel {tf} complete: {data_points} candles, {completeness:.1f}% complete")
                    
                    # Log quality warnings
                    if completeness < 90:
                        logger.warning(f"⚠️  Low completeness in parallel {tf}: {completeness:.1f}%")
                    expected_periods = config['periods']
                    if data_points < expected_periods * 0.8:
                        logger.warning(f"⚠️  Fewer data points in parallel {tf}: {data_points}/{expected_periods}")
                        
                else:
                    error_msg = 'Failed to fetch or analyze data'
                    if result and 'error' in result:
                        error_msg = result['error']
                    
                    logger.warning(f"❌ Parallel {tf} failed: {error_msg}")
                    result = {
                        'error': error_msg,
                        'status': 'failed',
                        'timeframe': tf,
                        'requested_periods': config['periods']
                    }
                
                return tf, result
                
            except Exception as e:
                error_msg = f"Critical error in parallel {tf}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return tf, {
                    'error': error_msg,
                    'status': 'error',
                    'timeframe': tf,
                    'exception_type': type(e).__name__
                }
        
        # Execute parallel analysis with ThreadPoolExecutor
        logger.info(f"DEBUG-PARALLEL: Starting ThreadPoolExecutor with {self.max_workers} workers for {len(sorted_timeframes)} timeframes")
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=f"TF-{symbol}") as executor:
            # Submit all timeframe analysis tasks
            logger.info(f"DEBUG-PARALLEL: Submitting tasks for timeframes: {[tf[0] for tf in sorted_timeframes]}")
            future_to_tf = {
                executor.submit(analyze_single_timeframe, tf_config): tf_config[0]
                for tf_config in sorted_timeframes
            }
            logger.info(f"DEBUG-PARALLEL: Submitted {len(future_to_tf)} tasks to executor")
            
            # Collect results as they complete
            logger.info(f"DEBUG-PARALLEL: Collecting results from {len(future_to_tf)} futures")
            for future in as_completed(future_to_tf):
                tf = future_to_tf[future]
                logger.info(f"DEBUG-PARALLEL: Processing future result for {tf}")
                try:
                    timeframe, result = future.result(timeout=60)  # 60-second timeout per timeframe
                    timeframe_data[timeframe] = result
                    logger.info(f"DEBUG-PARALLEL: Successfully completed parallel analysis for {timeframe}")
                    
                except Exception as e:
                    logger.error(f"Parallel execution failed for {tf}: {e}")
                    timeframe_data[tf] = {
                        'error': f'Parallel execution timeout or failure: {str(e)}',
                        'status': 'error',
                        'timeframe': tf
                    }
        
        return timeframe_data
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        total_requests = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
        cache_hit_rate = (self.performance_metrics['cache_hits'] / max(total_requests, 1)) * 100
        
        return {
            'cache_statistics': {
                'hit_rate': f"{cache_hit_rate:.1f}%",
                'total_hits': self.performance_metrics['cache_hits'],
                'total_misses': self.performance_metrics['cache_misses'],
                'api_calls_saved': self.performance_metrics['api_calls_saved']
            },
            'execution_statistics': {
                'parallel_executions': self.performance_metrics['parallel_executions'],
                'total_analysis_time': f"{self.performance_metrics['total_analysis_time']:.2f}s",
                'parallel_processing_enabled': self.parallel_processing,
                'max_workers': self.max_workers
            },
            'cache_status': {
                'cache_enabled': self._cache.config.enabled,
                'current_cache_size': len(self._cache.cache),
                'max_cache_size': self._cache.config.max_cache_size
            }
        }
    
    def clear_cache(self) -> None:
        """Clear performance cache and reset metrics"""
        self._cache.clear()
        self.performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_executions': 0,
            'total_analysis_time': 0,
            'api_calls_saved': 0
        }
        logger.info("Performance cache cleared and metrics reset")
    
    @cached_method(ttl=60)  # 1-minute cache for indicators
    # @safe_calculation(default_value={})  # Temporarily disabled to see actual errors
    def _calculate_enhanced_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Calculate enhanced technical indicators for a specific timeframe using configuration"""
        
        if df.empty or len(df) < self.config.analysis.min_data_points:
            return {'error': 'insufficient_data', 'periods': len(df)}
        
        indicators = {}
        indicator_config = self.config.indicators
        
        # Get enabled indicators for this timeframe
        enabled_indicators = self.timeframes[timeframe]['indicators_enabled'] if timeframe in self.timeframes else []
        
        try:
            # Moving Averages (always calculate for trend analysis)
            indicators['sma_short'] = ta.trend.sma_indicator(df['close'], window=indicator_config.sma_short).iloc[-1]
            indicators['sma_long'] = ta.trend.sma_indicator(df['close'], window=indicator_config.sma_long).iloc[-1] if len(df) >= indicator_config.sma_long else None
            
            # Add SMA 50 and SMA 200 for trend analysis
            indicators['sma_50'] = ta.trend.sma_indicator(df['close'], window=50).iloc[-1] if len(df) >= 50 else None
            indicators['sma_200'] = ta.trend.sma_indicator(df['close'], window=200).iloc[-1] if len(df) >= 200 else None
            
            indicators['ema'] = ta.trend.ema_indicator(df['close'], window=indicator_config.ema_period).iloc[-1]
            
            # MACD (if enabled)
            if 'macd' in enabled_indicators:
                macd_line = ta.trend.macd(df['close'], window_slow=indicator_config.macd_slow, window_fast=indicator_config.macd_fast)
                macd_signal = ta.trend.macd_signal(df['close'], window_slow=indicator_config.macd_slow, 
                                                 window_fast=indicator_config.macd_fast, window_sign=indicator_config.macd_signal)
                macd_histogram = ta.trend.macd_diff(df['close'], window_slow=indicator_config.macd_slow, window_fast=indicator_config.macd_fast)
                
                indicators['macd'] = macd_line.iloc[-1]
                indicators['macd_signal'] = macd_signal.iloc[-1]
                indicators['macd_histogram'] = macd_histogram.iloc[-1]
            
            # RSI (if enabled)
            if 'rsi' in enabled_indicators:
                indicators['rsi'] = ta.momentum.rsi(df['close'], window=indicator_config.rsi_period).iloc[-1]
                indicators['rsi_overbought'] = indicators['rsi'] > indicator_config.rsi_overbought
                indicators['rsi_oversold'] = indicators['rsi'] < indicator_config.rsi_oversold
            
            # Stochastic (if enabled)
            if 'stochastic' in enabled_indicators:
                if len(df) >= max(indicator_config.stoch_k_period, indicator_config.stoch_d_period):
                    try:
                        stoch_k = ta.momentum.stoch(df['high'], df['low'], df['close'], 
                                                  window=indicator_config.stoch_k_period, smooth_window=indicator_config.stoch_d_period)
                        stoch_d = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 
                                                         window=indicator_config.stoch_k_period, smooth_window=indicator_config.stoch_d_period)
                        
                        indicators['stoch'] = stoch_k.iloc[-1] if not stoch_k.empty and not pd.isna(stoch_k.iloc[-1]) else 50
                        indicators['stoch_signal'] = stoch_d.iloc[-1] if not stoch_d.empty and not pd.isna(stoch_d.iloc[-1]) else 50
                        indicators['stoch_overbought'] = indicators['stoch'] > indicator_config.stoch_overbought
                        indicators['stoch_oversold'] = indicators['stoch'] < indicator_config.stoch_oversold
                    except Exception as e:
                        logger.warning(f"Stochastic calculation failed for {timeframe}: {e}")
                        indicators['stoch'] = 50
                        indicators['stoch_signal'] = 50
                        indicators['stoch_overbought'] = False
                        indicators['stoch_oversold'] = False
                else:
                    indicators['stoch'] = 50
                    indicators['stoch_signal'] = 50
                    indicators['stoch_overbought'] = False
                    indicators['stoch_oversold'] = False
            
            # Bollinger Bands (if enabled)
            if 'bollinger_bands' in enabled_indicators:
                bb_analysis = self._calculate_bollinger_analysis(df)
                indicators.update(bb_analysis)
            
            # ATR (if enabled) 
            if 'atr' in enabled_indicators:
                indicators['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 
                                                                   window=indicator_config.atr_period).iloc[-1]
                indicators['atr_stop_loss'] = df['close'].iloc[-1] - (indicators['atr'] * indicator_config.atr_multiplier)
                indicators['atr_take_profit'] = df['close'].iloc[-1] + (indicators['atr'] * indicator_config.atr_multiplier * self.config.analysis.take_profit_ratio)
                indicators['atr_percent'] = (indicators['atr'] / df['close'].iloc[-1]) * 100
            else:
                # Default ATR calculation for volatility assessment
                indicators['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).iloc[-1]
                indicators['atr_percent'] = (indicators['atr'] / df['close'].iloc[-1]) * 100
            
            # ADX (Average Directional Movement Index) - if enabled
            if 'adx' in enabled_indicators:
                if len(df) >= 14:  # ADX needs at least 14 periods
                    try:
                        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
                        indicators['adx'] = adx_indicator.adx().iloc[-1]
                        indicators['adx_pos'] = adx_indicator.adx_pos().iloc[-1] 
                        indicators['adx_neg'] = adx_indicator.adx_neg().iloc[-1]
                    except Exception as e:
                        logger.warning(f"ADX calculation failed for {timeframe}: {e}")
                        indicators['adx'] = 0
                        indicators['adx_pos'] = 0
                        indicators['adx_neg'] = 0
                else:
                    indicators['adx'] = 0
                    indicators['adx_pos'] = 0
                    indicators['adx_neg'] = 0
            
            # Williams %R - if enabled
            if 'williams_r' in enabled_indicators:
                if len(df) >= 14:
                    try:
                        williams_r = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
                        indicators['williams_r'] = williams_r.iloc[-1] if not williams_r.empty else -50
                        indicators['williams_r_overbought'] = indicators['williams_r'] > -20
                        indicators['williams_r_oversold'] = indicators['williams_r'] < -80
                    except Exception as e:
                        logger.warning(f"Williams %R calculation failed for {timeframe}: {e}")
                        indicators['williams_r'] = -50
                        indicators['williams_r_overbought'] = False
                        indicators['williams_r_oversold'] = False
                else:
                    indicators['williams_r'] = -50
                    indicators['williams_r_overbought'] = False
                    indicators['williams_r_oversold'] = False
            
            # CCI (Commodity Channel Index) - if enabled
            if 'cci' in enabled_indicators:
                if len(df) >= 20:
                    try:
                        cci = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
                        indicators['cci'] = cci.iloc[-1] if not cci.empty else 0
                        indicators['cci_overbought'] = indicators['cci'] > 100
                        indicators['cci_oversold'] = indicators['cci'] < -100
                        indicators['cci_extreme_overbought'] = indicators['cci'] > 200
                        indicators['cci_extreme_oversold'] = indicators['cci'] < -200
                    except Exception as e:
                        logger.warning(f"CCI calculation failed for {timeframe}: {e}")
                        indicators['cci'] = 0
                        indicators['cci_overbought'] = False
                        indicators['cci_oversold'] = False
                        indicators['cci_extreme_overbought'] = False
                        indicators['cci_extreme_oversold'] = False
                else:
                    indicators['cci'] = 0
                    indicators['cci_overbought'] = False
                    indicators['cci_oversold'] = False
                    indicators['cci_extreme_overbought'] = False
                    indicators['cci_extreme_oversold'] = False
            
            # Parabolic SAR - if enabled  
            if 'parabolic_sar' in enabled_indicators:
                if len(df) >= 10:
                    try:
                        psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
                        indicators['psar'] = psar.psar().iloc[-1] if not psar.psar().empty else df['close'].iloc[-1]
                        indicators['psar_up'] = psar.psar_up().iloc[-1] if not psar.psar_up().empty else None
                        indicators['psar_down'] = psar.psar_down().iloc[-1] if not psar.psar_down().empty else None
                        indicators['psar_signal'] = 'BULLISH' if indicators['psar'] < df['close'].iloc[-1] else 'BEARISH'
                    except Exception as e:
                        logger.warning(f"Parabolic SAR calculation failed for {timeframe}: {e}")
                        indicators['psar'] = df['close'].iloc[-1]
                        indicators['psar_up'] = None
                        indicators['psar_down'] = None
                        indicators['psar_signal'] = 'NEUTRAL'
                else:
                    indicators['psar'] = df['close'].iloc[-1]
                    indicators['psar_up'] = None
                    indicators['psar_down'] = None
                    indicators['psar_signal'] = 'NEUTRAL'
            
            # Ichimoku Cloud - if enabled (complex but very valuable)
            if 'ichimoku' in enabled_indicators:
                if len(df) >= 52:  # Ichimoku needs longer periods
                    try:
                        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
                        indicators['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line().iloc[-1]
                        indicators['ichimoku_base'] = ichimoku.ichimoku_base_line().iloc[-1] 
                        indicators['ichimoku_span_a'] = ichimoku.ichimoku_a().iloc[-1]
                        indicators['ichimoku_span_b'] = ichimoku.ichimoku_b().iloc[-1]
                        
                        current_price = df['close'].iloc[-1]
                        # Determine cloud position
                        if current_price > max(indicators['ichimoku_span_a'], indicators['ichimoku_span_b']):
                            indicators['ichimoku_position'] = 'ABOVE_CLOUD'
                        elif current_price < min(indicators['ichimoku_span_a'], indicators['ichimoku_span_b']):
                            indicators['ichimoku_position'] = 'BELOW_CLOUD'
                        else:
                            indicators['ichimoku_position'] = 'IN_CLOUD'
                        
                        # Signal generation
                        if indicators['ichimoku_conversion'] > indicators['ichimoku_base']:
                            indicators['ichimoku_signal'] = 'BULLISH' if indicators['ichimoku_position'] == 'ABOVE_CLOUD' else 'WEAK_BULLISH'
                        else:
                            indicators['ichimoku_signal'] = 'BEARISH' if indicators['ichimoku_position'] == 'BELOW_CLOUD' else 'WEAK_BEARISH'
                            
                    except Exception as e:
                        logger.warning(f"Ichimoku calculation failed for {timeframe}: {e}")
                        indicators['ichimoku_conversion'] = df['close'].iloc[-1]
                        indicators['ichimoku_base'] = df['close'].iloc[-1]
                        indicators['ichimoku_span_a'] = df['close'].iloc[-1]
                        indicators['ichimoku_span_b'] = df['close'].iloc[-1]
                        indicators['ichimoku_position'] = 'NEUTRAL'
                        indicators['ichimoku_signal'] = 'NEUTRAL'
                else:
                    # Not enough data for Ichimoku
                    indicators['ichimoku_conversion'] = df['close'].iloc[-1]
                    indicators['ichimoku_base'] = df['close'].iloc[-1]
                    indicators['ichimoku_span_a'] = df['close'].iloc[-1]
                    indicators['ichimoku_span_b'] = df['close'].iloc[-1]
                    indicators['ichimoku_position'] = 'INSUFFICIENT_DATA'
                    indicators['ichimoku_signal'] = 'NEUTRAL'
            
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
            
            # Enhanced Volume Profile Analysis  
            volume_profile_data = self._calculate_volume_profile_analysis(df)
            indicators.update(volume_profile_data)
            
            # Market Structure Analysis
            market_structure_data = self._calculate_market_structure_analysis(df)
            indicators.update(market_structure_data)
            
            # Momentum Divergence Detection
            divergence_data = self._detect_momentum_divergences(df, indicators)
            indicators.update(divergence_data)
            
            # Price Action
            current_price = df['close'].iloc[-1]
            indicators['price'] = current_price
            
            # Support/Resistance levels
            if len(df) >= 20:
                recent_high = df['high'].tail(20).max()
                recent_low = df['low'].tail(20).min()
                indicators['resistance_level'] = recent_high
                indicators['support_level'] = recent_low
                
                # Fixed position calculation to handle outside range properly
                if recent_high != recent_low:
                    position_ratio = (current_price - recent_low) / (recent_high - recent_low)
                    if position_ratio > 1.0:  # Above range
                        indicators['position_in_range'] = 1.0  # Cap at 100%
                        indicators['range_status'] = f'ABOVE RANGE by {((current_price - recent_high) / recent_high * 100):.1f}%'
                    elif position_ratio < 0.0:  # Below range  
                        indicators['position_in_range'] = 0.0  # Cap at 0%
                        indicators['range_status'] = f'BELOW RANGE by {((recent_low - current_price) / current_price * 100):.1f}%'
                    else:  # Within range
                        indicators['position_in_range'] = position_ratio
                        if position_ratio > 0.7:
                            indicators['range_status'] = 'UPPER'
                        elif position_ratio < 0.3:
                            indicators['range_status'] = 'LOWER'
                        else:
                            indicators['range_status'] = 'MIDDLE'
                else:
                    indicators['position_in_range'] = 0.5
                    indicators['range_status'] = 'NEUTRAL'
            
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
    
    def _calculate_bollinger_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Enhanced Bollinger Band analysis with squeeze detection and position analysis using configuration
        """
        try:
            bb_data = {}
            
            # Bollinger Bands with configured parameters
            bb = ta.volatility.BollingerBands(df['close'], 
                                            window=self.config.indicators.bb_period, 
                                            window_dev=self.config.indicators.bb_std)
            bb_upper = bb.bollinger_hband()
            bb_middle = bb.bollinger_mavg()
            bb_lower = bb.bollinger_lband()
            
            current_price = df['close'].iloc[-1]
            
            # Current BB values
            bb_data['bb_upper'] = bb_upper.iloc[-1]
            bb_data['bb_middle'] = bb_middle.iloc[-1] 
            bb_data['bb_lower'] = bb_lower.iloc[-1]
            
            # BB Width Analysis - Key squeeze indicator
            bb_width = (bb_upper - bb_lower) / bb_middle
            bb_data['bb_width'] = (bb_width.iloc[-1]) * 100  # As percentage
            bb_data['bb_width_sma'] = bb_width.rolling(window=20).mean().iloc[-1] * 100
            
            # Historical percentile for squeeze detection
            bb_width_20 = bb_width.tail(20)
            bb_width_50 = bb_width.tail(50)
            bb_width_100 = bb_width.tail(100) if len(bb_width) >= 100 else bb_width
            
            current_width = bb_width.iloc[-1]
            bb_data['bb_width_percentile_20'] = (bb_width_20 < current_width).sum() / len(bb_width_20) * 100
            bb_data['bb_width_percentile_50'] = (bb_width_50 < current_width).sum() / len(bb_width_50) * 100
            bb_data['bb_width_percentile_100'] = (bb_width_100 < current_width).sum() / len(bb_width_100) * 100
            
            # Squeeze Detection (Multiple Methods)
            bb_data['squeeze_detected'] = False
            bb_data['squeeze_intensity'] = 'NORMAL'
            
            # Method 1: Width below 20-period average
            if current_width < bb_width.rolling(window=20).mean().iloc[-1]:
                bb_data['squeeze_detected'] = True
                
            # Method 2: Width in bottom 20th percentile of last 50 periods
            if bb_data['bb_width_percentile_50'] < 20:
                bb_data['squeeze_detected'] = True
                bb_data['squeeze_intensity'] = 'MODERATE'
                
            # Method 3: Extreme squeeze - bottom 10th percentile
            if bb_data['bb_width_percentile_50'] < 10:
                bb_data['squeeze_intensity'] = 'EXTREME'
                
            # Position Analysis
            if bb_data['bb_upper'] > bb_data['bb_lower']:  # Avoid division by zero
                bb_position = (current_price - bb_data['bb_lower']) / (bb_data['bb_upper'] - bb_data['bb_lower'])
                bb_data['bb_position'] = bb_position
                
                # Position classification
                if bb_position > 0.85:
                    bb_data['bb_zone'] = 'UPPER_EXTREME'
                    bb_data['bb_signal'] = 'POTENTIAL_RESISTANCE'
                elif bb_position > 0.7:
                    bb_data['bb_zone'] = 'UPPER'
                    bb_data['bb_signal'] = 'BULLISH'
                elif bb_position > 0.3:
                    bb_data['bb_zone'] = 'MIDDLE'
                    bb_data['bb_signal'] = 'NEUTRAL'
                elif bb_position > 0.15:
                    bb_data['bb_zone'] = 'LOWER'
                    bb_data['bb_signal'] = 'BEARISH'
                else:
                    bb_data['bb_zone'] = 'LOWER_EXTREME'
                    bb_data['bb_signal'] = 'POTENTIAL_SUPPORT'
            else:
                bb_data['bb_position'] = 0.5
                bb_data['bb_zone'] = 'ERROR'
                bb_data['bb_signal'] = 'UNKNOWN'
            
            # Band Distance Analysis
            bb_data['distance_to_upper'] = ((bb_data['bb_upper'] - current_price) / current_price) * 100
            bb_data['distance_to_lower'] = ((current_price - bb_data['bb_lower']) / current_price) * 100
            bb_data['distance_to_middle'] = ((current_price - bb_data['bb_middle']) / current_price) * 100
            
            # Band Slope Analysis (Trend Strength)
            if len(bb_upper) >= 5:
                upper_slope = (bb_upper.iloc[-1] - bb_upper.iloc[-5]) / bb_upper.iloc[-5] * 100
                middle_slope = (bb_middle.iloc[-1] - bb_middle.iloc[-5]) / bb_middle.iloc[-5] * 100
                lower_slope = (bb_lower.iloc[-1] - bb_lower.iloc[-5]) / bb_lower.iloc[-5] * 100
                
                bb_data['bb_upper_slope'] = upper_slope
                bb_data['bb_middle_slope'] = middle_slope
                bb_data['bb_lower_slope'] = lower_slope
                
                # Band expansion/contraction
                if upper_slope > 0.5 and lower_slope < -0.5:
                    bb_data['bb_expansion'] = True
                    bb_data['bb_trend_strength'] = 'EXPANDING'
                elif abs(upper_slope) < 0.2 and abs(lower_slope) < 0.2:
                    bb_data['bb_expansion'] = False
                    bb_data['bb_trend_strength'] = 'CONTRACTING'
                else:
                    bb_data['bb_expansion'] = None
                    bb_data['bb_trend_strength'] = 'MIXED'
            
            # Multi-timeframe squeeze comparison (if we have enough data)
            if len(df) >= 100:
                # Compare current squeeze to historical patterns
                bb_width_series = bb_width.dropna()
                if len(bb_width_series) >= 50:
                    squeeze_events = bb_width_series < bb_width_series.rolling(window=20).mean()
                    recent_squeeze_count = squeeze_events.tail(50).sum()
                    bb_data['recent_squeeze_frequency'] = recent_squeeze_count / 50 * 100
                    
                    # Average duration of squeezes
                    squeeze_lengths = []
                    current_squeeze_length = 0
                    for is_squeeze in squeeze_events.tail(50):
                        if is_squeeze:
                            current_squeeze_length += 1
                        else:
                            if current_squeeze_length > 0:
                                squeeze_lengths.append(current_squeeze_length)
                                current_squeeze_length = 0
                    
                    if squeeze_lengths:
                        bb_data['avg_squeeze_duration'] = np.mean(squeeze_lengths)
                    else:
                        bb_data['avg_squeeze_duration'] = 0
                        
            # Generate comprehensive BB summary
            summary_parts = []
            summary_parts.append(f"Position: {bb_data['bb_zone']} ({bb_data['bb_position']:.2f})")
            summary_parts.append(f"Width: {bb_data['bb_width']:.2f}% (Percentile: {bb_data['bb_width_percentile_50']:.0f}th)")
            
            if bb_data['squeeze_detected']:
                summary_parts.append(f"SQUEEZE: {bb_data['squeeze_intensity']}")
            
            if bb_data.get('bb_trend_strength'):
                summary_parts.append(f"Trend: {bb_data['bb_trend_strength']}")
                
            bb_data['bb_summary'] = " | ".join(summary_parts)
            
            return bb_data
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Band analysis: {e}")
            return {
                'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0, 'bb_width': 0,
                'squeeze_detected': False, 'bb_zone': 'ERROR', 'bb_signal': 'UNKNOWN',
                'bb_summary': f'ERROR: {str(e)}'
            }
    
    def _calculate_volume_profile_analysis(self, df: pd.DataFrame, price_bins: int = 50) -> Dict:
        """
        Enhanced Volume Profile Analysis with POC, Value Area, and distribution metrics
        """
        try:
            vp_data = {}
            current_price = df['close'].iloc[-1]
            
            # Prepare data for volume profile calculation
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Create price bins for volume distribution
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_range = price_max - price_min
            
            if price_range == 0:
                return {'vp_error': 'No price range available'}
            
            # Create price bins
            bin_size = price_range / price_bins
            price_levels = np.linspace(price_min, price_max, price_bins + 1)
            volume_at_price = np.zeros(price_bins)
            
            # Calculate volume at each price level
            for i in range(len(df)):
                # Find which bin this candle's typical price falls into
                tp = typical_price.iloc[i]
                bin_index = min(int((tp - price_min) / bin_size), price_bins - 1)
                volume_at_price[bin_index] += df['volume'].iloc[i]
            
            # Calculate Point of Control (POC)
            poc_index = np.argmax(volume_at_price)
            poc_price = (price_levels[poc_index] + price_levels[poc_index + 1]) / 2
            poc_volume = volume_at_price[poc_index]
            
            vp_data['poc_price'] = poc_price
            vp_data['poc_volume'] = poc_volume
            vp_data['poc_distance'] = ((current_price - poc_price) / current_price) * 100
            
            # Value Area calculation (70% of volume)
            total_volume = np.sum(volume_at_price)
            value_area_volume = total_volume * 0.70
            
            # Sort by volume to find value area
            volume_indices = np.argsort(volume_at_price)[::-1]  # Descending order
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in volume_indices:
                value_area_indices.append(idx)
                cumulative_volume += volume_at_price[idx]
                if cumulative_volume >= value_area_volume:
                    break
            
            # Calculate Value Area High and Low
            va_high = price_levels[max(value_area_indices) + 1]  # Upper bound of highest bin
            va_low = price_levels[min(value_area_indices)]       # Lower bound of lowest bin
            
            vp_data['va_high'] = va_high
            vp_data['va_low'] = va_low
            vp_data['va_width'] = ((va_high - va_low) / va_low) * 100
            
            # Current price position relative to Value Area
            if current_price > va_high:
                vp_data['va_position'] = 'ABOVE_VALUE_AREA'
                vp_data['va_signal'] = 'POTENTIAL_RESISTANCE'
            elif current_price < va_low:
                vp_data['va_position'] = 'BELOW_VALUE_AREA'
                vp_data['va_signal'] = 'POTENTIAL_SUPPORT'
            else:
                vp_data['va_position'] = 'WITHIN_VALUE_AREA'
                vp_data['va_signal'] = 'FAIR_VALUE'
                
            # Distance to value area boundaries
            vp_data['distance_to_va_high'] = ((va_high - current_price) / current_price) * 100
            vp_data['distance_to_va_low'] = ((current_price - va_low) / current_price) * 100
            
            # High Volume Nodes (HVN) - prices with significantly higher volume
            volume_mean = np.mean(volume_at_price)
            volume_std = np.std(volume_at_price)
            hvn_threshold = volume_mean + volume_std
            
            hvn_levels = []
            for i, volume in enumerate(volume_at_price):
                if volume > hvn_threshold:
                    hvn_price = (price_levels[i] + price_levels[i + 1]) / 2
                    hvn_levels.append({
                        'price': hvn_price,
                        'volume': volume,
                        'distance': ((current_price - hvn_price) / current_price) * 100
                    })
            
            vp_data['hvn_count'] = len(hvn_levels)
            vp_data['hvn_levels'] = hvn_levels
            
            # Low Volume Nodes (LVN) - potential breakout levels
            lvn_threshold = volume_mean - (volume_std * 0.5)
            lvn_levels = []
            
            for i, volume in enumerate(volume_at_price):
                if volume < lvn_threshold and volume > 0:
                    lvn_price = (price_levels[i] + price_levels[i + 1]) / 2
                    lvn_levels.append({
                        'price': lvn_price,
                        'volume': volume,
                        'distance': ((current_price - lvn_price) / current_price) * 100
                    })
            
            vp_data['lvn_count'] = len(lvn_levels)
            vp_data['lvn_levels'] = lvn_levels
            
            # Volume concentration analysis
            volume_concentration = (poc_volume / total_volume) * 100
            vp_data['volume_concentration'] = volume_concentration
            
            if volume_concentration > 15:
                vp_data['volume_distribution'] = 'HIGHLY_CONCENTRATED'
                vp_data['distribution_signal'] = 'STRONG_SUPPORT_RESISTANCE'
            elif volume_concentration > 10:
                vp_data['volume_distribution'] = 'MODERATELY_CONCENTRATED'
                vp_data['distribution_signal'] = 'MODERATE_SUPPORT_RESISTANCE'
            else:
                vp_data['volume_distribution'] = 'DISTRIBUTED'
                vp_data['distribution_signal'] = 'WEAK_SUPPORT_RESISTANCE'
            
            # Price acceptance analysis
            time_at_poc = 0
            poc_range = bin_size  # Price range around POC
            
            for price in df['close']:
                if abs(price - poc_price) <= poc_range:
                    time_at_poc += 1
            
            acceptance_ratio = (time_at_poc / len(df)) * 100
            vp_data['poc_acceptance'] = acceptance_ratio
            
            if acceptance_ratio > 30:
                vp_data['price_acceptance'] = 'HIGH'
            elif acceptance_ratio > 15:
                vp_data['price_acceptance'] = 'MODERATE'
            else:
                vp_data['price_acceptance'] = 'LOW'
            
            # Balance/Imbalance analysis
            upper_volume = np.sum(volume_at_price[poc_index:])
            lower_volume = np.sum(volume_at_price[:poc_index])
            
            if upper_volume > 0 and lower_volume > 0:
                volume_balance = upper_volume / lower_volume
                vp_data['volume_balance'] = volume_balance
                
                if 0.8 <= volume_balance <= 1.2:
                    vp_data['market_balance'] = 'BALANCED'
                elif volume_balance > 1.2:
                    vp_data['market_balance'] = 'UPPER_HEAVY'
                else:
                    vp_data['market_balance'] = 'LOWER_HEAVY'
            else:
                vp_data['volume_balance'] = 1.0
                vp_data['market_balance'] = 'BALANCED'
            
            # Support/Resistance strength based on volume profile
            nearest_hvn = None
            min_distance = float('inf')
            
            for hvn in hvn_levels:
                distance = abs(hvn['distance'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_hvn = hvn
            
            if nearest_hvn and min_distance < 5:  # Within 5%
                vp_data['nearest_hvn_distance'] = min_distance
                vp_data['hvn_strength'] = 'STRONG' if nearest_hvn['volume'] > volume_mean * 2 else 'MODERATE'
            else:
                vp_data['nearest_hvn_distance'] = None
                vp_data['hvn_strength'] = 'NONE'
            
            # Generate comprehensive summary
            summary_parts = []
            summary_parts.append(f"POC: ${poc_price:.2f} ({vp_data['poc_distance']:+.1f}%)")
            summary_parts.append(f"VA: {vp_data['va_position']}")
            summary_parts.append(f"HVN: {vp_data['hvn_count']} levels")
            summary_parts.append(f"Distribution: {vp_data['volume_distribution']}")
            
            vp_data['vp_summary'] = " | ".join(summary_parts)
            
            return vp_data
            
        except Exception as e:
            logger.error(f"Error calculating volume profile analysis: {e}")
            return {
                'vp_error': str(e),
                'poc_price': 0, 'va_high': 0, 'va_low': 0,
                'va_position': 'UNKNOWN', 'volume_distribution': 'UNKNOWN',
                'vp_summary': f'ERROR: {str(e)}'
            }
    
    def _calculate_market_structure_analysis(self, df: pd.DataFrame, swing_length: int = 5) -> Dict:
        """
        Advanced Market Structure Analysis - Higher Highs/Lower Lows, Swing Points, Structure Breaks
        """
        try:
            ms_data = {}
            
            if len(df) < swing_length * 2:
                return {'ms_error': 'Insufficient data for market structure analysis'}
            
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            for i in range(swing_length, len(df) - swing_length):
                # Check for swing high (higher than surrounding candles)
                is_swing_high = True
                for j in range(i - swing_length, i + swing_length + 1):
                    if j != i and df['high'].iloc[j] >= df['high'].iloc[i]:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swing_highs.append({
                        'index': i,
                        'price': df['high'].iloc[i],
                        'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                    })
                
                # Check for swing low (lower than surrounding candles)
                is_swing_low = True
                for j in range(i - swing_length, i + swing_length + 1):
                    if j != i and df['low'].iloc[j] <= df['low'].iloc[i]:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swing_lows.append({
                        'index': i,
                        'price': df['low'].iloc[i],
                        'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                    })
            
            ms_data['swing_highs_count'] = len(swing_highs)
            ms_data['swing_lows_count'] = len(swing_lows)
            
            # Analyze trend structure (Higher Highs/Lower Lows)
            current_price = df['close'].iloc[-1]
            
            # Recent swing points for trend analysis
            recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
            recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
            
            # Higher Highs/Lower Highs analysis
            if len(recent_highs) >= 2:
                hh_count = 0
                lh_count = 0
                for i in range(1, len(recent_highs)):
                    if recent_highs[i]['price'] > recent_highs[i-1]['price']:
                        hh_count += 1
                    else:
                        lh_count += 1
                
                ms_data['higher_highs'] = hh_count
                ms_data['lower_highs'] = lh_count
                
                if hh_count > lh_count:
                    ms_data['high_structure'] = 'HIGHER_HIGHS'
                elif lh_count > hh_count:
                    ms_data['high_structure'] = 'LOWER_HIGHS'
                else:
                    ms_data['high_structure'] = 'MIXED_HIGHS'
            else:
                ms_data['higher_highs'] = 0
                ms_data['lower_highs'] = 0
                ms_data['high_structure'] = 'INSUFFICIENT_DATA'
            
            # Higher Lows/Lower Lows analysis
            if len(recent_lows) >= 2:
                hl_count = 0
                ll_count = 0
                for i in range(1, len(recent_lows)):
                    if recent_lows[i]['price'] > recent_lows[i-1]['price']:
                        hl_count += 1
                    else:
                        ll_count += 1
                
                ms_data['higher_lows'] = hl_count
                ms_data['lower_lows'] = ll_count
                
                if hl_count > ll_count:
                    ms_data['low_structure'] = 'HIGHER_LOWS'
                elif ll_count > hl_count:
                    ms_data['low_structure'] = 'LOWER_LOWS'
                else:
                    ms_data['low_structure'] = 'MIXED_LOWS'
            else:
                ms_data['higher_lows'] = 0
                ms_data['lower_lows'] = 0
                ms_data['low_structure'] = 'INSUFFICIENT_DATA'
            
            # Overall market structure trend
            if ms_data['high_structure'] == 'HIGHER_HIGHS' and ms_data['low_structure'] == 'HIGHER_LOWS':
                ms_data['market_structure'] = 'STRONG_UPTREND'
                ms_data['structure_signal'] = 'BULLISH'
            elif ms_data['high_structure'] == 'LOWER_HIGHS' and ms_data['low_structure'] == 'LOWER_LOWS':
                ms_data['market_structure'] = 'STRONG_DOWNTREND'
                ms_data['structure_signal'] = 'BEARISH'
            elif ms_data['high_structure'] == 'HIGHER_HIGHS' and ms_data['low_structure'] == 'LOWER_LOWS':
                ms_data['market_structure'] = 'EXPANDING_RANGE'
                ms_data['structure_signal'] = 'NEUTRAL_EXPANSION'
            elif ms_data['high_structure'] == 'LOWER_HIGHS' and ms_data['low_structure'] == 'HIGHER_LOWS':
                ms_data['market_structure'] = 'CONTRACTING_RANGE'
                ms_data['structure_signal'] = 'NEUTRAL_CONTRACTION'
            else:
                ms_data['market_structure'] = 'MIXED_STRUCTURE'
                ms_data['structure_signal'] = 'NEUTRAL'
            
            # Structure break analysis
            structure_breaks = []
            
            # Check for breaks of recent swing highs/lows
            if swing_highs and swing_lows:
                # Most recent swing high and low
                last_swing_high = max(swing_highs, key=lambda x: x['index'])
                last_swing_low = max(swing_lows, key=lambda x: x['index'])
                
                # Check if current price broke structure
                if current_price > last_swing_high['price']:
                    structure_breaks.append({
                        'type': 'BULLISH_BREAK',
                        'level': last_swing_high['price'],
                        'distance': ((current_price - last_swing_high['price']) / last_swing_high['price']) * 100
                    })
                
                if current_price < last_swing_low['price']:
                    structure_breaks.append({
                        'type': 'BEARISH_BREAK',
                        'level': last_swing_low['price'],
                        'distance': ((last_swing_low['price'] - current_price) / last_swing_low['price']) * 100
                    })
            
            ms_data['structure_breaks'] = structure_breaks
            ms_data['break_count'] = len(structure_breaks)
            
            # Key levels analysis
            if swing_highs:
                recent_swing_high = swing_highs[-1]['price']
                ms_data['nearest_swing_high'] = recent_swing_high
                ms_data['distance_to_swing_high'] = ((recent_swing_high - current_price) / current_price) * 100
            else:
                ms_data['nearest_swing_high'] = None
                ms_data['distance_to_swing_high'] = None
            
            if swing_lows:
                recent_swing_low = swing_lows[-1]['price']
                ms_data['nearest_swing_low'] = recent_swing_low
                ms_data['distance_to_swing_low'] = ((current_price - recent_swing_low) / current_price) * 100
            else:
                ms_data['nearest_swing_low'] = None
                ms_data['distance_to_swing_low'] = None
            
            # Support/Resistance strength based on swing points
            if swing_highs and swing_lows:
                # Count how many times price has tested recent swing levels
                resistance_tests = 0
                support_tests = 0
                
                for i in range(max(0, len(df) - 20), len(df)):  # Last 20 periods
                    price_high = df['high'].iloc[i]
                    price_low = df['low'].iloc[i]
                    
                    # Check resistance tests (within 1% of swing high)
                    if swing_highs and abs(price_high - recent_swing_high) / recent_swing_high < 0.01:
                        resistance_tests += 1
                    
                    # Check support tests (within 1% of swing low)
                    if swing_lows and abs(price_low - recent_swing_low) / recent_swing_low < 0.01:
                        support_tests += 1
                
                ms_data['resistance_tests'] = resistance_tests
                ms_data['support_tests'] = support_tests
                
                # Strength assessment
                if resistance_tests >= 3:
                    ms_data['resistance_strength'] = 'STRONG'
                elif resistance_tests >= 2:
                    ms_data['resistance_strength'] = 'MODERATE'
                else:
                    ms_data['resistance_strength'] = 'WEAK'
                
                if support_tests >= 3:
                    ms_data['support_strength'] = 'STRONG'
                elif support_tests >= 2:
                    ms_data['support_strength'] = 'MODERATE'
                else:
                    ms_data['support_strength'] = 'WEAK'
            else:
                ms_data['resistance_tests'] = 0
                ms_data['support_tests'] = 0
                ms_data['resistance_strength'] = 'UNKNOWN'
                ms_data['support_strength'] = 'UNKNOWN'
            
            # Trend change detection
            if len(df) >= 10:
                # Compare recent price action to structure
                recent_closes = df['close'].tail(5)
                price_momentum = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0] * 100
                
                ms_data['recent_momentum'] = price_momentum
                
                # Check for potential trend changes
                if ms_data['market_structure'] == 'STRONG_UPTREND' and price_momentum < -2:
                    ms_data['trend_change_signal'] = 'POTENTIAL_REVERSAL_DOWN'
                elif ms_data['market_structure'] == 'STRONG_DOWNTREND' and price_momentum > 2:
                    ms_data['trend_change_signal'] = 'POTENTIAL_REVERSAL_UP'
                elif ms_data['market_structure'] in ['CONTRACTING_RANGE', 'MIXED_STRUCTURE'] and abs(price_momentum) > 3:
                    ms_data['trend_change_signal'] = 'BREAKOUT_BUILDING'
                else:
                    ms_data['trend_change_signal'] = 'TREND_CONTINUATION'
            else:
                ms_data['recent_momentum'] = 0
                ms_data['trend_change_signal'] = 'INSUFFICIENT_DATA'
            
            # Generate comprehensive summary
            summary_parts = []
            summary_parts.append(f"Structure: {ms_data['market_structure']}")
            summary_parts.append(f"Signal: {ms_data['structure_signal']}")
            
            if ms_data['break_count'] > 0:
                break_type = structure_breaks[0]['type']
                summary_parts.append(f"Break: {break_type}")
            
            if ms_data['trend_change_signal'] != 'TREND_CONTINUATION':
                summary_parts.append(f"Alert: {ms_data['trend_change_signal']}")
            
            ms_data['ms_summary'] = " | ".join(summary_parts)
            
            return ms_data
            
        except Exception as e:
            logger.error(f"Error calculating market structure analysis: {e}")
            return {
                'ms_error': str(e),
                'market_structure': 'ERROR', 'structure_signal': 'UNKNOWN',
                'swing_highs_count': 0, 'swing_lows_count': 0,
                'ms_summary': f'ERROR: {str(e)}'
            }
    
    def _detect_momentum_divergences(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """
        Advanced Momentum Divergence Detection - RSI, MACD, Price Action Analysis
        """
        try:
            div_data = {}
            
            if len(df) < 20:
                return {'div_error': 'Insufficient data for divergence analysis'}
            
            # Get price and indicator data
            closes = df['close']
            highs = df['high']
            lows = df['low']
            
            # Calculate RSI and MACD for divergence analysis
            rsi = ta.momentum.RSIIndicator(closes).rsi()
            macd_indicator = ta.trend.MACD(closes)
            macd_line = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()
            macd_histogram = macd_indicator.macd_diff()
            
            # Find recent swing highs and lows (simpler method for divergence)
            lookback = min(10, len(df) // 4)  # Adaptive lookback
            
            swing_highs_idx = []
            swing_lows_idx = []
            
            for i in range(lookback, len(df) - lookback):
                # Swing high: higher than surrounding periods
                if (highs.iloc[i] == highs.iloc[i-lookback:i+lookback+1].max()):
                    swing_highs_idx.append(i)
                
                # Swing low: lower than surrounding periods  
                if (lows.iloc[i] == lows.iloc[i-lookback:i+lookback+1].min()):
                    swing_lows_idx.append(i)
            
            # Keep only recent swing points
            swing_highs_idx = swing_highs_idx[-4:] if len(swing_highs_idx) > 4 else swing_highs_idx
            swing_lows_idx = swing_lows_idx[-4:] if len(swing_lows_idx) > 4 else swing_lows_idx
            
            # Initialize divergence counters
            bullish_divergences = []
            bearish_divergences = []
            
            # RSI Divergence Analysis
            if len(swing_highs_idx) >= 2:
                for i in range(1, len(swing_highs_idx)):
                    idx1, idx2 = swing_highs_idx[i-1], swing_highs_idx[i]
                    
                    price1, price2 = highs.iloc[idx1], highs.iloc[idx2]
                    rsi1, rsi2 = rsi.iloc[idx1], rsi.iloc[idx2]
                    
                    # Bearish divergence: Higher high in price, lower high in RSI
                    if price2 > price1 and rsi2 < rsi1 and not pd.isna(rsi1) and not pd.isna(rsi2):
                        strength = abs(rsi1 - rsi2) / rsi1 * 100  # Divergence strength
                        if strength > 5:  # Significant divergence threshold
                            bearish_divergences.append({
                                'type': 'RSI_BEARISH',
                                'strength': strength,
                                'price_diff': ((price2 - price1) / price1) * 100,
                                'rsi_diff': rsi2 - rsi1,
                                'periods_ago': len(df) - idx2
                            })
            
            if len(swing_lows_idx) >= 2:
                for i in range(1, len(swing_lows_idx)):
                    idx1, idx2 = swing_lows_idx[i-1], swing_lows_idx[i]
                    
                    price1, price2 = lows.iloc[idx1], lows.iloc[idx2]
                    rsi1, rsi2 = rsi.iloc[idx1], rsi.iloc[idx2]
                    
                    # Bullish divergence: Lower low in price, higher low in RSI
                    if price2 < price1 and rsi2 > rsi1 and not pd.isna(rsi1) and not pd.isna(rsi2):
                        strength = abs(rsi2 - rsi1) / rsi1 * 100
                        if strength > 5:
                            bullish_divergences.append({
                                'type': 'RSI_BULLISH',
                                'strength': strength,
                                'price_diff': ((price1 - price2) / price1) * 100,
                                'rsi_diff': rsi2 - rsi1,
                                'periods_ago': len(df) - idx2
                            })
            
            # MACD Divergence Analysis
            if len(swing_highs_idx) >= 2:
                for i in range(1, len(swing_highs_idx)):
                    idx1, idx2 = swing_highs_idx[i-1], swing_highs_idx[i]
                    
                    price1, price2 = highs.iloc[idx1], highs.iloc[idx2]
                    macd1, macd2 = macd_line.iloc[idx1], macd_line.iloc[idx2]
                    
                    # Bearish MACD divergence
                    if (price2 > price1 and macd2 < macd1 and 
                        not pd.isna(macd1) and not pd.isna(macd2)):
                        strength = abs(macd1 - macd2) / abs(macd1) * 100 if macd1 != 0 else 0
                        if strength > 10:  # MACD divergence threshold
                            bearish_divergences.append({
                                'type': 'MACD_BEARISH',
                                'strength': strength,
                                'price_diff': ((price2 - price1) / price1) * 100,
                                'macd_diff': macd2 - macd1,
                                'periods_ago': len(df) - idx2
                            })
            
            if len(swing_lows_idx) >= 2:
                for i in range(1, len(swing_lows_idx)):
                    idx1, idx2 = swing_lows_idx[i-1], swing_lows_idx[i]
                    
                    price1, price2 = lows.iloc[idx1], lows.iloc[idx2]
                    macd1, macd2 = macd_line.iloc[idx1], macd_line.iloc[idx2]
                    
                    # Bullish MACD divergence
                    if (price2 < price1 and macd2 > macd1 and 
                        not pd.isna(macd1) and not pd.isna(macd2)):
                        strength = abs(macd2 - macd1) / abs(macd1) * 100 if macd1 != 0 else 0
                        if strength > 10:
                            bullish_divergences.append({
                                'type': 'MACD_BULLISH',
                                'strength': strength,
                                'price_diff': ((price1 - price2) / price1) * 100,
                                'macd_diff': macd2 - macd1,
                                'periods_ago': len(df) - idx2
                            })
            
            # Hidden Divergence Detection (Continuation patterns)
            hidden_bullish = []
            hidden_bearish = []
            
            # Hidden Bullish: Higher lows in price, lower lows in oscillator (uptrend continuation)
            if len(swing_lows_idx) >= 2:
                for i in range(1, len(swing_lows_idx)):
                    idx1, idx2 = swing_lows_idx[i-1], swing_lows_idx[i]
                    
                    price1, price2 = lows.iloc[idx1], lows.iloc[idx2]
                    rsi1, rsi2 = rsi.iloc[idx1], rsi.iloc[idx2]
                    
                    if (price2 > price1 and rsi2 < rsi1 and 
                        not pd.isna(rsi1) and not pd.isna(rsi2)):
                        hidden_bullish.append({
                            'type': 'HIDDEN_BULLISH_RSI',
                            'strength': abs(rsi1 - rsi2),
                            'periods_ago': len(df) - idx2
                        })
            
            # Hidden Bearish: Lower highs in price, higher highs in oscillator (downtrend continuation)
            if len(swing_highs_idx) >= 2:
                for i in range(1, len(swing_highs_idx)):
                    idx1, idx2 = swing_highs_idx[i-1], swing_highs_idx[i]
                    
                    price1, price2 = highs.iloc[idx1], highs.iloc[idx2]
                    rsi1, rsi2 = rsi.iloc[idx1], rsi.iloc[idx2]
                    
                    if (price2 < price1 and rsi2 > rsi1 and 
                        not pd.isna(rsi1) and not pd.isna(rsi2)):
                        hidden_bearish.append({
                            'type': 'HIDDEN_BEARISH_RSI',
                            'strength': abs(rsi2 - rsi1),
                            'periods_ago': len(df) - idx2
                        })
            
            # Compile results
            div_data['bullish_divergences'] = bullish_divergences
            div_data['bearish_divergences'] = bearish_divergences
            div_data['hidden_bullish'] = hidden_bullish
            div_data['hidden_bearish'] = hidden_bearish
            
            div_data['bullish_div_count'] = len(bullish_divergences)
            div_data['bearish_div_count'] = len(bearish_divergences)
            div_data['hidden_bullish_count'] = len(hidden_bullish)
            div_data['hidden_bearish_count'] = len(hidden_bearish)
            
            # Overall divergence signal
            total_bullish = len(bullish_divergences) + len(hidden_bullish)
            total_bearish = len(bearish_divergences) + len(hidden_bearish)
            
            if total_bullish > total_bearish and total_bullish > 0:
                div_data['divergence_signal'] = 'BULLISH'
                div_data['divergence_strength'] = 'STRONG' if total_bullish >= 2 else 'MODERATE'
            elif total_bearish > total_bullish and total_bearish > 0:
                div_data['divergence_signal'] = 'BEARISH'
                div_data['divergence_strength'] = 'STRONG' if total_bearish >= 2 else 'MODERATE'
            else:
                div_data['divergence_signal'] = 'NEUTRAL'
                div_data['divergence_strength'] = 'NONE'
            
            # Recent divergence analysis (last 10 periods)
            recent_bullish = [d for d in bullish_divergences if d['periods_ago'] <= 10]
            recent_bearish = [d for d in bearish_divergences if d['periods_ago'] <= 10]
            
            div_data['recent_bullish_count'] = len(recent_bullish)
            div_data['recent_bearish_count'] = len(recent_bearish)
            
            if recent_bullish or recent_bearish:
                div_data['recent_divergence'] = True
                if len(recent_bullish) > len(recent_bearish):
                    div_data['recent_divergence_type'] = 'BULLISH'
                else:
                    div_data['recent_divergence_type'] = 'BEARISH'
            else:
                div_data['recent_divergence'] = False
                div_data['recent_divergence_type'] = 'NONE'
            
            # Momentum confirmation analysis
            current_rsi = indicators.get('rsi', 50)
            current_macd_hist = indicators.get('macd_histogram', 0)
            
            momentum_signals = []
            
            if current_rsi > 70 and bearish_divergences:
                momentum_signals.append('RSI_OVERBOUGHT_WITH_BEARISH_DIV')
            elif current_rsi < 30 and bullish_divergences:
                momentum_signals.append('RSI_OVERSOLD_WITH_BULLISH_DIV')
            
            if current_macd_hist > 0 and bullish_divergences:
                momentum_signals.append('MACD_BULLISH_WITH_BULLISH_DIV')
            elif current_macd_hist < 0 and bearish_divergences:
                momentum_signals.append('MACD_BEARISH_WITH_BEARISH_DIV')
            
            div_data['momentum_confirmation'] = momentum_signals
            div_data['confirmation_count'] = len(momentum_signals)
            
            # Generate comprehensive summary
            summary_parts = []
            summary_parts.append(f"Signal: {div_data['divergence_signal']}")
            summary_parts.append(f"Strength: {div_data['divergence_strength']}")
            
            if div_data['recent_divergence']:
                summary_parts.append(f"Recent: {div_data['recent_divergence_type']}")
            
            if div_data['confirmation_count'] > 0:
                summary_parts.append(f"Confirmed: {div_data['confirmation_count']}x")
            
            div_data['divergence_summary'] = " | ".join(summary_parts)
            
            return div_data
            
        except Exception as e:
            logger.error(f"Error detecting momentum divergences: {e}")
            return {
                'div_error': str(e),
                'divergence_signal': 'ERROR', 'divergence_strength': 'UNKNOWN',
                'bullish_div_count': 0, 'bearish_div_count': 0,
                'divergence_summary': f'ERROR: {str(e)}'
            }
    
    def _analyze_timeframe_confluence(self, timeframe_data: Dict) -> Dict:
        """
        Advanced multi-timeframe confluence analysis with weighted scoring and signal correlation
        """
        
        confluence = {
            'trend_alignment': {},
            'momentum_confluence': {},
            'indicator_agreement': {},
            'support_resistance_confluence': {},
            'volume_confirmation': {},
            'divergence_analysis': {},
            'signal_strength': {},
            'overall_confluence': {},
            'trading_recommendations': {}
        }
        
        try:
            valid_timeframes = [tf for tf, data in timeframe_data.items() 
                              if 'indicators' in data and 'error' not in data['indicators']]
            
            if not valid_timeframes:
                confluence['error'] = 'No valid timeframe data'
                return confluence
            
            # Get timeframe weights for weighted scoring
            tf_weights = {tf: self.timeframes[tf]['weight'] for tf in valid_timeframes if tf in self.timeframes}
            
            # 1. ENHANCED TREND ALIGNMENT ANALYSIS
            confluence['trend_alignment'] = self._analyze_trend_confluence(timeframe_data, valid_timeframes, tf_weights)
            
            # 2. MOMENTUM CONFLUENCE WITH MULTIPLE INDICATORS
            confluence['momentum_confluence'] = self._analyze_momentum_confluence(timeframe_data, valid_timeframes, tf_weights)
            
            # 3. INDICATOR AGREEMENT MATRIX
            confluence['indicator_agreement'] = self._analyze_indicator_agreement(timeframe_data, valid_timeframes, tf_weights)
            
            # 4. SUPPORT/RESISTANCE CONFLUENCE ZONES
            confluence['support_resistance_confluence'] = self._analyze_sr_confluence(timeframe_data, valid_timeframes)
            
            # 5. VOLUME CONFIRMATION ANALYSIS
            confluence['volume_confirmation'] = self._analyze_volume_confluence(timeframe_data, valid_timeframes, tf_weights)
            
            # 6. DIVERGENCE ANALYSIS ACROSS TIMEFRAMES
            confluence['divergence_analysis'] = self._analyze_cross_timeframe_divergences(timeframe_data, valid_timeframes)
            
            # 7. SIGNAL STRENGTH CALCULATION
            confluence['signal_strength'] = self._calculate_signal_strength(confluence, tf_weights)
            
            # 8. OVERALL CONFLUENCE SCORING
            confluence['overall_confluence'] = self._calculate_overall_confluence(confluence, tf_weights)
            
            # 9. TRADING RECOMMENDATIONS
            confluence['trading_recommendations'] = self._generate_confluence_recommendations(confluence, timeframe_data)
            
            return confluence
            
        except Exception as e:
            logger.error(f"Error in confluence analysis: {e}")
            confluence['error'] = str(e)
            return confluence
    
    def _analyze_trend_confluence(self, timeframe_data: Dict, valid_timeframes: List[str], tf_weights: Dict) -> Dict:
        """Enhanced trend alignment analysis with multiple indicators"""
        
        trend_analysis = {
            'individual_trends': {},
            'weighted_trend_score': 0,
            'trend_strength': {},
            'alignment_percentage': 0,
            'dominant_trend': 'NEUTRAL',
            'trend_reliability': 0
        }
        
        trend_scores = {}
        trend_strengths = {}
        
        for tf in valid_timeframes:
            indicators = timeframe_data[tf]['indicators']
            weight = tf_weights[tf]
            
            # Multi-indicator trend analysis
            trend_signals = []
            
            # SMA trend
            sma_short = indicators.get('sma_short', 0)
            sma_long = indicators.get('sma_long', 0)
            if sma_short and sma_long:
                trend_signals.append(1 if sma_short > sma_long else -1)
            
            # EMA trend
            ema = indicators.get('ema', 0)
            current_price = indicators.get('price', 0) or indicators.get('close', 0)
            if ema and current_price:
                trend_signals.append(1 if current_price > ema else -1)
            
            # MACD trend
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd and macd_signal:
                trend_signals.append(1 if macd > macd_signal else -1)
            
            # Calculate trend score for this timeframe
            tf_trend_score = sum(trend_signals) / len(trend_signals) if trend_signals else 0
            trend_scores[tf] = tf_trend_score
            
            # Calculate trend strength
            adx = indicators.get('adx', 0)
            atr = indicators.get('atr', 0)
            trend_strength = min(adx / 50, 1.0) if adx else 0.5  # Normalize ADX
            trend_strengths[tf] = trend_strength
            
            trend_analysis['individual_trends'][tf] = {
                'trend_score': tf_trend_score,
                'trend_direction': 'BULLISH' if tf_trend_score > 0.3 else 'BEARISH' if tf_trend_score < -0.3 else 'NEUTRAL',
                'trend_strength': trend_strength,
                'signals_count': len(trend_signals),
                'weight': weight
            }
        
        # Calculate weighted trend score
        weighted_score = sum(trend_scores[tf] * tf_weights[tf] for tf in valid_timeframes)
        trend_analysis['weighted_trend_score'] = weighted_score
        
        # Calculate alignment percentage
        bullish_timeframes = sum(1 for score in trend_scores.values() if score > 0.2)
        bearish_timeframes = sum(1 for score in trend_scores.values() if score < -0.2)
        alignment_pct = max(bullish_timeframes, bearish_timeframes) / len(valid_timeframes) * 100
        trend_analysis['alignment_percentage'] = alignment_pct
        
        # Determine dominant trend
        if weighted_score > 0.3:
            trend_analysis['dominant_trend'] = 'BULLISH'
        elif weighted_score < -0.3:
            trend_analysis['dominant_trend'] = 'BEARISH'
        else:
            trend_analysis['dominant_trend'] = 'NEUTRAL'
        
        # Calculate reliability based on alignment and strength
        avg_strength = sum(trend_strengths.values()) / len(trend_strengths)
        trend_analysis['trend_reliability'] = (alignment_pct / 100) * avg_strength
        
        return trend_analysis
    
    def _analyze_momentum_confluence(self, timeframe_data: Dict, valid_timeframes: List[str], tf_weights: Dict) -> Dict:
        """Advanced momentum confluence analysis with multiple oscillators"""
        
        momentum_analysis = {
            'rsi_confluence': {},
            'stoch_confluence': {},
            'macd_confluence': {},
            'weighted_momentum_score': 0,
            'momentum_alignment': 0,
            'momentum_bias': 'NEUTRAL',
            'overbought_oversold_signals': {},
            'momentum_divergence_risk': 0
        }
        
        rsi_values = {}
        stoch_values = {}
        macd_histogram_values = {}
        
        for tf in valid_timeframes:
            indicators = timeframe_data[tf]['indicators']
            
            # RSI analysis
            rsi = indicators.get('rsi', 50)
            rsi_values[tf] = {
                'value': rsi,
                'signal': 'OVERBOUGHT' if rsi > self.config.indicators.rsi_overbought else 
                         'OVERSOLD' if rsi < self.config.indicators.rsi_oversold else 'NEUTRAL',
                'weight': tf_weights[tf]
            }
            
            # Stochastic analysis
            stoch = indicators.get('stoch', 50)
            stoch_values[tf] = {
                'value': stoch,
                'signal': 'OVERBOUGHT' if stoch > self.config.indicators.stoch_overbought else 
                         'OVERSOLD' if stoch < self.config.indicators.stoch_oversold else 'NEUTRAL',
                'weight': tf_weights[tf]
            }
            
            # MACD histogram for momentum direction
            macd_hist = indicators.get('macd_histogram', 0)
            macd_histogram_values[tf] = {
                'value': macd_hist,
                'signal': 'BULLISH' if macd_hist > 0 else 'BEARISH' if macd_hist < 0 else 'NEUTRAL',
                'weight': tf_weights[tf]
            }
        
        # Calculate weighted momentum scores
        weighted_rsi = sum(rsi_values[tf]['value'] * tf_weights[tf] for tf in valid_timeframes)
        weighted_stoch = sum(stoch_values[tf]['value'] * tf_weights[tf] for tf in valid_timeframes)
        
        momentum_analysis['rsi_confluence'] = {
            'individual_values': rsi_values,
            'weighted_average': weighted_rsi,
            'alignment_score': self._calculate_indicator_alignment([v['value'] for v in rsi_values.values()]),
            'consensus_signal': self._get_consensus_signal([v['signal'] for v in rsi_values.values()])
        }
        
        momentum_analysis['stoch_confluence'] = {
            'individual_values': stoch_values,
            'weighted_average': weighted_stoch,
            'alignment_score': self._calculate_indicator_alignment([v['value'] for v in stoch_values.values()]),
            'consensus_signal': self._get_consensus_signal([v['signal'] for v in stoch_values.values()])
        }
        
        momentum_analysis['macd_confluence'] = {
            'individual_values': macd_histogram_values,
            'consensus_signal': self._get_consensus_signal([v['signal'] for v in macd_histogram_values.values()]),
            'bullish_timeframes': sum(1 for v in macd_histogram_values.values() if v['signal'] == 'BULLISH'),
            'bearish_timeframes': sum(1 for v in macd_histogram_values.values() if v['signal'] == 'BEARISH')
        }
        
        # Overall momentum scoring
        momentum_score = (weighted_rsi - 50) / 50  # Normalize to -1 to 1
        momentum_analysis['weighted_momentum_score'] = momentum_score
        
        # Determine momentum bias
        if momentum_score > 0.2:
            momentum_analysis['momentum_bias'] = 'BULLISH'
        elif momentum_score < -0.2:
            momentum_analysis['momentum_bias'] = 'BEARISH'
        else:
            momentum_analysis['momentum_bias'] = 'NEUTRAL'
        
        # Calculate alignment
        all_values = [v['value'] for v in rsi_values.values()]
        momentum_analysis['momentum_alignment'] = self._calculate_indicator_alignment(all_values)
        
        return momentum_analysis
    
    def _calculate_indicator_alignment(self, values: List[float]) -> float:
        """Calculate how aligned indicator values are across timeframes"""
        if len(values) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower = more aligned)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 1.0 if std_val == 0 else 0.0
        
        cv = std_val / abs(mean_val)
        # Convert to alignment score (0 = no alignment, 1 = perfect alignment)
        alignment = max(0, 1 - (cv / 2))  # Normalize CV to 0-1 scale
        return alignment
    
    def _get_consensus_signal(self, signals: List[str]) -> str:
        """Get consensus signal from multiple timeframes"""
        signal_counts = {}
        for signal in signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        # Return the most common signal
        return max(signal_counts.items(), key=lambda x: x[1])[0] if signal_counts else 'NEUTRAL'
    
    def _analyze_indicator_agreement(self, timeframe_data: Dict, valid_timeframes: List[str], tf_weights: Dict) -> Dict:
        """Analyze agreement between different indicators"""
        
        agreement_analysis = {
            'indicator_matrix': {},
            'agreement_scores': {},
            'conflicting_signals': [],
            'consensus_strength': 0
        }
        
        # Define indicator signal mapping
        indicator_signals = {}
        
        for tf in valid_timeframes:
            indicators = timeframe_data[tf]['indicators']
            tf_signals = {}
            
            # RSI signal
            rsi = indicators.get('rsi', 50)
            if tf in self.timeframes and 'rsi' in self.timeframes[tf]['indicators_enabled']:
                if rsi > self.config.indicators.rsi_overbought:
                    tf_signals['rsi'] = 'SELL'
                elif rsi < self.config.indicators.rsi_oversold:
                    tf_signals['rsi'] = 'BUY'
                else:
                    tf_signals['rsi'] = 'NEUTRAL'
            
            # MACD signal
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if tf in self.timeframes and 'macd' in self.timeframes[tf]['indicators_enabled']:
                if macd > macd_signal:
                    tf_signals['macd'] = 'BUY'
                elif macd < macd_signal:
                    tf_signals['macd'] = 'SELL'
                else:
                    tf_signals['macd'] = 'NEUTRAL'
            
            # Bollinger Bands signal
            if tf in self.timeframes and 'bollinger_bands' in self.timeframes[tf]['indicators_enabled']:
                bb_position = indicators.get('bb_position', 'middle')
                if bb_position == 'upper_breach':
                    tf_signals['bollinger'] = 'SELL'
                elif bb_position == 'lower_breach':
                    tf_signals['bollinger'] = 'BUY'
                else:
                    tf_signals['bollinger'] = 'NEUTRAL'
            
            indicator_signals[tf] = tf_signals
        
        # Calculate agreement matrix
        for tf1 in valid_timeframes:
            for tf2 in valid_timeframes:
                if tf1 != tf2:
                    agreement = self._calculate_timeframe_agreement(
                        indicator_signals[tf1], 
                        indicator_signals[tf2]
                    )
                    agreement_analysis['indicator_matrix'][f"{tf1}_vs_{tf2}"] = agreement
        
        # Calculate overall consensus strength
        all_agreements = list(agreement_analysis['indicator_matrix'].values())
        agreement_analysis['consensus_strength'] = np.mean(all_agreements) if all_agreements else 0
        
        return agreement_analysis
    
    def _calculate_timeframe_agreement(self, signals1: Dict, signals2: Dict) -> float:
        """Calculate agreement between two timeframes' signals"""
        common_indicators = set(signals1.keys()) & set(signals2.keys())
        
        if not common_indicators:
            return 0.5  # Neutral when no common indicators
        
        agreements = 0
        for indicator in common_indicators:
            if signals1[indicator] == signals2[indicator]:
                agreements += 1
        
        return agreements / len(common_indicators)
    
    def _analyze_sr_confluence(self, timeframe_data: Dict, valid_timeframes: List[str]) -> Dict:
        """Analyze support/resistance confluence across timeframes"""
        
        sr_analysis = {
            'support_levels': {},
            'resistance_levels': {},
            'confluence_zones': [],
            'nearest_confluence': None,
            'confluence_strength': 0
        }
        
        all_support_levels = []
        all_resistance_levels = []
        current_price = 0
        
        # Collect all S/R levels
        for tf in valid_timeframes:
            indicators = timeframe_data[tf]['indicators']
            
            # Get current price
            if not current_price:
                current_price = indicators.get('price', 0) or indicators.get('close', 0)
            
            # Support levels
            support = indicators.get('support_level')
            if support:
                all_support_levels.append({
                    'level': support,
                    'timeframe': tf,
                    'strength': self.timeframes[tf]['weight']
                })
            
            # Resistance levels  
            resistance = indicators.get('resistance_level')
            if resistance:
                all_resistance_levels.append({
                    'level': resistance,
                    'timeframe': tf,
                    'strength': self.timeframes[tf]['weight']
                })
        
        # Find confluence zones
        confluence_zones = self._find_level_confluences(all_support_levels + all_resistance_levels, current_price)
        
        sr_analysis['support_levels'] = all_support_levels
        sr_analysis['resistance_levels'] = all_resistance_levels
        sr_analysis['confluence_zones'] = confluence_zones
        
        # Find nearest confluence zone
        if confluence_zones and current_price:
            nearest = min(confluence_zones, key=lambda z: abs(z['level'] - current_price))
            sr_analysis['nearest_confluence'] = nearest
        
        return sr_analysis
    
    def _find_level_confluences(self, levels: List[Dict], current_price: float) -> List[Dict]:
        """Find price levels where multiple timeframes converge"""
        if len(levels) < 2:
            return []
        
        confluence_zones = []
        tolerance = current_price * 0.01 if current_price else 100  # 1% tolerance
        
        processed_levels = set()
        
        for i, level1 in enumerate(levels):
            if i in processed_levels:
                continue
                
            cluster_levels = [level1]
            cluster_strength = level1['strength']
            
            for j, level2 in enumerate(levels[i+1:], i+1):
                if j in processed_levels:
                    continue
                    
                if abs(level1['level'] - level2['level']) <= tolerance:
                    cluster_levels.append(level2)
                    cluster_strength += level2['strength']
                    processed_levels.add(j)
            
            if len(cluster_levels) >= 2:  # Confluence requires at least 2 timeframes
                avg_level = sum(l['level'] for l in cluster_levels) / len(cluster_levels)
                confluence_zones.append({
                    'level': avg_level,
                    'strength': cluster_strength,
                    'timeframes': [l['timeframe'] for l in cluster_levels],
                    'count': len(cluster_levels),
                    'distance_from_price': abs(avg_level - current_price) / current_price if current_price else float('inf')
                })
            
            processed_levels.add(i)
        
        # Sort by strength descending
        confluence_zones.sort(key=lambda z: z['strength'], reverse=True)
        return confluence_zones[:5]  # Return top 5 confluence zones
    
    def _analyze_volume_confluence(self, timeframe_data: Dict, valid_timeframes: List[str], tf_weights: Dict) -> Dict:
        """Analyze volume confirmation across timeframes"""
        
        volume_analysis = {
            'volume_trends': {},
            'volume_weighted_score': 0,
            'volume_confirmation': 'NEUTRAL',
            'institutional_activity': {}
        }
        
        volume_trends = {}
        
        for tf in valid_timeframes:
            indicators = timeframe_data[tf]['indicators']
            
            # Volume trend analysis
            current_volume = indicators.get('volume', 0)
            avg_volume = indicators.get('volume_sma_20', current_volume)
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            volume_trends[tf] = {
                'volume_ratio': volume_ratio,
                'trend': 'HIGH' if volume_ratio > 1.5 else 'NORMAL' if volume_ratio > 0.8 else 'LOW',
                'weight': tf_weights[tf]
            }
        
        # Calculate weighted volume score
        weighted_volume_score = sum(
            trends['volume_ratio'] * trends['weight'] 
            for trends in volume_trends.values()
        )
        
        volume_analysis['volume_trends'] = volume_trends
        volume_analysis['volume_weighted_score'] = weighted_volume_score
        
        # Determine volume confirmation (fixed thresholds)
        if weighted_volume_score > 1.5:
            volume_analysis['volume_confirmation'] = 'STRONG_CONFIRMATION'
        elif weighted_volume_score > 1.2:
            volume_analysis['volume_confirmation'] = 'CONFIRMATION' 
        elif weighted_volume_score > 0.8:
            volume_analysis['volume_confirmation'] = 'WEAK_CONFIRMATION'
        else:
            volume_analysis['volume_confirmation'] = 'LOW_VOLUME'
        
        return volume_analysis
    
    def _analyze_cross_timeframe_divergences(self, timeframe_data: Dict, valid_timeframes: List[str]) -> Dict:
        """Analyze divergences between price action and indicators across timeframes"""
        
        divergence_analysis = {
            'rsi_divergences': {},
            'macd_divergences': {},
            'volume_divergences': {},
            'divergence_risk_score': 0,
            'critical_divergences': []
        }
        
        # For each timeframe, check for divergences
        for tf in valid_timeframes:
            indicators = timeframe_data[tf]['indicators']
            
            # RSI divergence (simplified - would need historical data for full analysis)
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                divergence_analysis['rsi_divergences'][tf] = 'POTENTIAL_BEARISH'
            elif rsi < 30:
                divergence_analysis['rsi_divergences'][tf] = 'POTENTIAL_BULLISH'
            
            # MACD divergence analysis
            macd_hist = indicators.get('macd_histogram', 0)
            if abs(macd_hist) < 0.1:  # Weakening momentum
                divergence_analysis['macd_divergences'][tf] = 'MOMENTUM_WEAKENING'
        
        return divergence_analysis
    
    def _calculate_signal_strength(self, confluence: Dict, tf_weights: Dict) -> Dict:
        """Calculate overall signal strength from confluence analysis"""
        
        signal_strength = {
            'trend_strength': 0,
            'momentum_strength': 0,
            'support_resistance_strength': 0,
            'volume_strength': 0,
            'overall_strength': 0,
            'confidence_level': 'LOW'
        }
        
        # Trend strength from alignment
        trend_data = confluence.get('trend_alignment', {})
        signal_strength['trend_strength'] = trend_data.get('trend_reliability', 0) * 100
        
        # Momentum strength
        momentum_data = confluence.get('momentum_confluence', {})
        signal_strength['momentum_strength'] = momentum_data.get('momentum_alignment', 0) * 100
        
        # Support/Resistance strength
        sr_data = confluence.get('support_resistance_confluence', {})
        confluence_zones = sr_data.get('confluence_zones', [])
        if confluence_zones:
            signal_strength['support_resistance_strength'] = min(confluence_zones[0]['strength'] * 100, 100)
        
        # Volume strength
        volume_data = confluence.get('volume_confirmation', {})
        volume_score = volume_data.get('volume_weighted_score', 1.0)
        signal_strength['volume_strength'] = min((volume_score - 0.5) * 100, 100)
        
        # Calculate overall strength
        weights = [0.3, 0.3, 0.2, 0.2]  # Trend, Momentum, S/R, Volume
        strengths = [
            signal_strength['trend_strength'],
            signal_strength['momentum_strength'], 
            signal_strength['support_resistance_strength'],
            signal_strength['volume_strength']
        ]
        
        overall_strength = sum(w * s for w, s in zip(weights, strengths))
        signal_strength['overall_strength'] = overall_strength
        
        # Determine confidence level
        if overall_strength > 75:
            signal_strength['confidence_level'] = 'VERY_HIGH'
        elif overall_strength > 60:
            signal_strength['confidence_level'] = 'HIGH'
        elif overall_strength > 40:
            signal_strength['confidence_level'] = 'MEDIUM'
        elif overall_strength > 25:
            signal_strength['confidence_level'] = 'LOW'
        else:
            signal_strength['confidence_level'] = 'VERY_LOW'
        
        return signal_strength
    
    def _calculate_overall_confluence(self, confluence: Dict, tf_weights: Dict) -> Dict:
        """Calculate overall confluence score and rating"""
        
        overall = {
            'confluence_score': 0,
            'confluence_rating': 'NEUTRAL',
            'key_factors': [],
            'risk_factors': [],
            'confluence_summary': ''
        }
        
        score_components = []
        
        # Trend confluence score
        trend_data = confluence.get('trend_alignment', {})
        trend_score = trend_data.get('alignment_percentage', 0)
        if trend_score > 70:
            score_components.append(('trend_alignment', trend_score * 0.3))
            overall['key_factors'].append(f"Strong trend alignment ({trend_score:.0f}%)")
        
        # Momentum confluence score
        momentum_data = confluence.get('momentum_confluence', {})
        momentum_alignment = momentum_data.get('momentum_alignment', 0) * 100
        if momentum_alignment > 60:
            score_components.append(('momentum_alignment', momentum_alignment * 0.25))
            overall['key_factors'].append(f"Good momentum alignment ({momentum_alignment:.0f}%)")
        
        # Indicator agreement score
        indicator_data = confluence.get('indicator_agreement', {})
        consensus_strength = indicator_data.get('consensus_strength', 0) * 100
        if consensus_strength > 50:
            score_components.append(('indicator_consensus', consensus_strength * 0.2))
            overall['key_factors'].append(f"Indicator consensus ({consensus_strength:.0f}%)")
        
        # Support/Resistance confluence
        sr_data = confluence.get('support_resistance_confluence', {})
        confluence_zones = sr_data.get('confluence_zones', [])
        if confluence_zones:
            sr_score = min(confluence_zones[0]['strength'] * 100, 100)
            score_components.append(('sr_confluence', sr_score * 0.15))
            overall['key_factors'].append(f"S/R confluence at {confluence_zones[0]['level']:.2f}")
        
        # Volume confirmation
        volume_data = confluence.get('volume_confirmation', {})
        volume_confirmation = volume_data.get('volume_confirmation', 'NEUTRAL')
        if volume_confirmation in ['STRONG_CONFIRMATION', 'CONFIRMATION']:
            score_components.append(('volume_confirmation', 70 * 0.1))
            overall['key_factors'].append(f"Volume {volume_confirmation.lower()}")
        
        # Calculate final confluence score
        total_score = sum(score for _, score in score_components)
        overall['confluence_score'] = min(total_score, 100)
        
        # Determine confluence rating
        if total_score > 80:
            overall['confluence_rating'] = 'VERY_STRONG'
        elif total_score > 65:
            overall['confluence_rating'] = 'STRONG'
        elif total_score > 50:
            overall['confluence_rating'] = 'MODERATE'
        elif total_score > 35:
            overall['confluence_rating'] = 'WEAK'
        else:
            overall['confluence_rating'] = 'VERY_WEAK'
        
        return overall
    
    def _generate_confluence_recommendations(self, confluence: Dict, timeframe_data: Dict) -> Dict:
        """Generate trading recommendations based on confluence analysis"""
        
        recommendations = {
            'primary_signal': 'NEUTRAL',
            'confidence': 'LOW',
            'entry_strategy': 'WAIT',
            'risk_level': 'HIGH',
            'timeframe_focus': None,
            'key_levels': [],
            'action_items': []
        }
        
        overall_confluence = confluence.get('overall_confluence', {})
        signal_strength = confluence.get('signal_strength', {})
        trend_data = confluence.get('trend_alignment', {})
        
        confluence_score = overall_confluence.get('confluence_score', 0)
        confidence_level = signal_strength.get('confidence_level', 'LOW')
        dominant_trend = trend_data.get('dominant_trend', 'NEUTRAL')
        
        # Determine primary signal
        recommendations['primary_signal'] = dominant_trend
        recommendations['confidence'] = confidence_level
        
        # Determine entry strategy
        if confluence_score > 70 and confidence_level in ['HIGH', 'VERY_HIGH']:
            recommendations['entry_strategy'] = 'AGGRESSIVE'
            recommendations['risk_level'] = 'LOW'
        elif confluence_score > 50 and confidence_level in ['MEDIUM', 'HIGH']:
            recommendations['entry_strategy'] = 'MODERATE'  
            recommendations['risk_level'] = 'MEDIUM'
        elif confluence_score > 35:
            recommendations['entry_strategy'] = 'CONSERVATIVE'
            recommendations['risk_level'] = 'HIGH'
        else:
            recommendations['entry_strategy'] = 'WAIT'
            recommendations['risk_level'] = 'VERY_HIGH'
        
        # Identify key focus timeframe
        trend_individual = trend_data.get('individual_trends', {})
        if trend_individual:
            # Find strongest trend signal
            strongest_tf = max(trend_individual.items(), 
                             key=lambda x: x[1].get('trend_strength', 0) * x[1].get('weight', 0))
            recommendations['timeframe_focus'] = strongest_tf[0]
        
        # Add confluence zones as key levels
        sr_data = confluence.get('support_resistance_confluence', {})
        confluence_zones = sr_data.get('confluence_zones', [])
        for zone in confluence_zones[:3]:  # Top 3 zones
            recommendations['key_levels'].append({
                'level': zone['level'],
                'strength': zone['strength'],
                'timeframes': zone['timeframes']
            })
        
        # Generate action items
        if recommendations['primary_signal'] != 'NEUTRAL':
            recommendations['action_items'].append(
                f"Monitor {recommendations['timeframe_focus']} timeframe for {dominant_trend.lower()} continuation"
            )
        
        if confluence_zones:
            nearest_zone = confluence_zones[0]
            recommendations['action_items'].append(
                f"Watch confluence zone at {nearest_zone['level']:.2f} ({len(nearest_zone['timeframes'])} timeframes)"
            )
        
        return recommendations
    
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