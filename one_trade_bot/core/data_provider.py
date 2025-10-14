"""
Core Data Provider for One Good Trade Per Day Bot
🎯 Reliable OHLCV data fetching and validation
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

class DataProvider:
    """
    Handles all market data fetching with proper error handling
    NO FALLBACKS - Real data or explicit failure
    """
    
    def __init__(self, exchange_name: str, api_key: str = "", secret: str = "", sandbox: bool = True):
        """Initialize exchange connection"""
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        self.exchange = None
        self.connected = False
        
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange_name.lower())
            
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': sandbox,
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds
            })
            
            # Test connection
            self.exchange.load_markets()
            self.connected = True
            logger.info(f"✅ Connected to {exchange_name} {'(sandbox)' if sandbox else '(live)'}")
            
        except Exception as e:
            logger.error(f"🚨 Failed to connect to {exchange_name}: {str(e)}")
            raise ConnectionError(f"Cannot connect to {exchange_name}: {str(e)}")
    
    def get_ohlcv(self, symbol: str, timeframe: str, days: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data for analysis
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USD')
            timeframe: Candle size ('1h', '4h', '1d')
            days: How many days of history
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If insufficient data or connection issues
        """
        if not self.connected:
            raise ValueError(f"Not connected to {self.exchange_name}")
            
        try:
            # Calculate required candles based on timeframe
            candles_needed = self._calculate_candles_needed(timeframe, days)
            
            # Fetch data with retries
            ohlcv = self._fetch_with_retries(symbol, timeframe, candles_needed)
            
            if not ohlcv or len(ohlcv) < candles_needed * 0.8:  # Need at least 80% of requested data
                raise ValueError(f"Insufficient data for {symbol}: got {len(ohlcv) if ohlcv else 0}, needed {candles_needed}")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()
            
            # Validate data quality
            self._validate_ohlcv_data(df, symbol, timeframe)
            
            logger.debug(f"✅ Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"🚨 Data fetch failed for {symbol} {timeframe}: {str(e)}")
            raise ValueError(f"Cannot fetch data for {symbol}: {str(e)}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        if not self.connected:
            raise ValueError(f"Not connected to {self.exchange_name}")
            
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            if not price or price <= 0:
                raise ValueError(f"Invalid price for {symbol}: {price}")
                
            logger.debug(f"Current price {symbol}: ${price}")
            return price
            
        except Exception as e:
            logger.error(f"🚨 Failed to get current price for {symbol}: {str(e)}")
            raise ValueError(f"Cannot get price for {symbol}: {str(e)}")
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data including spreads and volume"""
        if not self.connected:
            raise ValueError(f"Not connected to {self.exchange_name}")
            
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            orderbook = self.exchange.fetch_order_book(symbol, limit=5)
            
            bid = orderbook['bids'][0][0] if orderbook['bids'] else None
            ask = orderbook['asks'][0][0] if orderbook['asks'] else None
            
            if not all([bid, ask, ticker['last']]):
                raise ValueError(f"Incomplete market data for {symbol}")
            
            spread = (ask - bid) / ((bid + ask) / 2)
            
            market_data = {
                'symbol': symbol,
                'last': ticker['last'],
                'bid': bid,
                'ask': ask,
                'mid': (bid + ask) / 2,
                'spread_pct': spread,
                'volume_24h': ticker['quoteVolume'] or 0,
                'timestamp': datetime.now(timezone.utc)
            }
            
            logger.debug(f"Market data {symbol}: price=${market_data['last']:.4f}, spread={spread*100:.3f}%")
            return market_data
            
        except Exception as e:
            logger.error(f"🚨 Failed to get market data for {symbol}: {str(e)}")
            raise ValueError(f"Cannot get market data for {symbol}: {str(e)}")
    
    def _calculate_candles_needed(self, timeframe: str, days: int) -> int:
        """Calculate how many candles needed for N days"""
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080
        }
        
        if timeframe not in timeframe_minutes:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        minutes_per_day = 1440
        minutes_per_candle = timeframe_minutes[timeframe]
        candles_per_day = minutes_per_day / minutes_per_candle
        
        return int(days * candles_per_day * 1.1)  # 10% buffer for weekends/gaps
    
    def _fetch_with_retries(self, symbol: str, timeframe: str, limit: int, max_retries: int = 3) -> List:
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if ohlcv:
                    return ohlcv
            except Exception as e:
                logger.warning(f"Fetch attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e
        return []
    
    def _validate_ohlcv_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Validate OHLCV data quality"""
        if df.empty:
            raise ValueError(f"Empty dataset for {symbol}")
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for {symbol}: {missing_cols}")
        
        # Check for NaN values
        nan_cols = df[required_cols].columns[df[required_cols].isnull().any()].tolist()
        if nan_cols:
            raise ValueError(f"NaN values in {symbol} data: {nan_cols}")
        
        # Check for impossible OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['volume'] < 0)
        ).sum()
        
        if invalid_ohlc > 0:
            raise ValueError(f"Invalid OHLCV relationships in {symbol}: {invalid_ohlc} candles")
        
        # Check for reasonable price ranges (no massive gaps)
        price_changes = df['close'].pct_change().abs()
        extreme_moves = (price_changes > 0.5).sum()  # >50% moves
        
        if extreme_moves > len(df) * 0.01:  # >1% of candles
            logger.warning(f"⚠️ Extreme price movements detected in {symbol}: {extreme_moves} candles")
        
        logger.debug(f"✅ Data validation passed for {symbol} {timeframe}")


class TechnicalIndicators:
    """
    Calculate technical indicators with proper validation
    NO FALLBACKS - Real calculations or explicit errors
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        if len(data) < period:
            raise ValueError(f"Insufficient data for SMA({period}): need {period}, got {len(data)}")
        return data.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if len(data) < period + 1:
            raise ValueError(f"Insufficient data for RSI({period}): need {period + 1}, got {len(data)}")
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Line, Signal Line, Histogram"""
        if len(data) < slow + signal:
            raise ValueError(f"Insufficient data for MACD: need {slow + signal}, got {len(data)}")
        
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        if len(df) < period:
            raise ValueError(f"Insufficient data for ATR({period}): need {period}, got {len(df)}")
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands: Upper, Middle (SMA), Lower"""
        if len(data) < period:
            raise ValueError(f"Insufficient data for Bollinger Bands({period}): need {period}, got {len(data)}")
        
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def bb_squeeze(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20) -> pd.Series:
        """Bollinger Band Squeeze Detection"""
        if len(df) < max(bb_period, kc_period):
            raise ValueError(f"Insufficient data for BB Squeeze")
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'], bb_period)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Keltner Channels (using ATR)
        atr = TechnicalIndicators.atr(df, kc_period)
        kc_upper = bb_middle + (atr * 1.5)
        kc_lower = bb_middle - (atr * 1.5)
        kc_width = (kc_upper - kc_lower) / bb_middle
        
        # Squeeze occurs when BB width < KC width
        squeeze_ratio = bb_width / kc_width
        
        return squeeze_ratio