#!/usr/bin/env python3
"""
Real-time Data Integration System
Provides WebSocket connections for live price feeds and real-time analysis updates
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import threading
import time
import queue
from collections import deque
import ccxt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PriceUpdate:
    """Real-time price update data"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None

@dataclass
class RealtimeConfig:
    """Configuration for real-time data integration"""
    enabled: bool = True
    update_interval: int = 5  # seconds
    price_buffer_size: int = 1000
    analysis_trigger_interval: int = 60  # seconds
    websocket_timeout: int = 30
    max_reconnect_attempts: int = 5
    
    # Exchange-specific WebSocket URLs
    websocket_urls: Dict[str, str] = None
    
    def __post_init__(self):
        if self.websocket_urls is None:
            self.websocket_urls = {
                'binance': 'wss://stream.binance.com:9443/ws/',
                'coinbase': 'wss://ws-feed.pro.coinbase.com',
                'kraken': 'wss://ws.kraken.com',
                'bitstamp': 'wss://ws.bitstamp.net'
            }

class RealtimeDataManager:
    """
    Real-time data management system with WebSocket connections
    """
    
    def __init__(self, config: RealtimeConfig, exchange_id: str = 'kraken'):
        self.config = config
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)()
        
        # Data storage
        self.price_buffers: Dict[str, deque] = {}
        self.latest_prices: Dict[str, PriceUpdate] = {}
        self.subscribed_symbols: List[str] = []
        
        # WebSocket connections
        self.websockets: Dict[str, Any] = {}
        self.connection_status: Dict[str, bool] = {}
        
        # Threading and asyncio
        self.event_loop = None
        self.websocket_thread = None
        self.is_running = False
        
        # Callbacks
        self.price_update_callbacks: List[Callable] = []
        self.analysis_trigger_callbacks: List[Callable] = []
        
        # Analysis triggering
        self.last_analysis_trigger = {}
        
        logger.info(f"Real-time data manager initialized for {exchange_id}")
    
    def add_price_update_callback(self, callback: Callable[[str, PriceUpdate], None]):
        """Add callback for price updates"""
        self.price_update_callbacks.append(callback)
    
    def add_analysis_trigger_callback(self, callback: Callable[[str], None]):
        """Add callback for analysis triggers"""
        self.analysis_trigger_callbacks.append(callback)
    
    def subscribe_symbol(self, symbol: str):
        """Subscribe to real-time data for a symbol"""
        if symbol not in self.subscribed_symbols:
            self.subscribed_symbols.append(symbol)
            self.price_buffers[symbol] = deque(maxlen=self.config.price_buffer_size)
            self.last_analysis_trigger[symbol] = datetime.now()
            
            logger.info(f"Subscribed to real-time data for {symbol}")
            
            # If WebSocket is already running, add subscription
            if self.is_running:
                asyncio.run_coroutine_threadsafe(
                    self._subscribe_websocket(symbol),
                    self.event_loop
                )
    
    def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from real-time data for a symbol"""
        if symbol in self.subscribed_symbols:
            self.subscribed_symbols.remove(symbol)
            
            if symbol in self.price_buffers:
                del self.price_buffers[symbol]
            if symbol in self.latest_prices:
                del self.latest_prices[symbol]
            if symbol in self.last_analysis_trigger:
                del self.last_analysis_trigger[symbol]
            
            logger.info(f"Unsubscribed from real-time data for {symbol}")
    
    def start_realtime_feeds(self):
        """Start real-time data feeds"""
        if self.is_running:
            logger.warning("Real-time feeds already running")
            return
        
        if not self.subscribed_symbols:
            logger.warning("No symbols subscribed for real-time data")
            return
        
        self.is_running = True
        
        # Start WebSocket thread
        self.websocket_thread = threading.Thread(
            target=self._run_websocket_loop,
            daemon=True,
            name="RealtimeWebSocket"
        )
        self.websocket_thread.start()
        
        logger.info(f"Started real-time feeds for {len(self.subscribed_symbols)} symbols")
    
    def stop_realtime_feeds(self):
        """Stop real-time data feeds"""
        self.is_running = False
        
        if self.event_loop:
            # Cancel all WebSocket connections
            for symbol in self.subscribed_symbols:
                if symbol in self.websockets:
                    asyncio.run_coroutine_threadsafe(
                        self.websockets[symbol].close(),
                        self.event_loop
                    )
        
        if self.websocket_thread:
            self.websocket_thread.join(timeout=5)
        
        logger.info("Stopped real-time data feeds")
    
    def _run_websocket_loop(self):
        """Run the WebSocket event loop in a separate thread"""
        try:
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            
            # Start WebSocket connections for all subscribed symbols
            tasks = []
            for symbol in self.subscribed_symbols:
                task = self.event_loop.create_task(self._websocket_handler(symbol))
                tasks.append(task)
            
            # Run the event loop
            self.event_loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
        finally:
            if self.event_loop:
                self.event_loop.close()
    
    async def _websocket_handler(self, symbol: str):
        """Handle WebSocket connection for a specific symbol"""
        
        reconnect_attempts = 0
        
        while self.is_running and reconnect_attempts < self.config.max_reconnect_attempts:
            try:
                # Get WebSocket URL and message format for exchange
                ws_url, subscribe_message = self._get_websocket_config(symbol)
                
                logger.info(f"Connecting to WebSocket for {symbol}: {ws_url}")
                
                async with websockets.connect(
                    ws_url,
                    timeout=self.config.websocket_timeout
                ) as websocket:
                    
                    self.websockets[symbol] = websocket
                    self.connection_status[symbol] = True
                    reconnect_attempts = 0  # Reset on successful connection
                    
                    # Send subscription message
                    if subscribe_message:
                        await websocket.send(json.dumps(subscribe_message))
                        logger.debug(f"Sent subscription for {symbol}")
                    
                    # Listen for messages
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        try:
                            await self._process_websocket_message(symbol, message)
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message for {symbol}: {e}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"WebSocket connection closed for {symbol}")
                self.connection_status[symbol] = False
                
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                self.connection_status[symbol] = False
                reconnect_attempts += 1
                
                if reconnect_attempts < self.config.max_reconnect_attempts:
                    wait_time = min(2 ** reconnect_attempts, 30)  # Exponential backoff, max 30s
                    logger.info(f"Reconnecting to {symbol} in {wait_time}s (attempt {reconnect_attempts})")
                    await asyncio.sleep(wait_time)
        
        logger.error(f"Max reconnection attempts reached for {symbol}")
        self.connection_status[symbol] = False
    
    def _get_websocket_config(self, symbol: str) -> tuple:
        """Get WebSocket URL and subscription message for exchange"""
        
        # Convert symbol to exchange format
        exchange_symbol = self._format_symbol_for_exchange(symbol)
        
        if self.exchange_id == 'kraken':
            ws_url = self.config.websocket_urls['kraken']
            subscribe_message = {
                "event": "subscribe",
                "pair": [exchange_symbol],
                "subscription": {"name": "ticker"}
            }
            
        elif self.exchange_id == 'binance':
            # Binance uses individual streams
            stream_name = f"{exchange_symbol.lower()}@ticker"
            ws_url = f"{self.config.websocket_urls['binance']}{stream_name}"
            subscribe_message = None  # No subscription message needed
            
        elif self.exchange_id == 'coinbase':
            ws_url = self.config.websocket_urls['coinbase']
            subscribe_message = {
                "type": "subscribe",
                "product_ids": [exchange_symbol],
                "channels": ["ticker"]
            }
            
        else:
            # Generic fallback - use REST API polling
            return None, None
        
        return ws_url, subscribe_message
    
    def _format_symbol_for_exchange(self, symbol: str) -> str:
        """Format symbol for specific exchange WebSocket"""
        
        if self.exchange_id == 'kraken':
            # Kraken uses different format (e.g., XBTUSD instead of BTC/USD)
            return symbol.replace('BTC/', 'XBT').replace('/', '')
        
        elif self.exchange_id == 'binance':
            # Binance uses BTCUSDT format
            return symbol.replace('/', '')
        
        elif self.exchange_id == 'coinbase':
            # Coinbase uses BTC-USD format  
            return symbol.replace('/', '-')
        
        return symbol
    
    async def _process_websocket_message(self, symbol: str, message: str):
        """Process incoming WebSocket message"""
        
        try:
            data = json.loads(message)
            
            # Parse message based on exchange format
            price_update = self._parse_exchange_message(symbol, data)
            
            if price_update:
                # Store price update
                self.price_buffers[symbol].append(price_update)
                self.latest_prices[symbol] = price_update
                
                # Trigger callbacks
                for callback in self.price_update_callbacks:
                    try:
                        callback(symbol, price_update)
                    except Exception as e:
                        logger.error(f"Price update callback error: {e}")
                
                # Check if analysis should be triggered
                await self._check_analysis_trigger(symbol)
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message for {symbol}: {message}")
        except Exception as e:
            logger.error(f"Error processing message for {symbol}: {e}")
    
    def _parse_exchange_message(self, symbol: str, data: Dict) -> Optional[PriceUpdate]:
        """Parse exchange-specific WebSocket message"""
        
        try:
            if self.exchange_id == 'kraken':
                # Kraken ticker format
                if isinstance(data, list) and len(data) >= 2:
                    ticker_data = data[1]
                    if 'c' in ticker_data:  # 'c' is current price
                        return PriceUpdate(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            price=float(ticker_data['c'][0]),
                            volume=float(ticker_data['v'][1]) if 'v' in ticker_data else 0,
                            bid=float(ticker_data['b'][0]) if 'b' in ticker_data else None,
                            ask=float(ticker_data['a'][0]) if 'a' in ticker_data else None,
                            high_24h=float(ticker_data['h'][1]) if 'h' in ticker_data else None,
                            low_24h=float(ticker_data['l'][1]) if 'l' in ticker_data else None
                        )
            
            elif self.exchange_id == 'binance':
                # Binance ticker format
                if 'c' in data:  # Current price
                    return PriceUpdate(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        price=float(data['c']),
                        volume=float(data['v']) if 'v' in data else 0,
                        bid=float(data['b']) if 'b' in data else None,
                        ask=float(data['a']) if 'a' in data else None,
                        high_24h=float(data['h']) if 'h' in data else None,
                        low_24h=float(data['l']) if 'l' in data else None,
                        change_24h=float(data['P']) if 'P' in data else None
                    )
            
            elif self.exchange_id == 'coinbase':
                # Coinbase ticker format
                if 'type' in data and data['type'] == 'ticker':
                    return PriceUpdate(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        price=float(data['price']),
                        volume=float(data['volume_24h']) if 'volume_24h' in data else 0,
                        bid=float(data['best_bid']) if 'best_bid' in data else None,
                        ask=float(data['best_ask']) if 'best_ask' in data else None,
                        high_24h=float(data['high_24h']) if 'high_24h' in data else None,
                        low_24h=float(data['low_24h']) if 'low_24h' in data else None
                    )
        
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing {self.exchange_id} message: {e}")
        
        return None
    
    async def _check_analysis_trigger(self, symbol: str):
        """Check if analysis should be triggered for a symbol"""
        
        now = datetime.now()
        last_trigger = self.last_analysis_trigger.get(symbol, now)
        
        # Check if enough time has passed
        if (now - last_trigger).total_seconds() >= self.config.analysis_trigger_interval:
            self.last_analysis_trigger[symbol] = now
            
            # Trigger analysis callbacks
            for callback in self.analysis_trigger_callbacks:
                try:
                    # Run callback in thread pool to avoid blocking WebSocket
                    asyncio.get_event_loop().run_in_executor(None, callback, symbol)
                except Exception as e:
                    logger.error(f"Analysis trigger callback error: {e}")
    
    async def _subscribe_websocket(self, symbol: str):
        """Subscribe to WebSocket for a new symbol (if already running)"""
        
        if symbol not in self.websockets:
            # Start new WebSocket handler
            task = self.event_loop.create_task(self._websocket_handler(symbol))
            return task
    
    def get_latest_price(self, symbol: str) -> Optional[PriceUpdate]:
        """Get the latest price for a symbol"""
        return self.latest_prices.get(symbol)
    
    def get_price_history(self, symbol: str, minutes: int = 60) -> List[PriceUpdate]:
        """Get price history for the last N minutes"""
        
        if symbol not in self.price_buffers:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            update for update in self.price_buffers[symbol]
            if update.timestamp >= cutoff_time
        ]
    
    def get_realtime_ohlcv(self, symbol: str, timeframe: str = '1m') -> Optional[pd.DataFrame]:
        """Generate OHLCV data from real-time price updates"""
        
        if symbol not in self.price_buffers:
            return None
        
        updates = list(self.price_buffers[symbol])
        if not updates:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': update.timestamp,
            'price': update.price,
            'volume': update.volume
        } for update in updates])
        
        df.set_index('timestamp', inplace=True)
        
        # Resample to requested timeframe
        timeframe_map = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D'
        }
        
        resample_freq = timeframe_map.get(timeframe, '1T')
        
        ohlcv = df['price'].resample(resample_freq).ohlc()
        ohlcv['volume'] = df['volume'].resample(resample_freq).sum()
        
        return ohlcv.dropna()
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get WebSocket connection status for all symbols"""
        return self.connection_status.copy()
    
    def get_statistics(self) -> Dict:
        """Get real-time data statistics"""
        
        total_updates = sum(len(buffer) for buffer in self.price_buffers.values())
        connected_symbols = sum(1 for status in self.connection_status.values() if status)
        
        return {
            'is_running': self.is_running,
            'subscribed_symbols': len(self.subscribed_symbols),
            'connected_symbols': connected_symbols,
            'total_price_updates': total_updates,
            'price_update_callbacks': len(self.price_update_callbacks),
            'analysis_trigger_callbacks': len(self.analysis_trigger_callbacks),
            'exchange': self.exchange_id,
            'connection_status': self.connection_status
        }

class RealtimeAnalysisIntegrator:
    """
    Integrates real-time data with the enhanced analyzer
    """
    
    def __init__(self, analyzer, realtime_manager: RealtimeDataManager):
        self.analyzer = analyzer
        self.realtime_manager = realtime_manager
        
        # Setup callbacks
        self.realtime_manager.add_price_update_callback(self._on_price_update)
        self.realtime_manager.add_analysis_trigger_callback(self._on_analysis_trigger)
        
        # Analysis cache
        self.last_analysis = {}
        self.analysis_lock = threading.Lock()
        
        logger.info("Real-time analysis integrator initialized")
    
    def _on_price_update(self, symbol: str, price_update: PriceUpdate):
        """Handle real-time price updates"""
        
        # Update any cached analysis with latest price
        with self.analysis_lock:
            if symbol in self.last_analysis:
                analysis = self.last_analysis[symbol]
                
                # Update current price in analysis
                for tf_data in analysis.get('timeframe_data', {}).values():
                    if 'indicators' in tf_data:
                        tf_data['indicators']['current_price_realtime'] = price_update.price
                        tf_data['indicators']['price_update_time'] = price_update.timestamp.isoformat()
        
        logger.debug(f"Price update for {symbol}: {price_update.price}")
    
    def _on_analysis_trigger(self, symbol: str):
        """Handle analysis trigger from real-time data"""
        
        try:
            logger.info(f"Triggered real-time analysis for {symbol}")
            
            # Run full analysis
            analysis_result = self.analyzer.analyze_multi_timeframe(symbol)
            
            # Cache the result
            with self.analysis_lock:
                self.last_analysis[symbol] = analysis_result
            
            # Add real-time metadata
            analysis_result['realtime_metadata'] = {
                'triggered_by_realtime': True,
                'trigger_time': datetime.now().isoformat(),
                'latest_price': self.realtime_manager.get_latest_price(symbol),
                'connection_status': self.realtime_manager.get_connection_status().get(symbol, False)
            }
            
            logger.info(f"Real-time analysis completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Real-time analysis failed for {symbol}: {e}")
    
    def get_latest_analysis(self, symbol: str) -> Optional[Dict]:
        """Get the latest analysis result for a symbol"""
        with self.analysis_lock:
            return self.last_analysis.get(symbol)
    
    def start_realtime_analysis(self, symbols: List[str]):
        """Start real-time analysis for specified symbols"""
        
        for symbol in symbols:
            self.realtime_manager.subscribe_symbol(symbol)
        
        self.realtime_manager.start_realtime_feeds()
        logger.info(f"Started real-time analysis for {symbols}")
    
    def stop_realtime_analysis(self):
        """Stop real-time analysis"""
        self.realtime_manager.stop_realtime_feeds()
        logger.info("Stopped real-time analysis")

if __name__ == "__main__":
    # Demo usage
    config = RealtimeConfig(
        update_interval=5,
        analysis_trigger_interval=30
    )
    
    realtime_manager = RealtimeDataManager(config, exchange_id='kraken')
    
    # Subscribe to symbols
    realtime_manager.subscribe_symbol('BTC/USD')
    realtime_manager.subscribe_symbol('ETH/USD')
    
    print("Real-time Data Integration Demo")
    print("===============================")
    print(f"Statistics: {realtime_manager.get_statistics()}")
    
    # Start feeds (in a real application)
    # realtime_manager.start_realtime_feeds()
    # time.sleep(60)  # Run for 1 minute
    # realtime_manager.stop_realtime_feeds()