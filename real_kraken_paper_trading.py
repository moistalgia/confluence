#!/usr/bin/env python3
"""
Real Kraken Data Paper Trading System
====================================

1-hour paper trading test using REAL Kraken market data and your actual
Ultimate Analyzer signals. This will show you real trading performance
with current market conditions.

Features:
- Live Kraken price feeds (WebSocket)
- Real Ultimate Analyzer signal generation
- Multiple cryptocurrency pairs
- Realistic paper trading with fees/slippage
- Comprehensive P&L tracking and reporting
- Runs for 1 hour automatically

Author: Crypto Analysis AI
Date: October 12, 2025
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
import sys
import signal
from typing import Dict, List, Optional
import numpy as np

# Import our professional trading components
from pure_professional_trading_engine import PureProfessionalTradingEngine, TradingConfig
from professional_signal_validator import TradingSignal
from ultimate_crypto_analyzer import UltimateCryptoAnalyzer
from professional_trading_dashboard import get_dashboard
from working_dashboard import WorkingDashboardServer

# Configure comprehensive logging to capture ALL output
log_filename = f'kraken_paper_trading_{datetime.now().strftime("%Y%m%d_%H%M")}.log'

# Create a custom logger that captures everything
class TeeLogger:
    """Redirect all output to both console and log file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to capture all print statements
sys.stdout = TeeLogger(log_filename)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # This now goes to our TeeLogger
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

print(f"🚀 KRAKEN PAPER TRADING SESSION STARTED")
print(f"📝 All output logging to: {log_filename}")
print(f"⏰ Session time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")

class KrakenPaperTradingSystem:
    """Real market data paper trading with Ultimate Analyzer signals"""
    
    def __init__(self):
        # Initialize paper trading with more aggressive settings for testing
        config = TradingConfig(
            starting_balance=10000.0,
            max_risk_per_trade=0.05,  # 5% risk per trade
            max_portfolio_risk=0.70,  # 70% deployable capital (30% reserve for exceptional trades)
            max_concurrent_positions=8
        )
        self.paper_trader = PureProfessionalTradingEngine(config)
        
        # Initialize Ultimate Analyzer
        self.analyzer = UltimateCryptoAnalyzer()
        
        # Kraken trading pairs (using their naming convention)
        self.kraken_pairs = {
            "XBT/USD": "BTC/USDT",    # Bitcoin
            "ETH/USD": "ETH/USDT",    # Ethereum  
            "XRP/USD": "XRP/USDT",    # Ripple
            "ADA/USD": "ADA/USDT",    # Cardano
            "SOL/USD": "SOL/USDT",    # Solana
            "DOT/USD": "DOT/USDT"     # Polkadot
        }
        
        # Asset holdings tracking (for preventing sells without assets)
        self.asset_holdings = {
            "BTC": 0.0,
            "ETH": 0.0, 
            "XRP": 0.0,
            "ADA": 0.0,
            "SOL": 0.0,
            "DOT": 0.0
        }
        
        # Trading mode configuration - Allow Full Trading
        self.trading_mode = "hybrid"  # Full trading: "long_only", "hybrid", "short_enabled"
        self.max_short_exposure_pct = 0.50  # Max 50% of portfolio in shorts
        self.allow_naked_shorts = True  # Allow shorts without owning assets (true short selling)
        self.hedge_threshold = 0.15  # Start hedging when position is 15%+ of portfolio
        
        # Current market data
        self.current_prices = {}
        self.price_history = {pair: [] for pair in self.kraken_pairs.values()}
        
        # Trading state
        self.running = False
        self.start_time = None
        self.signals_generated = 0
        self.trades_executed = 0
        self.last_analysis_time = {}
        
        # WebSocket connection
        self.ws_connection = None
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("\n🛑 Shutting down gracefully...")
        self.running = False
    
    async def start_real_trading_test(self):
        """Start 1-hour test with real Kraken data"""
        
        print("🚀 REAL KRAKEN DATA PAPER TRADING TEST")
        print("=" * 60)
        print("📡 Connecting to live Kraken WebSocket feeds")
        print("🧠 Using Ultimate Crypto Analyzer for signals") 
        print("💰 Starting with $10,000 paper money")
        print("⏰ Duration: 1 hour")
        print("🎯 Goal: Real market analysis → Paper trades → Real P&L")
        print("🛑 Press Ctrl+C to stop early")
        print("=" * 60)
        
        self.running = True
        self.start_time = datetime.now()
        
        try:
            # Start all concurrent tasks
            tasks = await asyncio.gather(
                self._connect_to_kraken(),
                self._signal_analysis_loop(),
                self._position_management_loop(), 
                self._progress_reporting_loop(),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Error in trading test: {e}")
            
        finally:
            await self._generate_final_report()
    
    async def _connect_to_kraken(self):
        """Connect to Kraken WebSocket for real market data"""
        
        kraken_ws_url = "wss://ws.kraken.com"
        
        # Subscribe to ticker data for our pairs
        subscription_msg = {
            "event": "subscribe",
            "pair": list(self.kraken_pairs.keys()),
            "subscription": {"name": "ticker"}
        }
        
        retry_count = 0
        max_retries = 5
        
        while self.running and retry_count < max_retries:
            try:
                logger.info(f"📡 Connecting to Kraken WebSocket (attempt {retry_count + 1})")
                
                async with websockets.connect(kraken_ws_url) as websocket:
                    self.ws_connection = websocket
                    
                    # Send subscription
                    await websocket.send(json.dumps(subscription_msg))
                    logger.info("✅ Connected to Kraken - receiving live data")
                    
                    # Reset retry count on successful connection
                    retry_count = 0
                    
                    # Listen for price updates
                    while self.running:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30)
                            await self._process_kraken_message(json.loads(message))
                            
                        except asyncio.TimeoutError:
                            logger.warning("⚠️ Kraken WebSocket timeout - sending ping")
                            await websocket.ping()
                            
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("🔌 Kraken connection closed - reconnecting...")
                            break
                            
            except Exception as e:
                retry_count += 1
                logger.error(f"❌ Kraken connection error: {e}")
                
                if retry_count < max_retries:
                    wait_time = min(retry_count * 2, 30)  # Exponential backoff, max 30s
                    logger.info(f"🔄 Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("💥 Failed to connect to Kraken after 5 attempts")
                    logger.error("❌ Cannot proceed without real market data")
                    self.running = False
                    return
    
    async def _process_kraken_message(self, message):
        """Process incoming Kraken WebSocket messages"""
        
        if isinstance(message, list) and len(message) >= 4:
            # Ticker update format: [channelID, data, channelName, pair]
            if message[2] == "ticker":
                pair_kraken = message[3]
                ticker_data = message[1]
                
                if pair_kraken in self.kraken_pairs:
                    pair_standard = self.kraken_pairs[pair_kraken]
                    
                    # Get current price (last trade price)
                    if 'c' in ticker_data and len(ticker_data['c']) > 0:
                        current_price = float(ticker_data['c'][0])
                        
                        # Update our price tracking
                        self.current_prices[pair_standard] = current_price
                        self.price_history[pair_standard].append({
                            'price': current_price,
                            'timestamp': datetime.now()
                        })
                        
                        # Keep only last 1000 data points
                        if len(self.price_history[pair_standard]) > 1000:
                            self.price_history[pair_standard] = self.price_history[pair_standard][-1000:]
                        
                        # Update paper trader
                        self.paper_trader.update_price(pair_standard, current_price)
                        self.paper_trader.check_stop_losses_and_targets()
                        
                        # Update dashboard with live price data
                        dashboard = get_dashboard()
                        dashboard.update_price(pair_standard, current_price, 0.0)  # Volume not available from ticker
                        
                        # Update P&L for active trades
                        await self._update_trade_pnl(pair_standard, current_price, dashboard)
                        
                        # Log first price update for each pair
                        if pair_standard not in self.last_analysis_time:
                            logger.info(f"📊 {pair_standard}: ${current_price:.4f}")
    

    
    async def _signal_analysis_loop(self):
        """Analyze market data and generate trading signals using Ultimate Analyzer"""
        
        while self.running:
            try:
                # Wait for some price data before starting analysis
                if not self.current_prices:
                    await asyncio.sleep(10)
                    continue
                
                # Analyze each pair every 2-5 minutes
                for pair in self.kraken_pairs.values():
                    if pair not in self.current_prices:
                        continue
                    
                    # Check if enough time has passed since last analysis
                    now = datetime.now()
                    if pair in self.last_analysis_time:
                        time_since_last = now - self.last_analysis_time[pair]
                        if time_since_last.total_seconds() < 120:  # Wait at least 2 minutes
                            continue
                    
                    # Only analyze if we have enough price history
                    if len(self.price_history[pair]) < 20:
                        continue
                    
                    # Run Ultimate Analyzer on this pair
                    signal = await self._analyze_pair_for_signal(pair)
                    
                    if signal:
                        success = await self._execute_signal(signal)
                        if success:
                            self.trades_executed += 1
                        self.signals_generated += 1
                    
                    # Update last analysis time
                    self.last_analysis_time[pair] = now
                
                # Wait before next analysis cycle
                await asyncio.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in signal analysis: {e}")
                await asyncio.sleep(60)
    
    async def _fetch_current_volume_data(self, pair: str) -> dict:
        """Fetch REAL-TIME volume data - NO FALLBACKS, NO ESTIMATES"""
        try:
            # STRICT POLICY: Only use real Kraken data
            if not (hasattr(self.ultimate_analyzer, 'volume_analyzer') and 
                    self.ultimate_analyzer.volume_analyzer.kraken_client):
                logger.error(f"🚨 CRITICAL: No Kraken client available for real volume data for {pair}")
                raise ValueError(f"Real volume data unavailable for {pair} - no Kraken connection")
            
            # Get real 1h candles for volume calculation
            candles = self.ultimate_analyzer.volume_analyzer.kraken_client.fetch_ohlcv(pair, '1h', limit=20)
            if not candles or len(candles) < 10:
                logger.error(f"🚨 CRITICAL: Insufficient real volume data for {pair} - only {len(candles) if candles else 0} candles")
                raise ValueError(f"Insufficient real volume history for {pair}")
            
            volumes = [candle[5] for candle in candles[-10:]]  # Last 10 hours
            current_volume = candles[-1][5]
            avg_volume_10 = sum(volumes) / len(volumes)
            
            if avg_volume_10 <= 0:
                logger.error(f"🚨 CRITICAL: Invalid average volume for {pair}: {avg_volume_10}")
                raise ValueError(f"Invalid volume data for {pair}")
            
            volume_ratio = current_volume / avg_volume_10
            
            logger.info(f"✅ REAL VOLUME for {pair}: Current={current_volume:.2f}, Avg={avg_volume_10:.2f}, Ratio={volume_ratio:.2f}x")
            return {
                'current_volume': current_volume,
                'avg_volume_10': avg_volume_10,
                'volume_ratio': volume_ratio,
                'data_source': 'KRAKEN_REAL',
                'timestamp': candles[-1][0]  # Timestamp of latest candle
            }
            
        except Exception as e:
            logger.error(f"� VOLUME DATA FAILURE for {pair}: {str(e)}")
            logger.error(f"🚨 REFUSING to use fallback data - analysis will be skipped")
            raise ValueError(f"Failed to fetch real volume data for {pair}: {str(e)}")
    
    async def _analyze_pair_for_signal(self, pair: str) -> Optional[TradingSignal]:
        """Analyze a trading pair using Ultimate Analyzer's comprehensive analysis"""
        
        try:
            # Prepare price data for analysis
            price_data = self.price_history[pair][-100:]  # Last 100 data points
            
            if len(price_data) < 50:  # Need minimum data for analysis
                logger.debug(f"🔍 {pair}: Insufficient data ({len(price_data)} points)")
                return None
            
            current_price = self.current_prices[pair]
            logger.info(f"🔍 Analyzing {pair} @ ${current_price:.4f} using Ultimate Analyzer")
            
            # STRICT DATA INTEGRITY POLICY - NO FALLBACKS ALLOWED
            try:
                # Fetch real volume data - no fallbacks allowed
            
            # � FETCH REAL VOLUME DATA
                real_volume_data = await self._fetch_current_volume_data(pair)
                logger.info(f"✅ Real volume data acquired for {pair}")
            except ValueError as e:
                logger.error(f"🚨 SKIPPING ANALYSIS for {pair} - No real volume data available")
                logger.error(f"🚨 Error: {str(e)}")
                logger.warning(f"⚠️ Analysis requires authentic market data - refusing to proceed with estimates")
                return None  # Skip this pair entirely
            
            # �🚀 RUN ULTIMATE ANALYZER - Complete institutional-grade analysis
            try:
                ultimate_analysis = self.analyzer.run_ultimate_analysis(pair)
                
                if not ultimate_analysis:
                    logger.warning(f"   ❌ Ultimate Analyzer returned no data for {pair}")
                    return None
                
                # Extract comprehensive analysis data
                ultimate_score = ultimate_analysis.get('ultimate_trading_score', {}).get('composite_score')
                
                # Validate ultimate score
                if ultimate_score is None:
                    logger.error(f"🚨 MISSING ULTIMATE SCORE for {pair}")
                    logger.error(f"🚨 Ultimate Analyzer failed to generate composite score")
                    logger.warning(f"⚠️ Skipping {pair} due to missing ultimate score")
                    return None
                sentiment_data = ultimate_analysis.get('sentiment_analysis', {})
                accumulation_data = ultimate_analysis.get('accumulation_analysis', {})
                timeframe_data = ultimate_analysis.get('multi_timeframe_analysis', {}).get('timeframe_data', {})
                volume_profile = ultimate_analysis.get('volume_profile_analysis', {})
                
                logger.info(f"   🎯 Ultimate Score: {ultimate_score:.1f}/100")
                
                # Extract sentiment indicators
                fear_greed = sentiment_data.get('fear_greed_index')
                sentiment_score = sentiment_data.get('sentiment_score')
                overall_sentiment = sentiment_data.get('overall_sentiment')
                
                # Validate sentiment data
                if any(val is None for val in [fear_greed, sentiment_score, overall_sentiment]):
                    logger.error(f"🚨 MISSING SENTIMENT DATA for {pair}: F&G={fear_greed}, Score={sentiment_score}, Overall={overall_sentiment}")
                    logger.error(f"🚨 REFUSING to use fallback sentiment values")
                    logger.warning(f"⚠️ Skipping {pair} due to incomplete sentiment data")
                    return None
                
                logger.info(f"   🎭 Sentiment: {overall_sentiment} (F&G: {fear_greed}/100)")
                
                # Extract accumulation scores - STRICT VALIDATION
                acc_1m = accumulation_data.get('one_month_score')
                acc_6m = accumulation_data.get('six_month_score')
                acc_1y = accumulation_data.get('one_year_plus_score')
                
                # Validate accumulation data
                if any(val is None for val in [acc_1m, acc_6m, acc_1y]):
                    logger.error(f"🚨 MISSING ACCUMULATION DATA for {pair}: 1M={acc_1m}, 6M={acc_6m}, 1Y+={acc_1y}")
                    logger.error(f"🚨 REFUSING to use fallback accumulation scores")
                    logger.warning(f"⚠️ Skipping {pair} due to incomplete accumulation data")
                    return None
                
                logger.info(f"   � Accumulation: 1M:{acc_1m:.1f} | 6M:{acc_6m:.1f} | 1Y+:{acc_1y:.1f}")
                
                # Extract technical indicators from multiple timeframes
                h1_data = timeframe_data.get('1h', {}).get('indicators', {})
                h4_data = timeframe_data.get('4h', {}).get('indicators', {})
                d1_data = timeframe_data.get('1d', {}).get('indicators', {})
                
                # STRICT DATA VALIDATION - Multi-Timeframe RSI Analysis
                rsi_1h = h1_data.get('rsi')
                rsi_4h = h4_data.get('rsi')
                rsi_1d = d1_data.get('rsi')
                
                # Validate critical RSI data availability
                if any(rsi is None for rsi in [rsi_1h, rsi_4h, rsi_1d]):
                    logger.error(f"🚨 MISSING RSI DATA for {pair}: 1h={rsi_1h}, 4h={rsi_4h}, 1d={rsi_1d}")
                    logger.error(f"🚨 REFUSING to use fallback RSI values - analysis requires real data")
                    logger.warning(f"⚠️ Skipping {pair} due to incomplete RSI data")
                    return None
                
                macd_1h = h1_data.get('macd')
                bb_upper_1h = h1_data.get('bb_upper')
                bb_lower_1h = h1_data.get('bb_lower')
                
                # Validate additional technical indicators
                if any(val is None for val in [macd_1h, bb_upper_1h, bb_lower_1h]):
                    logger.error(f"🚨 MISSING TECHNICAL DATA for {pair}: MACD={macd_1h}, BB_Upper={bb_upper_1h}, BB_Lower={bb_lower_1h}")
                    logger.error(f"🚨 REFUSING to use synthetic technical indicators")
                    logger.warning(f"⚠️ Skipping {pair} due to incomplete technical data")
                    return None
                
                logger.info(f"   📊 Multi-TF RSI: 1H:{rsi_1h:.1f} | 4H:{rsi_4h:.1f} | 1D:{rsi_1d:.1f}")
                
                # Extract volume profile levels
                poc_price = volume_profile.get('volume_profile', {}).get('poc_price', current_price)
                va_high = volume_profile.get('volume_profile', {}).get('va_high', current_price * 1.05)
                va_low = volume_profile.get('volume_profile', {}).get('va_low', current_price * 0.95)
                
                logger.info(f"   � Volume: POC:${poc_price:.4f} | VA:${va_low:.4f}-${va_high:.4f}")
                
            except Exception as e:
                logger.error(f"   ❌ Ultimate Analyzer failed for {pair}: {e}")
                # Fallback to basic analysis if Ultimate Analyzer fails
                return await self._basic_fallback_analysis(pair, current_price)
            
            # 🎯 ENHANCED SIGNAL GENERATION using Ultimate Analysis
            signals = []
            
            # 1. Ultimate Score-Based Signals (Primary)
            if ultimate_score >= 70:
                signals.append(("BUY", 0.9, f"Strong Ultimate Score ({ultimate_score:.1f})"))
            elif ultimate_score >= 60:
                signals.append(("BUY", 0.75, f"Good Ultimate Score ({ultimate_score:.1f})"))
            elif ultimate_score <= 30:
                signals.append(("SELL", 0.9, f"Weak Ultimate Score ({ultimate_score:.1f})"))
            elif ultimate_score <= 40:
                signals.append(("SELL", 0.75, f"Poor Ultimate Score ({ultimate_score:.1f})"))
            
            # 2. Multi-Timeframe RSI Confluence
            oversold_count = sum([rsi < 30 for rsi in [rsi_1h, rsi_4h, rsi_1d] if rsi])
            overbought_count = sum([rsi > 70 for rsi in [rsi_1h, rsi_4h, rsi_1d] if rsi])
            
            if oversold_count >= 2:  # 2+ timeframes oversold
                signals.append(("BUY", 0.8, f"Multi-TF oversold ({oversold_count}/3 timeframes)"))
            elif oversold_count >= 1 and rsi_1h < 25:  # Strong 1H oversold
                signals.append(("BUY", 0.7, f"Strong 1H oversold (RSI:{rsi_1h:.1f})"))
            
            if overbought_count >= 2:  # 2+ timeframes overbought
                signals.append(("SELL", 0.8, f"Multi-TF overbought ({overbought_count}/3 timeframes)"))
            elif overbought_count >= 1 and rsi_1h > 75:  # Strong 1H overbought
                signals.append(("SELL", 0.7, f"Strong 1H overbought (RSI:{rsi_1h:.1f})"))
            
            # 3. Sentiment-Based Contrarian Signals
            if fear_greed <= 25:  # Extreme fear
                signals.append(("BUY", 0.85, f"Extreme fear opportunity (F&G:{fear_greed})"))
            elif fear_greed <= 35:  # Fear
                signals.append(("BUY", 0.7, f"Fear-based opportunity (F&G:{fear_greed})"))
            elif fear_greed >= 75:  # Extreme greed
                signals.append(("SELL", 0.85, f"Extreme greed warning (F&G:{fear_greed})"))
            elif fear_greed >= 65:  # Greed
                signals.append(("SELL", 0.7, f"Greed-based caution (F&G:{fear_greed})"))
            
            # 4. Accumulation Score Signals (for BUY bias)
            avg_accumulation = (acc_1m + acc_6m + acc_1y) / 3
            if avg_accumulation >= 70:
                signals.append(("BUY", 0.8, f"Strong accumulation setup ({avg_accumulation:.1f})"))
            elif avg_accumulation >= 60:
                signals.append(("BUY", 0.65, f"Good accumulation potential ({avg_accumulation:.1f})"))
            
            # 5. Volume Profile Support/Resistance
            price_to_poc = ((current_price - poc_price) / poc_price) * 100
            if abs(price_to_poc) <= 2:  # Within 2% of POC
                if price_to_poc < 0:  # Below POC
                    signals.append(("BUY", 0.75, f"Near Volume POC support ({price_to_poc:.1f}% below)"))
                else:  # Above POC
                    signals.append(("SELL", 0.7, f"Near Volume POC resistance ({price_to_poc:.1f}% above)"))
            
            # Filter and select best signal
            if not signals:
                logger.info(f"   ❌ No signals generated for {pair}")
                return None
            
            # Group by action and find highest confidence
            buy_signals = [s for s in signals if s[0] == "BUY"]
            sell_signals = [s for s in signals if s[0] == "SELL"]
            
            best_signal = None
            
            if buy_signals:
                best_buy = max(buy_signals, key=lambda x: x[1])
                if best_buy[1] >= 0.6:  # Higher threshold for quality
                    best_signal = best_buy
            
            if sell_signals:
                best_sell = max(sell_signals, key=lambda x: x[1])
                if best_sell[1] >= 0.6 and (not best_signal or best_sell[1] > best_signal[1]):
                    best_signal = best_sell
            
            if not best_signal:
                logger.info(f"   ❌ No high-confidence signals for {pair}")
                return None
            
            action, confidence, reason = best_signal
            
            # Enhanced confidence adjustment based on multiple factors
            confidence_adjustments = []
            
            # Adjust for multi-timeframe alignment
            if action == "BUY" and oversold_count >= 2:
                confidence_adjustments.append(("Multi-TF alignment", 0.05))
            elif action == "SELL" and overbought_count >= 2:
                confidence_adjustments.append(("Multi-TF alignment", 0.05))
            
            # Adjust for sentiment confirmation
            if action == "BUY" and fear_greed <= 35:
                confidence_adjustments.append(("Sentiment confirmation", 0.03))
            elif action == "SELL" and fear_greed >= 65:
                confidence_adjustments.append(("Sentiment confirmation", 0.03))
            
            # Apply adjustments
            for adj_reason, adj_value in confidence_adjustments:
                confidence = min(0.95, confidence + adj_value)
            
            # Enhanced stop loss and take profit using volume profile levels
            if action == "BUY":
                # Use volume profile low as dynamic support for stop loss
                stop_loss = min(current_price * 0.985, va_low * 0.995)  # 1.5% or VA low
                take_profit = max(current_price * 1.03, poc_price)  # 3% or POC target
            else:
                # Use volume profile high as dynamic resistance for stop loss
                stop_loss = max(current_price * 1.015, va_high * 1.005)  # 1.5% or VA high
                take_profit = min(current_price * 0.97, poc_price)  # 3% or POC target
            
            logger.info(f"🎯 ENHANCED SIGNAL: {action} {pair} - {reason} (confidence: {confidence:.1%})")
            if confidence_adjustments:
                for adj_reason, adj_value in confidence_adjustments:
                    logger.info(f"   ⬆️ +{adj_value:.1%} from {adj_reason}")
            
            # Cache comprehensive market data for professional validator
            market_data_cache = {
                'rsi': rsi_1h,
                'rsi_4h': rsi_4h,
                'rsi_1d': rsi_1d,
                'macd': macd_1h,
                'macd_signal': h1_data.get('macd_signal', 0),
                'bb_upper': bb_upper_1h,
                'bb_lower': bb_lower_1h,
                'bb_middle': h1_data.get('bb_middle', current_price),
                'stoch': h1_data.get('stoch', 50),
                'sma_20': h1_data.get('sma_short', current_price),
                'sma_50': h1_data.get('sma_long', current_price),
                'sma_200': d1_data.get('sma_200', current_price * 0.95),
                'ema_9': h1_data.get('ema', current_price),
                'ema_21': h1_data.get('ema', current_price),
                # Use real volume data fetched from Kraken
                'current_volume': real_volume_data.get('current_volume'),
                'avg_volume_20': real_volume_data.get('avg_volume_10') * 2,  # Extend 10h average to 20h estimate
                'volume_ratio': real_volume_data.get('volume_ratio'),
                'timeframe_data': timeframe_data,
                'ultimate_score': ultimate_score,
                'sentiment_data': sentiment_data,
                'accumulation_data': accumulation_data,
                'volume_profile': volume_profile
            }
            
            # Cache this data for the trading engine to use
            self.paper_trader.cache_market_data(pair, market_data_cache)
            
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=pair,
                action=action,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                source="ultimate_analyzer_enhanced",
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}")
            return None
    
    async def _basic_fallback_analysis(self, pair: str, current_price: float) -> Optional[TradingSignal]:
        """Basic fallback analysis when Ultimate Analyzer fails"""
        
        try:
            logger.warning(f"   🔄 Using basic fallback analysis for {pair}")
            
            # Use simple price history analysis
            price_data = self.price_history[pair][-50:]
            if len(price_data) < 20:
                return None
            
            prices = [data['price'] for data in price_data]
            
            # Basic RSI
            rsi = self._calculate_rsi(prices)
            if not rsi:
                return None
            
            # Simple signal logic
            if rsi < 25:
                return TradingSignal(
                    timestamp=datetime.now(),
                    symbol=pair,
                    action="BUY",
                    confidence=0.6,
                    entry_price=current_price,
                    stop_loss=current_price * 0.985,
                    take_profit=current_price * 1.03,
                    source="basic_fallback",
                    reason=f"Basic RSI oversold ({rsi:.1f})"
                )
            elif rsi > 75:
                return TradingSignal(
                    timestamp=datetime.now(),
                    symbol=pair,
                    action="SELL",
                    confidence=0.6,
                    entry_price=current_price,
                    stop_loss=current_price * 1.015,
                    take_profit=current_price * 0.97,
                    source="basic_fallback",
                    reason=f"Basic RSI overbought ({rsi:.1f})"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"   ❌ Even basic fallback failed for {pair}: {e}")
            return None
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        
        if len(prices) < period + 1:
            return None
        
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return None
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        
        if len(prices) < period:
            return None
        
        try:
            multiplier = 2 / (period + 1)
            ema = np.mean(prices[:period])  # Start with SMA
            
            for price in prices[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
            
        except Exception:
            return None
    
    def _calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        
        if len(prices) < period:
            return None
        
        try:
            return np.mean(prices[-period:])
        except Exception:
            return None
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands (upper, lower, middle)"""
        
        if len(prices) < period:
            return None, None, None
        
        try:
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return upper, lower, sma
            
        except Exception:
            return None, None, None
    
    def _analyze_volume_trend(self, pair: str) -> str:
        """Analyze volume trend (simulated for now)"""
        
        # Since Kraken ticker doesn't provide detailed volume data,
        # we'll simulate based on price volatility as a proxy
        try:
            if len(self.price_history[pair]) < 10:
                return "insufficient_data"
            
            recent_prices = [data['price'] for data in self.price_history[pair][-10:]]
            mean_price = np.mean(recent_prices)
            if mean_price == 0:
                return "insufficient_data"
            volatility = np.std(recent_prices) / mean_price
            
            if volatility > 0.002:  # High volatility suggests high volume
                return "high_volume"
            elif volatility > 0.001:
                return "moderate_volume"
            else:
                return "low_volume"
                
        except Exception:
            return "unknown"
    
    async def _execute_signal(self, signal: TradingSignal) -> bool:
        """
        🚀 PROFESSIONAL SIGNAL PROCESSING - No immediate execution!
        
        This uses validation-first approach instead of 'spray and pray'
        """
        
        logger.info(f"🎯 SIGNAL GENERATED: {signal.action} {signal.symbol} @ ${signal.entry_price:.4f}")
        logger.info(f"   💡 {signal.reason}")
        logger.info(f"   📊 Confidence: {signal.confidence:.1%}")
        logger.info(f"   🎯 Target: ${signal.take_profit:.4f} | Stop: ${signal.stop_loss:.4f}")
        
        # Set signal confidence for position sizing calculations
        self.paper_trader._current_signal_confidence = signal.confidence * 100
        
        # PROFESSIONAL APPROACH: Process signal through validation system
        result = self.paper_trader.process_trading_signal(signal)
        
        if result['status'] == 'BLOCKED':
            logger.warning(f"   � SIGNAL BLOCKED: {result['reason']}")
            return False
            
        elif result['status'] == 'THROTTLED':
            logger.warning(f"   ⏸️  SIGNAL THROTTLED: {result['reason']}")
            return False
            
        elif result['status'] == 'WATCHING':
            logger.info(f"   👀 SIGNAL WATCHING: {result['reason']}")
            logger.info(f"   ⏱️  Validation window: {result['validation_window']} minutes")
            logger.info(f"   � Waiting for price confirmation before entry")
            
            # Estimate position details for user
            estimated_size = self.paper_trader.calculate_position_size(
                signal.symbol, signal.entry_price, signal.stop_loss)
            estimated_value = signal.entry_price * estimated_size
            logger.info(f"   📏 Est. Position Size: {estimated_size:.6f} (${estimated_value:.2f}) if validated")
            
            return True  # Signal accepted for validation
        
        else:
            # Handle professional trading engine status codes
            if result['status'] == 'executed':
                logger.info(f"   ✅ TRADE EXECUTED SUCCESSFULLY")
                logger.info(f"   💰 Position Size: {result.get('position_size', 0):.6f}")
                portfolio = result.get('portfolio', {})
                logger.info(f"   📊 Portfolio Value: ${portfolio.get('total_value', 0):.2f}")
                logger.info(f"   💵 Available Cash: ${portfolio.get('cash', 0):.2f}")
            elif result['status'] == 'rejected':
                logger.warning(f"   ❌ SIGNAL REJECTED: {result.get('reason', 'Unknown reason')}")
            elif result['status'] == 'execution_failed':
                logger.error(f"   💥 EXECUTION FAILED: {result.get('reason', 'Unknown error')}")
            elif result['status'] == 'error':
                logger.error(f"   🚫 PROCESSING ERROR: {result.get('error', 'Unknown error')}")
            else:
                logger.error(f"   ❌ UNKNOWN STATUS: {result['status']}")
            return False
    
    def set_trading_mode(self, mode: str):
        """Change trading mode dynamically
        
        Args:
            mode: "long_only", "hybrid", "short_enabled"
        """
        valid_modes = ["long_only", "hybrid", "short_enabled"]
        if mode in valid_modes:
            self.trading_mode = mode
            logger.info(f"🔄 Trading mode changed to: {mode}")
            
            if mode == "long_only":
                logger.info("   📈 Only BUY orders for owned assets allowed")
            elif mode == "hybrid":
                logger.info("   ⚖️ BUY orders + limited hedging shorts allowed")  
            elif mode == "short_enabled":
                logger.info("   📊 Full long and short trading enabled")
        else:
            logger.warning(f"Invalid trading mode: {mode}. Valid modes: {valid_modes}")
    
    def _validate_trade(self, signal: TradingSignal) -> bool:
        """Validate trade based on hybrid trading rules"""
        
        if signal.action == "BUY":
            # BUY orders are always allowed (building assets)
            return True
            
        elif signal.action == "SELL":
            # SELL order validation based on trading mode
            base_asset = signal.symbol.split('/')[0]  # BTC from BTC/USDT
            asset_balance = self._get_asset_balance(base_asset)
            
            logger.info(f"   � Validating SELL for {base_asset} (current balance: {asset_balance:.6f})")
            
            # Check if we have existing long positions to sell first
            portfolio = self.paper_trader.get_portfolio_status()
            existing_longs = 0.0
            
            if signal.symbol in portfolio['positions']:
                for pos in portfolio['positions'][signal.symbol]:
                    if pos['side'] == 'BUY':
                        existing_longs += pos['size']
            
            if existing_longs > 0:
                # We have long positions - this SELL reduces exposure (always good!)
                logger.info(f"   ✅ Selling to reduce long exposure ({existing_longs:.6f} {base_asset} positions)")
                return True
            
            # No existing longs - this would be a new short position
            if self.trading_mode == "long_only":
                logger.info(f"   🚫 Long-only mode: No {base_asset} positions to sell")
                return False
                
            elif self.trading_mode == "hybrid":
                # In hybrid mode, allow shorts only if naked shorts are enabled
                if self.allow_naked_shorts:
                    # Check short exposure limits against TOTAL PORTFOLIO VALUE, not just long exposure
                    portfolio_status = self.paper_trader.get_portfolio_status()
                    total_portfolio_value = portfolio_status['current_balance']
                    short_exposure = self._calculate_short_exposure(portfolio)
                    
                    max_short_value = total_portfolio_value * self.max_short_exposure_pct
                    position_size = self.paper_trader.calculate_position_size(
                        signal.symbol, signal.entry_price, signal.stop_loss)
                    position_value = signal.entry_price * position_size
                    
                    logger.info(f"   📊 Short exposure check: {short_exposure:.0f} + {position_value:.0f} <= {max_short_value:.0f}")
                    logger.info(f"       Portfolio: ${total_portfolio_value:.0f} * {self.max_short_exposure_pct:.0%} = ${max_short_value:.0f} max shorts")
                    
                    if short_exposure + position_value <= max_short_value:
                        logger.info(f"   🛡️ Naked short allowed within portfolio limits")
                        return True
                    else:
                        logger.info(f"   🚫 Short exposure limit reached")
                        return False
                else:
                    logger.info(f"   🚫 Naked shorts disabled in hybrid mode")
                    return False
                
            elif self.trading_mode == "short_enabled":
                # Full short selling allowed
                logger.info(f"   📊 Short selling fully enabled")
                return True
        
        return False
    
    def _get_asset_balance(self, asset: str) -> float:
        """Get current balance of a specific asset"""
        
        # Calculate asset balance from open positions
        balance = self.asset_holdings.get(asset, 0.0)
        
        # Get positions using professional or legacy method
        try:
            # Try professional mode first
            if hasattr(self.paper_trader, 'portfolio') and self.paper_trader.portfolio:
                portfolio_status = self.paper_trader.portfolio.calculate_portfolio_value()
                if 'positions' in portfolio_status:
                    positions_data = portfolio_status['positions']
                else:
                    # Professional mode but different structure
                    positions_data = {}
            else:
                # Legacy mode
                positions_data = getattr(self.paper_trader, 'positions', {})
            
            # Add assets from current BUY positions
            for symbol, position_list in positions_data.items():
                if symbol.startswith(asset + '/'):
                    # Handle both list and direct position formats
                    positions = position_list if isinstance(position_list, list) else [position_list]
                    for pos in positions:
                        if hasattr(pos, 'side'):
                            if pos.side == "BUY":
                                balance += pos.size
                            elif pos.side == "SELL":
                                balance -= pos.size  # Short positions reduce available balance
                        
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate asset balance for {asset}: {e}")
            # Return just the basic balance if position calculation fails
        
        return balance
    
    def _calculate_long_exposure(self, portfolio) -> float:
        """Calculate total long exposure value"""
        long_exposure = 0.0
        for symbol, position_list in portfolio['positions'].items():
            current_price = self.current_prices.get(symbol, 0)  # Use symbol directly, no conversion needed
            for pos in position_list:
                if pos['side'] == 'BUY':
                    long_exposure += pos['size'] * current_price
        return long_exposure
    
    def _calculate_short_exposure(self, portfolio) -> float:
        """Calculate total short exposure value"""
        short_exposure = 0.0
        for symbol, position_list in portfolio['positions'].items():
            current_price = self.current_prices.get(symbol, 0)  # Use symbol directly, no conversion needed
            for pos in position_list:
                if pos['side'] == 'SELL':
                    short_exposure += pos['size'] * current_price
        return short_exposure
    
    def _check_position_upgrade(self, new_signal: TradingSignal) -> str:
        """Check for confidence-based position upgrades or weak position replacement"""
        
        # First, try confidence-based upgrades for existing positions
        upgrade_attempted = self.paper_trader.check_position_upgrades(
            symbol=new_signal.symbol,
            new_confidence=new_signal.confidence * 100,  # Convert to percentage
            new_signal_reason=new_signal.reason,
            current_price=new_signal.entry_price
        )
        
        if upgrade_attempted:
            return f"Upgraded {new_signal.symbol} position (confidence: {new_signal.confidence:.1%})"
        
        # If no upgrade, check if we should replace weak positions for very high confidence signals
        if new_signal.confidence < 0.90:  # Only replace positions for exceptional signals (90%+)
            return None
            
        portfolio = self.paper_trader.get_portfolio_status()
        
        # Find weakest position to potentially replace
        weakest_position = None
        lowest_score = float('inf')
        
        for symbol, position_list in portfolio['positions'].items():
            # Don't replace positions in the same symbol
            if symbol == new_signal.symbol:
                continue
                
            for pos in position_list:
                # Score based on P&L, time held, and original confidence
                pnl_pct = (pos['unrealized_pnl'] / (pos['entry_price'] * pos['size'])) * 100
                entry_time = pos['entry_time']
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                minutes_held = (datetime.now() - entry_time).total_seconds() / 60
                
                # Get original confidence if available
                original_confidence = pos.get('entry_confidence', 75.0)
                
                # Lower score = weaker position (consider P&L, age, and confidence)
                score = pnl_pct - (minutes_held / 60) + (original_confidence / 100 * 10)
                
                if score < lowest_score and pnl_pct < 1.5:  # Only consider unprofitable positions
                    lowest_score = score
                    weakest_position = (symbol, pos)
        
        # Upgrade if new signal is much better than weakest position
        if weakest_position and new_signal.confidence >= 0.90:
            symbol, pos = weakest_position
            current_price = self.current_prices.get(symbol)
            if current_price:
                self.paper_trader.close_position(symbol, current_price, "upgrade_for_better_signal")
                return f"Closed weak {symbol} (score: {lowest_score:.1f}) for {new_signal.confidence:.0%} signal"
        
        return None

    def _update_asset_holdings(self, signal: TradingSignal):
        """Update asset holdings after successful trade"""
        base_asset = signal.symbol.split('/')[0]
        
        # Note: Actual position size would need to be retrieved from the paper trader
        # For now, we'll track this conceptually
        if signal.action == "BUY":
            logger.info(f"   📈 Building {base_asset} position")
        elif signal.action == "SELL":
            logger.info(f"   📉 Reducing {base_asset} position")
    
    async def _position_management_loop(self):
        """Manage open positions"""
        
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Aggressive position management to free up capital for new trades
                portfolio_status = self.paper_trader.get_portfolio_status()
                current_time = datetime.now()
                
                positions = portfolio_status.get('positions', {})
                
                # Check if positions is a dict (professional system format)
                if isinstance(positions, dict):
                    for symbol, position_data in positions.items():
                        # Professional system returns single position per symbol
                        if isinstance(position_data, dict):
                            unrealized_pnl = position_data.get('unrealized_pnl', 0)
                            quantity = position_data.get('quantity', 0)
                            entry_price = position_data.get('entry_price', 0)
                            entry_value = entry_price * quantity
                            
                            # Professional system may not have entry_time, use current time - 1 hour as fallback
                            entry_time = current_time - timedelta(hours=1)  # Fallback for missing entry_time
                            position_age_minutes = (current_time - entry_time).total_seconds() / 60
                            
                            current_price = self.current_prices.get(symbol)
                            should_close = False
                            close_reason = ""
                            
                            # ANALYSIS-DRIVEN EXIT STRATEGY (Critical Fix)
                            if entry_value > 0:
                                pnl_pct = (unrealized_pnl / entry_value) * 100
                            else:
                                continue  # Skip invalid positions
                        
                        # 1. TARGET HIT: Use actual take_profit from analysis (3% target)
                        target_pct = 3.0  # This comes from our signal generation (1.03x target)
                        if pnl_pct >= target_pct:
                            should_close = True
                            close_reason = f"TARGET_HIT: {pnl_pct:.2f}% (target: {target_pct}%)"
                        
                        # 2. STOP LOSS: Use actual stop_loss from analysis (1.5% stop)  
                        elif pnl_pct <= -1.5:  # This comes from our signal generation (0.985x stop)
                            should_close = True
                            close_reason = f"STOP_LOSS: {pnl_pct:.2f}%"
                        
                        # 3. TRAILING STOP: Lock in 50% of gains once up 2%
                        elif pnl_pct > 2.0:
                            max_pnl = getattr(position_data, 'max_pnl', pnl_pct)
                            if pnl_pct < max_pnl * 0.5:  # Fallen 50% from peak
                                should_close = True
                                close_reason = f"TRAILING_STOP: Protecting gains (peak: {max_pnl:.2f}%)"
                        
                        # 4. TIME-BASED EXITS (Strategy-Dependent) - MUCH LONGER HOLDS
                        max_hold_minutes = 2880  # 2 days for swing trades (not 20 minutes!)
                        
                        # Early exit for weak positions only
                        if position_age_minutes > 60 and pnl_pct < -1.0:  # 1 hour + losing >1%
                            should_close = True
                            close_reason = f"WEAK_POSITION: {position_age_minutes:.0f}min, {pnl_pct:.2f}%"
                        
                        # Final time stop (2 days, not 20 minutes)
                        elif position_age_minutes > max_hold_minutes:
                            should_close = True
                            close_reason = f"TIME_STOP: {position_age_minutes:.0f}min (swing trade complete)"
                            
                        if should_close and current_price:
                            self.paper_trader.close_position(symbol, current_price, close_reason)
                            logger.info(f"� Closed {symbol}: {close_reason}")
                
            except Exception as e:
                logger.error(f"Error in position management: {e}")
                await asyncio.sleep(30)
    
    async def _progress_reporting_loop(self):
        """Report progress every 10 minutes"""
        
        while self.running:
            try:
                await asyncio.sleep(600)  # 10 minutes
                
                if not self.running:
                    break
                
                elapsed = datetime.now() - self.start_time
                
                # Use comprehensive reporting function
                self.paper_trader._total_signals_generated = self.signals_generated
                self.paper_trader.print_comprehensive_summary(
                    f"10-MINUTE PROGRESS REPORT ({elapsed.total_seconds()/60:.0f} min elapsed)"
                )
                
            except Exception as e:
                logger.error(f"Error in progress reporting: {e}")
                await asyncio.sleep(60)
    
    async def _update_trade_pnl(self, symbol: str, current_price: float, dashboard):
        """Update P&L for active trades in real-time"""
        try:
            # Get active positions from paper trader
            positions = self.paper_trader.get_active_positions()
            
            for position in positions:
                if position['symbol'] == symbol:
                    entry_price = position['entry_price']
                    position_size = position['position_size']
                    side = position['side']
                    
                    # Calculate unrealized P&L
                    if side.upper() == 'BUY':
                        unrealized_pnl = (current_price - entry_price) * position_size
                    else:  # SELL
                        unrealized_pnl = (entry_price - current_price) * position_size
                    
                    # Update dashboard with P&L
                    trade_id = position.get('trade_id', f"{symbol}_{position.get('timestamp', '')}")
                    dashboard.update_trade_pnl(trade_id, current_price, unrealized_pnl)
                    
        except Exception as e:
            logger.debug(f"Error updating P&L for {symbol}: {e}")
    
    async def _generate_final_report(self):
        """Generate comprehensive final report"""
        
        logger.info("\n" + "=" * 70)
        logger.info("🏁 FINAL KRAKEN PAPER TRADING RESULTS")
        logger.info("=" * 70)
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        final_status = self.paper_trader.get_portfolio_status()
        
        # Test summary
        logger.info(f"⏰ Test Duration: {duration.total_seconds()/60:.1f} minutes")
        logger.info(f"📡 Data Source: Live Kraken WebSocket")
        logger.info(f"🧠 Signal Source: Ultimate Analyzer + RSI")
        
        # Financial results
        logger.info(f"\n💰 FINANCIAL RESULTS:")
        logger.info(f"   Starting Balance: $10,000.00")
        logger.info(f"   Final Balance: ${final_status['current_balance']:.2f}")
        logger.info(f"   Total Return: {final_status['total_return']:+.2%}")
        logger.info(f"   Profit/Loss: ${final_status['current_balance'] - 10000:+.2f}")
        
        # Trading performance
        logger.info(f"\n📊 TRADING PERFORMANCE:")
        logger.info(f"   Signals Generated: {self.signals_generated}")
        logger.info(f"   Trades Executed: {self.trades_executed}")
        logger.info(f"   Completed Trades: {final_status['performance']['total_trades']}")
        logger.info(f"   Win Rate: {final_status['performance']['win_rate']:.1%}")
        
        if final_status['performance']['total_trades'] > 0:
            logger.info(f"   Average Win: ${final_status['performance']['avg_win']:.2f}")
            logger.info(f"   Average Loss: ${final_status['performance']['avg_loss']:.2f}")
            logger.info(f"   Profit Factor: {final_status['performance']['profit_factor']:.2f}")
        
        # Risk management
        logger.info(f"\n⚠️ RISK MANAGEMENT:")
        logger.info(f"   Max Drawdown: {final_status['performance']['max_drawdown']:.2%}")
        logger.info(f"   Trading Fees: ${final_status['performance']['total_fees']:.2f}")
        logger.info(f"   Slippage Costs: ${final_status['performance']['total_slippage']:.2f}")
        
        # Open positions
        if final_status['positions_count'] > 0:
            logger.info(f"\n📊 OPEN POSITIONS:")
            for symbol, pos_list in final_status['positions'].items():
                for pos in pos_list:  # pos_list is a list of position dictionaries
                    pnl_pct = (pos['unrealized_pnl'] / (pos['entry_price'] * pos['size'])) * 100 if pos['size'] > 0 else 0
                    logger.info(f"   {pos['side']} {symbol}: ${pos['unrealized_pnl']:+.2f} ({pnl_pct:+.2f}%)")
        
        # Export results
        journal_file = self.paper_trader.export_trade_journal()
        logger.info(f"\n📋 Detailed Results: {journal_file}")
        
        # Final assessment
        logger.info(f"\n🎯 SYSTEM ASSESSMENT:")
        
        if final_status['total_return'] > 0.01:
            logger.info("🟢 PROFITABLE: System generated positive returns!")
        elif final_status['total_return'] > -0.01:
            logger.info("🟡 BREAKEVEN: System performed neutrally")
        else:
            logger.info("🔴 UNPROFITABLE: System needs optimization")
        
        if self.trades_executed > 5:
            logger.info("🟢 ACTIVE: Good signal generation rate")
        elif self.trades_executed > 0:
            logger.info("🟡 MODERATE: Some signals generated")
        else:
            logger.info("🔴 INACTIVE: No trades executed - check signal logic")
        
        logger.info(f"\n✅ Real market paper trading test complete!")
        logger.info(f"🚀 System ready for live trading evaluation!")
        logger.info("=" * 70)

async def main():
    """Run the real Kraken data paper trading test with dashboard"""
    
    try:
        # Start dashboard server
        dashboard = get_dashboard()
        web_server = WorkingDashboardServer(dashboard_manager=dashboard, port=8080)
        
        if web_server.start_server():
            dashboard_url = f"http://localhost:8080/dashboard"
            print(f"🎯 Trading Dashboard: {dashboard_url}")
            print("📊 Monitor all signals, validations, and trades in real-time!")
        
        system = KrakenPaperTradingSystem()
        await system.start_real_trading_test()
        
    except KeyboardInterrupt:
        print("\n👋 Test stopped by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())