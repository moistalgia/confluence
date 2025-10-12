#!/usr/bin/env python3
"""Debug stochastic calculation"""

import pandas as pd
import ta
from enhanced_multi_timeframe_analyzer import EnhancedMultiTimeframeAnalyzer

# Test stochastic calculation
analyzer = EnhancedMultiTimeframeAnalyzer()

# Get some test data
import ccxt
exchange = ccxt.kraken()
ohlcv = exchange.fetch_ohlcv('BTC/USD', '1d', limit=100)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

print(f"Data shape: {df.shape}")
print(f"High/Low/Close available: {df['high'].iloc[-1]}, {df['low'].iloc[-1]}, {df['close'].iloc[-1]}")

# Test different stochastic approaches
try:
    # Approach 1: ta.momentum.stoch
    stoch_k = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    print(f"Approach 1 - ta.momentum.stoch: {stoch_k.iloc[-1] if not stoch_k.empty else 'EMPTY'}")
except Exception as e:
    print(f"Approach 1 ERROR: {e}")

try:
    # Approach 2: StochasticOscillator class
    stoch_osc = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    stoch_k = stoch_osc.stoch()
    stoch_d = stoch_osc.stoch_signal()
    print(f"Approach 2 - StochasticOscillator: K={stoch_k.iloc[-1] if not stoch_k.empty else 'EMPTY'}, D={stoch_d.iloc[-1] if not stoch_d.empty else 'EMPTY'}")
except Exception as e:
    print(f"Approach 2 ERROR: {e}")

# Check if indicators are enabled
timeframes = analyzer.timeframes
for tf, config in timeframes.items():
    print(f"{tf}: enabled_indicators = {config.get('indicators_enabled', [])}")