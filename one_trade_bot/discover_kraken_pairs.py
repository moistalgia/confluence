#!/usr/bin/env python3
"""
Kraken Pairs Discovery - Find all tradeable pairs
"""

import ccxt
import json
from datetime import datetime

def discover_kraken_pairs():
    """Discover all available Kraken trading pairs"""
    
    try:
        # Initialize Kraken exchange
        kraken = ccxt.kraken({'enableRateLimit': True})
        
        print("=== KRAKEN PAIRS DISCOVERY ===")
        print(f"Timestamp: {datetime.now()}")
        
        # Fetch all markets
        markets = kraken.load_markets()
        
        # Filter for USDT pairs (most liquid)
        usdt_pairs = []
        usd_pairs = []
        
        for symbol, market in markets.items():
            if '/USDT' in symbol and market['active']:
                usdt_pairs.append({
                    'symbol': symbol,
                    'base': market['base'],
                    'quote': market['quote'],
                    'active': market['active'],
                    'spot': market['spot'],
                })
            elif '/USD' in symbol and market['active'] and 'USDT' not in symbol:
                usd_pairs.append({
                    'symbol': symbol,
                    'base': market['base'], 
                    'quote': market['quote'],
                    'active': market['active'],
                    'spot': market['spot'],
                })
        
        print(f"\n📊 USDT PAIRS FOUND: {len(usdt_pairs)}")
        for pair in sorted(usdt_pairs, key=lambda x: x['symbol']):
            print(f"  {pair['symbol']}")
            
        print(f"\n💵 USD PAIRS FOUND: {len(usd_pairs)}")  
        for pair in sorted(usd_pairs, key=lambda x: x['symbol']):
            print(f"  {pair['symbol']}")
            
        # Test fetching ticker for a few pairs to verify liquidity
        print(f"\n🧪 TESTING LIQUIDITY (Sample Pairs):")
        test_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'] if usdt_pairs else ['BTC/USD', 'ETH/USD']
        
        liquid_pairs = []
        for pair in test_pairs:
            try:
                ticker = kraken.fetch_ticker(pair)
                volume_24h = ticker.get('quoteVolume', 0)
                spread = (ticker['ask'] - ticker['bid']) / ticker['bid'] * 100
                
                print(f"  {pair}:")
                print(f"    Price: ${ticker['last']:,.2f}")
                print(f"    24h Volume: ${volume_24h:,.0f}")
                print(f"    Spread: {spread:.3f}%")
                
                # Consider liquid if >$1M daily volume and <0.5% spread
                if volume_24h > 1000000 and spread < 0.5:
                    liquid_pairs.append(pair)
                    print(f"    ✅ LIQUID")
                else:
                    print(f"    ⚠️  Low liquidity")
                    
            except Exception as e:
                print(f"  {pair}: ❌ Error - {e}")
                
        print(f"\n🎯 RECOMMENDED WATCHLIST ({len(liquid_pairs)} pairs):")
        for pair in liquid_pairs:
            print(f"  - {pair}")
            
        # Save configuration
        config = {
            'kraken_pairs': {
                'usdt_pairs': [p['symbol'] for p in usdt_pairs],
                'usd_pairs': [p['symbol'] for p in usd_pairs],
                'liquid_pairs': liquid_pairs,
                'updated': datetime.now().isoformat()
            }
        }
        
        with open('kraken_pairs.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"\n✅ Pairs data saved to kraken_pairs.json")
        return liquid_pairs
        
    except Exception as e:
        print(f"❌ Error discovering Kraken pairs: {e}")
        return []

if __name__ == "__main__":
    pairs = discover_kraken_pairs()