#!/usr/bin/env python3
"""
Test script for the multi-pair Kraken scanner
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.multi_pair_kraken_scanner import MultiPairKrakenScanner

async def test_scanner():
    config = {
        'scanner': {'min_volume_usdt': 50000, 'max_spread_pct': 1.0},
        'filters': {'confluence': {'min_confluence_score': 50}}
    }
    
    print("🔍 Testing Multi-Pair Kraken Scanner")
    scanner = MultiPairKrakenScanner(config)
    result = await scanner.scan_all_liquid_pairs()
    
    print(f"\n✅ Scan Results:")
    print(f"   Pairs analyzed: {result['pairs_analyzed']}")
    print(f"   Liquid pairs: {result['liquid_pairs']}")
    print(f"   Valid setups: {len([r for r in result['rankings'] if r.setup_valid])}")
    
    if result['best_setup']:
        best = result['best_setup']
        print(f"\n🎯 Best Setup Found:")
        print(f"   Symbol: {best['symbol']}")
        print(f"   Entry: ${best['entry_price']:.4f}")
        print(f"   Score: {best['confluence_score']}/100")
    else:
        print("\n❌ No valid setups found")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_scanner())