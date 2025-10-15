#!/usr/bin/env python3
"""
Test script for DisciplinedTradingEngine with multi-pair scanner
"""

import asyncio
import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.disciplined_trading_engine import DisciplinedTradingEngine
from core.data_provider import DataProvider

async def test_disciplined_engine():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components - use None for data_provider since we have live data
    engine = DisciplinedTradingEngine(
        config=config.get('paper_trading', {}),
        data_provider=None,
        db_path='test_trading.db'
    )
    
    print("🎯 Testing DisciplinedTradingEngine with Multi-Pair Scanner")
    
    # Test daily scan
    print("\n🌅 Running daily scan...")
    await engine._run_daily_scan()
    
    # Check results
    status = engine.get_status()
    print(f"\n📊 Engine Status:")
    print(f"   State: {status['execution_state']}")
    if engine.daily_target:
        print(f"   Target: {engine.daily_target.symbol}")
        print(f"   Score: {engine.daily_target.confluence_score}")
        print(f"   Entry Zone: ${engine.daily_target.entry_low:.4f} - ${engine.daily_target.entry_high:.4f}")
    else:
        print("   Target: No target selected (rest day)")

if __name__ == "__main__":
    asyncio.run(test_disciplined_engine())