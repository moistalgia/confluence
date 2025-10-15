#!/usr/bin/env python3
"""
Test the transparency dashboard functionality
"""

import asyncio
import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.disciplined_trading_engine import DisciplinedTradingEngine
from core.transparency_dashboard import print_transparency_report

async def test_transparency():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize engine
    engine = DisciplinedTradingEngine(
        config=config.get('paper_trading', {}),
        data_provider=None,
        db_path='transparency_test.db'
    )
    
    print("🔍 Testing Transparency Dashboard")
    
    # Run a scan to generate data
    print("\n🌅 Running scan with transparency logging...")
    await engine._run_daily_scan()
    
    # Generate and display transparency report
    print("\n📊 TRANSPARENCY REPORT:")
    print("=" * 60)
    print_transparency_report('transparency_test.db')
    
    # Show recent scans
    print("\n📅 RECENT SCANS:")
    print("-" * 30)
    recent_scans = engine.transparency.get_recent_scans(5)
    for scan in recent_scans:
        print(f"Scan {scan['scan_id']}: {scan['pairs_analyzed']} pairs, "
              f"Best: {scan['best_symbol']} ({scan['best_score']}/100)")

if __name__ == "__main__":
    asyncio.run(test_transparency())