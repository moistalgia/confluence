#!/usr/bin/env python3
"""
Debug test for transparency dashboard logging
"""

import asyncio
import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.multi_pair_kraken_scanner import MultiPairKrakenScanner
from core.transparency_dashboard import ScanningTransparencyDashboard

async def debug_transparency():
    print("🔍 DEBUG: Testing transparency logging")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run a scan
    scanner = MultiPairKrakenScanner(config.get('paper_trading', {}))
    scan_results = await scanner.scan_all_liquid_pairs()
    
    print(f"✅ Scan completed:")
    print(f"   Pairs analyzed: {scan_results['pairs_analyzed']}")
    print(f"   Rankings type: {type(scan_results['rankings'])}")
    print(f"   Rankings length: {len(scan_results['rankings'])}")
    if scan_results['rankings']:
        print(f"   First ranking type: {type(scan_results['rankings'][0])}")
        print(f"   First ranking: {scan_results['rankings'][0]}")
    
    # Try to log it
    dashboard = ScanningTransparencyDashboard('debug_transparency.db')
    
    try:
        scan_id = dashboard.log_scan_results(scan_results)
        print(f"✅ Logged successfully with scan_id: {scan_id}")
        
        # Try to generate report
        report = dashboard.generate_transparency_report()
        print(f"✅ Report generated:")
        print(report[:500] + "..." if len(report) > 500 else report)
        
    except Exception as e:
        print(f"❌ Error logging scan results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_transparency())