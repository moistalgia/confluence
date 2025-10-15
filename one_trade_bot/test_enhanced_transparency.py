#!/usr/bin/env python3
"""
Test the enhanced transparency report with filter breakdown
"""
import asyncio
import sys
import os
from pathlib import Path

# Set working directory and path
project_dir = Path(r"c:\Dev\crypto-analyzer\one_trade_bot")
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

import sqlite3
from core.transparency_dashboard import ScanningTransparencyDashboard

def test_enhanced_report():
    """Test the enhanced transparency report format"""
    print("🧪 Testing enhanced transparency report...")
    
    db_path = 'paper_trading.db'
    
    # Check if we have scan data
    conn = sqlite3.connect(db_path)
    scan_count = conn.execute('SELECT COUNT(*) FROM scan_results').fetchone()[0]
    pair_count = conn.execute('SELECT COUNT(*) FROM pair_analysis').fetchone()[0]
    filter_count = conn.execute('SELECT COUNT(*) FROM filter_breakdown').fetchone()[0]
    
    print(f"📊 Database status:")
    print(f"   Scans: {scan_count}")
    print(f"   Pair analyses: {pair_count}")
    print(f"   Filter breakdowns: {filter_count}")
    
    if scan_count > 0:
        # Generate enhanced transparency report
        transparency = ScanningTransparencyDashboard(db_path)
        report = transparency.generate_transparency_report()
        
        if report:
            print(f"\n📋 ENHANCED TRANSPARENCY REPORT ({len(report)} chars):")
            print("=" * 60)
            # Print ASCII-safe version
            safe_report = report.encode('ascii', 'ignore').decode('ascii')
            print(safe_report)
            print("=" * 60)
        else:
            print("❌ No report generated")
    else:
        print("⚠️ No scan data available - run a daily scan first")
    
    conn.close()

if __name__ == "__main__":
    test_enhanced_report()