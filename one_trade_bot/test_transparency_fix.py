#!/usr/bin/env python3
"""
Quick test of transparency fix
"""
import asyncio
import sys
import os
from pathlib import Path

# Set working directory
project_dir = Path(r"c:\Dev\crypto-analyzer\one_trade_bot")
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

import yaml
import sqlite3
from core.disciplined_trading_engine import DisciplinedTradingEngine
from core.transparency_dashboard import ScanningTransparencyDashboard

async def test_transparency_fix():
    """Test if the transparency issue is fixed"""
    print("🧪 Testing transparency fix...")
    
    # Check current database state
    db_path = 'paper_trading.db'
    conn = sqlite3.connect(db_path)
    scan_count = conn.execute('SELECT COUNT(*) FROM scan_results').fetchone()[0]
    pair_count = conn.execute('SELECT COUNT(*) FROM pair_analysis').fetchone()[0]
    print(f"📊 Before test - Scans: {scan_count}, Pairs: {pair_count}")
    conn.close()
    
    try:
        # Load config
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Create engine
        engine = DisciplinedTradingEngine(config, None)
        
        # Run daily scan
        print("🔍 Running daily scan...")
        await engine._run_daily_scan()
        
        # Check database after scan
        conn = sqlite3.connect(db_path)
        scan_count_after = conn.execute('SELECT COUNT(*) FROM scan_results').fetchone()[0]
        pair_count_after = conn.execute('SELECT COUNT(*) FROM pair_analysis').fetchone()[0]
        print(f"📊 After scan - Scans: {scan_count_after}, Pairs: {pair_count_after}")
        
        if scan_count_after > scan_count:
            print("✅ SUCCESS! New scan data logged")
            
            # Test transparency report
            transparency = ScanningTransparencyDashboard(db_path)
            report = transparency.generate_transparency_report()
            
            if report and len(report) > 100:
                print(f"✅ SUCCESS! Transparency report: {len(report)} chars")
                return True
            else:
                print(f"❌ Report failed: {report[:50] if report else 'None'}")
                return False
        else:
            print("❌ No new scan data logged")
            return False
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_transparency_fix())