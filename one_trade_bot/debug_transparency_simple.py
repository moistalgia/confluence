import asyncio
import sqlite3
import os
import sys
import yaml
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

async def debug_transparency_issue():
    """Debug the transparency reporting issue"""
    print("🔍 DEBUGGING TRANSPARENCY ISSUE")
    print("=" * 50)
    
    # 1. Check database file
    db_path = 'paper_trading.db'
    print(f"1️⃣ Database file check:")
    print(f"   Path: {db_path}")
    print(f"   Exists: {os.path.exists(db_path)}")
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"   Tables: {[t[0] for t in tables]}")
        
        if 'scan_results' in [t[0] for t in tables]:
            count = conn.execute('SELECT COUNT(*) FROM scan_results').fetchone()[0]
            print(f"   Scan results count: {count}")
            
            if count > 0:
                recent = conn.execute('SELECT id, scan_timestamp, total_pairs FROM scan_results ORDER BY scan_timestamp DESC LIMIT 3').fetchall()
                print(f"   Recent scans: {recent}")
        
        if 'pair_analysis' in [t[0] for t in tables]:
            count = conn.execute('SELECT COUNT(*) FROM pair_analysis').fetchone()[0]
            print(f"   Pair analysis count: {count}")
        
        conn.close()
    
    # 2. Test scanner directly
    print(f"\n2️⃣ Testing scanner directly:")
    try:
        # Load config
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        from core.multi_pair_kraken_scanner import MultiPairKrakenScanner
        scanner = MultiPairKrakenScanner(config)
        
        print("   Running scanner...")
        scan_result = await scanner.scan_all_liquid_pairs()
        
        if scan_result:
            print(f"   ✅ Scanner success!")
            print(f"   Valid setups: {len(scan_result.get('valid_setups', []))}")
            if scan_result.get('best_setup'):
                best = scan_result['best_setup']
                print(f"   Best: {best['symbol']} (Score: {best['confluence_score']})")
        else:
            print("   ❌ Scanner returned None")
            
    except Exception as e:
        print(f"   ❌ Scanner error: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Test transparency directly
    print(f"\n3️⃣ Testing transparency directly:")
    try:
        from core.transparency_dashboard import ScanningTransparencyDashboard
        transparency = ScanningTransparencyDashboard(db_path)
        
        if 'scan_result' in locals() and scan_result:
            print("   Logging scan result...")
            scan_id = transparency.log_scan_results(scan_result)
            print(f"   ✅ Logged with ID: {scan_id}")
        
        print("   Generating report...")
        report = transparency.generate_transparency_report()
        
        if report and len(report) > 100:
            print(f"   ✅ Report generated ({len(report)} chars)")
            print("   First 300 characters:")
            print("   " + "-" * 50)
            print("   " + report[:300])
            print("   " + "-" * 50)
        else:
            print(f"   ❌ Report failed or empty: {report}")
            
    except Exception as e:
        print(f"   ❌ Transparency error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_transparency_issue())