import asyncio
import yaml
from core.multi_pair_kraken_scanner import MultiPairKrakenScanner
from core.transparency_dashboard import ScanningTransparencyDashboard

async def test_scanner_transparency():
    """Test if scanner works and logs properly"""
    print("🧪 Testing scanner and transparency integration...")
    
    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Test scanner
    print("\n1️⃣ Testing MultiPairKrakenScanner...")
    scanner = MultiPairKrakenScanner(config)
    
    try:
        scan_result = await scanner.scan_all_liquid_pairs()
        print(f"✅ Scanner returned: {type(scan_result)}")
        if scan_result:
            print(f"   Keys: {scan_result.keys()}")
            if 'valid_setups' in scan_result:
                print(f"   Valid setups: {len(scan_result['valid_setups'])}")
            if 'best_setup' in scan_result:
                print(f"   Best setup: {scan_result['best_setup']['symbol']} (Score: {scan_result['best_setup']['confluence_score']})")
        else:
            print("❌ Scanner returned None/empty")
    except Exception as e:
        print(f"❌ Scanner error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test transparency
    print("\n2️⃣ Testing ScanningTransparencyDashboard...")
    db_path = 'paper_trading.db'
    transparency = ScanningTransparencyDashboard(db_path)
    
    try:
        scan_id = transparency.log_scan_results(scan_result)
        print(f"✅ Transparency logged scan_id: {scan_id}")
        
        # Test report
        report = transparency.generate_transparency_report()
        print(f"✅ Report generated: {len(report)} characters")
        print("First 200 chars of report:")
        print(report[:200] + "...")
        
    except Exception as e:
        print(f"❌ Transparency error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_scanner_transparency())