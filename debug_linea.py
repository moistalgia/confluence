#!/usr/bin/env python3
"""
LINEA/USD Debug Script
Test LINEA/USD analysis to identify the specific failure
"""

import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ultimate_crypto_analyzer import UltimateCryptoAnalyzer

def test_linea_analysis():
    """Test LINEA/USD analysis with detailed error handling"""
    
    print("🔍 DEBUGGING LINEA/USD ANALYSIS")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        print("1. Initializing Ultimate Crypto Analyzer...")
        analyzer = UltimateCryptoAnalyzer()
        print("   ✅ Analyzer initialized")
        
        # Test the specific symbol
        symbol = "LINEA/USD"
        print(f"\n2. Testing {symbol} analysis...")
        
        # Run analysis with detailed error catching
        analysis = analyzer.analyze_symbol(symbol)
        
        if analysis:
            print("   ✅ Analysis completed successfully!")
            
            # Check key components
            ultimate_score = analysis.get('ultimate_score', {})
            composite_score = ultimate_score.get('composite_score', 0)
            
            print(f"   Score: {composite_score}/100")
            
            # Check for DCA analysis
            dca_analysis = ultimate_score.get('dca_analysis', {})
            if dca_analysis:
                print(f"   DCA: {dca_analysis.get('recommendation', 'N/A')}")
            
            # Check multi-timeframe
            if 'multi_timeframe_analysis' in analysis:
                print("   ✅ Multi-timeframe analysis: PRESENT")
            else:
                print("   ❌ Multi-timeframe analysis: MISSING")
                
            # Check volume profile
            if 'volume_profile_analysis' in analysis:
                print("   ✅ Volume profile analysis: PRESENT")
            else:
                print("   ❌ Volume profile analysis: MISSING")
                
            return True
            
        else:
            print("   ❌ Analysis returned None")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        print(f"\n🔍 Full traceback:")
        print("-" * 50)
        traceback.print_exc()
        print("-" * 50)
        return False

if __name__ == "__main__":
    success = test_linea_analysis()
    
    if success:
        print(f"\n🎉 LINEA/USD analysis works - issue was likely in pipeline")
    else:
        print(f"\n💡 LINEA/USD analysis failed - see error above")
        print(f"   Recommendation: Use a different pair or fix the specific issue")