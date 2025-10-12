#!/usr/bin/env python3
"""
Custom Crypto Analysis Pipeline
Run complete analysis pipeline on specific trading pairs for opportunity discovery

Features:
- Custom pair selection (including LINEA/USD)
- Kraken top pairs for opportunity discovery  
- Complete enhanced analysis with DCA scoring
- Professional AI prompts generation
"""

import asyncio
import logging
from pathlib import Path
from typing import List
from datetime import datetime

# Import our pipeline components
from complete_pipeline import CryptoAnalysisPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define trading pairs for different strategies - VERIFIED KRAKEN PAIRS ONLY
CUSTOM_PAIRS = {
    'specific_request': ['XRP/USD'],  # User requested LINEA/USD
    # 'LINEA/USD', 'MANA/USD', 'XRP/USD', 'BTC/USD', 'NEAR/USD'
    'kraken_top_majors': [
        # Verified top pairs on Kraken (removed MATIC/USD - not available)
        #'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD',
        #'AVAX/USD', 'ATOM/USD', 'ALGO/USD', 'SOL/USD'
    ],
    
    'opportunity_discovery': [
        # Verified high-potential altcoins on Kraken  
        #'NEAR/USD', 'MANA/USD', 'SAND/USD', 'UNI/USD', 'AAVE/USD', 'CRV/USD',
        #'FIL/USD', 'LTC/USD', 'XRP/USD'
    ],
    
    'defi_focus': [
        # Verified DeFi tokens for advanced analysis
        #'AAVE/USD', 'UNI/USD', 'CRV/USD'
    ]
}

async def run_custom_analysis(pair_group: str, pairs: List[str]):
    """Run complete analysis pipeline on custom pairs"""
    
    print(f"\n{'='*80}")
    print(f"🎯 CUSTOM CRYPTO ANALYSIS PIPELINE")
    print(f"   Group: {pair_group.upper().replace('_', ' ')}")
    print(f"   Pairs: {len(pairs)} symbols")
    print(f"{'='*80}")
    
    for pair in pairs:
        print(f"   • {pair}")
    print()
    
    # Initialize pipeline with custom pairs
    pipeline = CryptoAnalysisPipeline(symbols=pairs)
    
    # Run complete analysis
    start_time = datetime.now()
    
    try:
        results = await pipeline.run_complete_analysis(force_refresh=True)
        
        # Print summary
        duration = datetime.now() - start_time
        print(f"\n{'='*80}")
        print(f"📊 ANALYSIS COMPLETE - {pair_group.upper()}")
        print(f"   Duration: {duration.total_seconds():.1f} seconds")
        print(f"   Success: {results['success_count']}/{results['total_symbols']}")
        print(f"   Errors: {results['error_count']}")
        print(f"{'='*80}")
        
        # Show key insights for each pair
        for symbol, result in results['results'].items():
            if result['status'] == 'success' and 'analysis' in result:
                analysis = result['analysis']
                ultimate_score = analysis.get('ultimate_score', {})
                composite_score = ultimate_score.get('composite_score', 0)
                
                # Get DCA recommendation if available
                dca_analysis = ultimate_score.get('dca_analysis', {})
                dca_recommendation = dca_analysis.get('recommendation', 'N/A')
                
                # Get bias from signals
                signals = analysis.get('enhanced_trading_signals', {})
                primary_bias = signals.get('primary_bias', 'NEUTRAL')
                
                print(f"\n🔍 {symbol}:")
                print(f"   Score: {composite_score}/100")
                print(f"   Bias: {primary_bias}")
                print(f"   DCA: {dca_recommendation}")
                
                # Show volume confirmation fix in action
                if 'multi_timeframe_analysis' in analysis:
                    mtf = analysis['multi_timeframe_analysis']
                    if 'volume_analysis' in mtf:
                        vol_conf = mtf['volume_analysis'].get('volume_confirmation', 'N/A')
                        weighted_score = mtf['volume_analysis'].get('weighted_volume_score', 0)
                        print(f"   Volume: {vol_conf} ({weighted_score:.2f}x)")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed for {pair_group}: {e}")
        return None

async def main():
    """Run custom analysis pipelines"""
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         🚀 CUSTOM CRYPTO ANALYSIS - OPPORTUNITY DISCOVERY     ║")
    print("║                    Enhanced Pipeline v2.0                     ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Show available groups
    print("📋 Available Analysis Groups:")
    for group, pairs in CUSTOM_PAIRS.items():
        print(f"   {group}: {len(pairs)} pairs")
    print()
    
    # 1. Run LINEA/USD specific analysis (user request)
    print("🎯 STEP 1: Specific Request - LINEA/USD")
    linea_results = await run_custom_analysis('specific_request', CUSTOM_PAIRS['specific_request'])
    
    # 2. Run Kraken top majors for comparison
    print("\n🏆 STEP 2: Kraken Top Majors Analysis")
    majors_results = await run_custom_analysis('kraken_top_majors', CUSTOM_PAIRS['kraken_top_majors'])
    
    # 3. Run opportunity discovery on altcoins
    print("\n💎 STEP 3: Opportunity Discovery - Altcoins")
    discovery_results = await run_custom_analysis('opportunity_discovery', CUSTOM_PAIRS['opportunity_discovery'])
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 CUSTOM ANALYSIS PIPELINE COMPLETE")
    print(f"{'='*80}")
    
    total_pairs = len(CUSTOM_PAIRS['specific_request']) + len(CUSTOM_PAIRS['kraken_top_majors']) + len(CUSTOM_PAIRS['opportunity_discovery'])
    
    print(f"📊 Total Analysis:")
    print(f"   • LINEA/USD: {'✅' if linea_results else '❌'}")
    print(f"   • Top Majors: {'✅' if majors_results else '❌'}")  
    print(f"   • Alt Discovery: {'✅' if discovery_results else '❌'}")
    print(f"   • Total Pairs: {total_pairs}")
    
    print(f"\n💡 Key Enhancements Active:")
    print(f"   ✅ Fixed Volume Confirmation (LOW_VOLUME vs CONFIRMATION)")
    print(f"   ✅ DCA Timing Analysis (0-15 point scoring)")
    print(f"   ✅ Enhanced Ultimate Scoring (+19% improvement)")
    print(f"   ✅ 14KB Complete Prompts for AI analysis")
    print(f"   ✅ Multi-timeframe confluence analysis")
    
    print(f"\n📁 Results saved to: output/ultimate_analysis/")
    print(f"🤖 Copy AI prompts from output files to ChatGPT/Claude for trading insights!")

if __name__ == "__main__":
    # Run the custom analysis pipeline
    results = asyncio.run(main())
    print("\n🎉 Custom analysis pipeline execution complete!")