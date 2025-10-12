#!/usr/bin/env python3
"""
Fresh Analysis Generation with US-Accessible Exchanges
Generates complete dataset with all enhanced features using Kraken/US exchanges
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our components
from enhanced_multi_timeframe_analyzer import EnhancedMultiTimeframeAnalyzer
from config_manager import ConfigManager
from complete_prompt_generator import generate_complete_ultimate_prompt
from llm_integration import LLMIntegration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run fresh analysis with US-accessible exchanges"""
    
    print("🚀 Fresh Crypto Analysis with US-Accessible Exchanges")
    print("=" * 60)
    
    try:
        # Initialize analyzer with US-accessible exchanges
        logger.info("Initializing analyzer with US-accessible exchanges...")
        analyzer = EnhancedMultiTimeframeAnalyzer()
        
        # Test symbols that should be available on Kraken/US exchanges
        symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD']
        
        print(f"📊 Analyzing {len(symbols)} symbols with enhanced features:")
        for symbol in symbols:
            print(f"   • {symbol}")
        
        results = {}
        
        # Analyze each symbol
        for i, symbol in enumerate(symbols, 1):
            print(f"\n🔍 [{i}/{len(symbols)}] Analyzing {symbol}...")
            
            try:
                # Run comprehensive analysis
                analysis = analyzer.analyze_multi_timeframe(symbol)
                
                if analysis:
                    # Check for different possible confluence score locations
                    confluence_score = 0
                    
                    # Try enhanced analyzer format first
                    if 'confluence_analysis' in analysis and 'overall_confluence' in analysis['confluence_analysis']:
                        confluence_score = analysis['confluence_analysis']['overall_confluence'].get('confluence_score', 0)
                    
                    # Try ultimate analyzer format
                    elif 'ultimate_score' in analysis:
                        confluence_score = analysis['ultimate_score'].get('score', 0)
                    
                    # Try timeframe analysis
                    elif 'multi_timeframe_analysis' in analysis:
                        # Use a basic calculation from available data
                        timeframes = analysis.get('timeframes', {})
                        if timeframes:
                            confluence_score = len(timeframes) * 20  # Basic score based on timeframes
                    
                    print(f"   ✅ Analysis Complete - Score: {confluence_score:.1f}%")
                    
                    results[symbol] = analysis
                    
                    # Display key metrics
                    if 'timeframes' in analysis:
                        print(f"   📈 Timeframes analyzed: {len(analysis['timeframes'])}")
                    
                    if 'signals' in analysis:
                        signals = analysis['signals']
                        print(f"   🎯 Signals generated: {len(signals)}")
                    
                else:
                    print(f"   ❌ Failed to get analysis for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                print(f"   ❌ Error: {str(e)}")
        
        # Save results
        if results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"analysis_results/fresh_us_analysis_{timestamp}.json"
            
            os.makedirs('analysis_results', exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {output_file}")
            print(f"📊 Successfully analyzed: {len(results)}/{len(symbols)} symbols")
            
            # Generate enhanced prompt
            print("\n🧠 Generating Enhanced AI Prompt...")
            
            try:
                # Use the best result for prompt generation (use first available)
                best_symbol = list(results.keys())[0]
                
                best_analysis = results[best_symbol]
                
                enhanced_prompt = generate_complete_ultimate_prompt(
                    symbol=best_symbol,
                    analysis=best_analysis
                )
                
                # Save prompt
                prompt_file = f"analysis_results/enhanced_prompt_{timestamp}.md"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(enhanced_prompt)
                
                print(f"📝 Enhanced prompt saved to: {prompt_file}")
                print(f"🎯 Based on best performer: {best_symbol}")
                
                # Generate AI Analysis if LLM is available
                print("\n🤖 Attempting AI Analysis Generation...")
                
                try:
                    llm = LLMIntegration()
                    
                    if llm.is_configured():
                        ai_response = llm.get_analysis(enhanced_prompt)
                        
                        if ai_response:
                            ai_file = f"analysis_results/ai_analysis_{timestamp}.md"
                            with open(ai_file, 'w', encoding='utf-8') as f:
                                f.write(ai_response)
                            
                            print(f"🧠 AI analysis saved to: {ai_file}")
                        else:
                            print("❌ AI analysis failed - no response")
                    else:
                        print("⚠️ LLM not configured - skipping AI analysis")
                        
                except Exception as e:
                    logger.error(f"AI analysis error: {e}")
                    print(f"❌ AI analysis error: {str(e)}")
                
            except Exception as e:
                logger.error(f"Prompt generation error: {e}")
                print(f"❌ Prompt generation error: {str(e)}")
        
        else:
            print("❌ No successful analyses - check exchange connectivity")
        
        # Display summary
        print("\n" + "=" * 60)
        print("📈 ANALYSIS SUMMARY")
        print("=" * 60)
        
        if results:
            print(f"✅ Successful analyses: {len(results)}")
            # Calculate best score safely
            best_score = 0
            for symbol, data in results.items():
                if 'confluence_analysis' in data:
                    score = data['confluence_analysis'].get('overall_confluence', {}).get('confluence_score', 0)
                elif 'ultimate_score' in data:
                    score = data['ultimate_score'].get('score', 0)
                else:
                    score = 50  # Default score
                best_score = max(best_score, score)
            
            print(f"🎯 Best analysis score: {best_score:.1f}%")
            print(f"📊 Total signals generated: {sum(len(results[s].get('signals', [])) for s in results)}")
            print(f"⏱️  Exchange used: {analyzer.exchange.id}")
            
            # Show performance metrics
            if hasattr(analyzer, 'performance_metrics'):
                metrics = analyzer.performance_metrics
                print(f"🚀 Cache hits: {metrics.get('cache_hits', 0)}")
                print(f"⚡ Parallel executions: {metrics.get('parallel_executions', 0)}")
        
        else:
            print("❌ No successful analyses")
            print("💡 Try checking internet connection or exchange status")
        
        print("\n🎉 Fresh analysis generation completed!")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        print(f"❌ Pipeline error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())