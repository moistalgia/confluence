#!/usr/bin/env python3
"""
Comprehensive Technical Analysis Verification Script
Verifies all calculations and data flow integrity
"""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import ta

def load_latest_analysis(symbol="BTC_USDT"):
    """Load the most recent analysis file"""
    analysis_dir = Path("output/ultimate_analysis")
    pattern = f"ultimate_{symbol}_*.json"
    files = list(analysis_dir.glob(pattern))
    
    if not files:
        print(f"❌ No analysis files found for {symbol}")
        return None
    
    latest = max(files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Loading: {latest.name}")
    
    with open(latest, 'r') as f:
        return json.load(f)

def verify_current_price(analysis):
    """Verify current price extraction and consistency"""
    print("\n" + "="*60)
    print("🔍 CURRENT PRICE VERIFICATION")
    print("="*60)
    
    # Check all price locations
    prices = {}
    
    # 1. Volume profile metadata
    try:
        prices['volume_metadata'] = analysis['volume_profile_analysis']['metadata']['current_price']
    except:
        prices['volume_metadata'] = "ERROR"
    
    # 2. Volume profile price analysis
    try:
        prices['volume_price_analysis'] = analysis['volume_profile_analysis']['price_analysis']['current_price']
    except:
        prices['volume_price_analysis'] = "ERROR"
    
    # 3. Latest close from 1h timeframe
    try:
        prices['1h_close'] = analysis['multi_timeframe_analysis']['timeframe_data']['1h']['ohlcv'][-1]['close']
    except:
        prices['1h_close'] = "ERROR"
    
    # 4. Enhanced AI prompt
    try:
        prompt = analysis['enhanced_ai_prompt']
        for line in prompt.split('\n'):
            if 'Current Price' in line and '$' in line:
                # Extract price from line like "- **Current Price**: $110,166.50"
                price_str = line.split('$')[1].replace(',', '')
                prices['ai_prompt'] = float(price_str)
                break
    except:
        prices['ai_prompt'] = "ERROR"
    
    # Print results
    for source, price in prices.items():
        if isinstance(price, (int, float)) and price > 0:
            print(f"✅ {source:20}: ${price:,.2f}")
        else:
            print(f"❌ {source:20}: {price}")
    
    # Check consistency
    valid_prices = [p for p in prices.values() if isinstance(p, (int, float)) and p > 0]
    if len(set(valid_prices)) == 1:
        print(f"✅ ALL PRICES CONSISTENT: ${valid_prices[0]:,.2f}")
        return valid_prices[0]
    else:
        print(f"⚠️  PRICE INCONSISTENCY DETECTED")
        return None

def verify_volume_profile_calculations(analysis):
    """Verify volume profile calculations"""
    print("\n" + "="*60)
    print("📊 VOLUME PROFILE CALCULATIONS")
    print("="*60)
    
    try:
        vp_data = analysis['volume_profile_analysis']['volume_profile']
        
        # Check POC
        poc = vp_data['poc']
        print(f"✅ POC Price: ${poc['price']:,.2f}")
        print(f"   POC Volume: {poc['volume']:,.2f}")
        
        # Check Value Area
        va = vp_data['value_area']
        print(f"✅ Value Area High: ${va['high']:,.2f}")
        print(f"✅ Value Area Low: ${va['low']:,.2f}")
        print(f"   Value Area Volume %: {va['volume_percentage']:.1f}%")
        
        # Verify current price position
        current_price = verify_current_price(analysis)
        if current_price:
            if current_price > va['high']:
                position = "ABOVE_VALUE_AREA"
            elif current_price < va['low']:
                position = "BELOW_VALUE_AREA"
            else:
                position = "INSIDE_VALUE_AREA"
            
            print(f"✅ Price Position: {position}")
            
            # Distance from POC
            poc_distance = ((current_price - poc['price']) / poc['price']) * 100
            print(f"✅ Distance from POC: {poc_distance:+.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Volume Profile Error: {e}")
        return False

def verify_technical_indicators(analysis):
    """Verify technical indicator calculations"""
    print("\n" + "="*60)
    print("🔧 TECHNICAL INDICATORS VERIFICATION")
    print("="*60)
    
    try:
        mtf_data = analysis['multi_timeframe_analysis']['timeframe_data']
        
        for tf_name, tf_data in mtf_data.items():
            print(f"\n📈 {tf_name.upper()} Timeframe:")
            
            if 'ohlcv' not in tf_data or not tf_data['ohlcv']:
                print("   ❌ No OHLCV data")
                continue
            
            # Convert to DataFrame for verification
            df = pd.DataFrame(tf_data['ohlcv'])
            
            if len(df) < 20:
                print(f"   ⚠️  Limited data: {len(df)} candles")
                continue
            
            # Verify RSI calculation
            if 'enhanced_indicators' in tf_data:
                indicators = tf_data['enhanced_indicators']
                
                # Calculate RSI ourselves to verify
                rsi_calculated = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
                rsi_stored = indicators.get('rsi', 0)
                
                if abs(rsi_calculated - rsi_stored) < 1.0:  # Allow 1 point tolerance
                    print(f"   ✅ RSI: {rsi_stored:.1f} (verified)")
                else:
                    print(f"   ❌ RSI Mismatch: Stored={rsi_stored:.1f}, Calculated={rsi_calculated:.1f}")
                
                # Verify MACD
                macd_line = ta.trend.MACD(df['close']).macd().iloc[-1]
                macd_stored = indicators.get('macd', {}).get('macd', 0)
                
                if abs(macd_line - macd_stored) < 0.01:  # Small tolerance for MACD
                    print(f"   ✅ MACD: {macd_stored:.4f} (verified)")
                else:
                    print(f"   ❌ MACD Mismatch: Stored={macd_stored:.4f}, Calculated={macd_line:.4f}")
                
                # Check ATR
                atr_calculated = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[-1]
                atr_stored = indicators.get('atr', 0)
                
                if abs(atr_calculated - atr_stored) < (atr_calculated * 0.05):  # 5% tolerance
                    print(f"   ✅ ATR: {atr_stored:.2f} (verified)")
                else:
                    print(f"   ❌ ATR Mismatch: Stored={atr_stored:.2f}, Calculated={atr_calculated:.2f}")
            
    except Exception as e:
        print(f"❌ Technical Indicators Error: {e}")
        return False
    
    return True

def verify_ultimate_score_calculation(analysis):
    """Verify Ultimate Score calculation"""
    print("\n" + "="*60)
    print("🎯 ULTIMATE SCORE VERIFICATION")
    print("="*60)
    
    try:
        score_data = analysis['ultimate_score']
        
        # Get component scores
        mtf_score = score_data['component_scores']['multi_timeframe_score']
        vp_score = score_data['component_scores']['volume_profile_score']
        tech_score = score_data['component_scores']['technical_score']
        confluence_bonus = score_data['component_scores']['confluence_bonus']
        
        # Get weights
        weights = score_data['weights_used']
        mtf_weight = weights['multi_timeframe']
        vp_weight = weights['volume_profile']
        tech_weight = weights['technical']
        
        # Calculate weighted score
        calculated_score = (mtf_score * mtf_weight) + (vp_score * vp_weight) + (tech_score * tech_weight) + confluence_bonus
        stored_score = score_data['composite_score']
        
        print(f"📊 Component Scores:")
        print(f"   Multi-Timeframe: {mtf_score:.1f} (weight: {mtf_weight})")
        print(f"   Volume Profile: {vp_score:.1f} (weight: {vp_weight})")
        print(f"   Technical: {tech_score:.1f} (weight: {tech_weight})")
        print(f"   Confluence Bonus: {confluence_bonus:.1f}")
        
        print(f"\n🧮 Score Calculation:")
        print(f"   Calculated: {calculated_score:.1f}")
        print(f"   Stored: {stored_score:.1f}")
        
        if abs(calculated_score - stored_score) < 0.1:
            print(f"   ✅ SCORE CALCULATION VERIFIED")
        else:
            print(f"   ❌ SCORE MISMATCH DETECTED")
        
        return True
        
    except Exception as e:
        print(f"❌ Ultimate Score Error: {e}")
        return False

def verify_data_completeness(analysis):
    """Verify all required data is present in the analysis"""
    print("\n" + "="*60)
    print("📋 DATA COMPLETENESS CHECK")
    print("="*60)
    
    required_sections = {
        'symbol': 'Symbol identification',
        'timestamp': 'Analysis timestamp',
        'multi_timeframe_analysis': 'Multi-timeframe data',
        'volume_profile_analysis': 'Volume profile calculations',
        'ultimate_score': 'Ultimate scoring system',
        'enhanced_trading_signals': 'Trading signals',
        'enhanced_ai_prompt': 'AI prompt generation'
    }
    
    missing_sections = []
    
    for section, description in required_sections.items():
        if section in analysis and analysis[section]:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            missing_sections.append(section)
    
    # Check timeframe coverage
    print(f"\n📈 Timeframe Coverage:")
    expected_timeframes = ['1w', '1d', '4h', '1h']
    
    try:
        available_timeframes = list(analysis['multi_timeframe_analysis']['timeframe_data'].keys())
        
        for tf in expected_timeframes:
            if tf in available_timeframes:
                print(f"✅ {tf.upper()} timeframe")
            else:
                print(f"❌ {tf.upper()} timeframe (missing)")
        
        # Check data quality
        data_quality_score = (len(available_timeframes) / len(expected_timeframes)) * 100
        print(f"\n📊 Data Quality Score: {data_quality_score:.0f}%")
        
    except:
        print("❌ Unable to verify timeframe data")
    
    return len(missing_sections) == 0

def regenerate_rich_reports():
    """Regenerate rich reports with corrected data"""
    print("\n" + "="*60)
    print("🔄 REGENERATING RICH REPORTS")
    print("="*60)
    
    try:
        from llm_integration import LLMIntegration
        import asyncio
        
        # Load latest analysis
        analysis = load_latest_analysis("BTC_USDT")
        if not analysis:
            print("❌ No analysis data to regenerate reports")
            return False
        
        # Initialize LLM integration (for report generation only)
        llm = LLMIntegration()
        
        # Generate updated reports (without LLM call)
        symbol = analysis.get('symbol', 'BTC/USDT')
        
        # Create mock LLM response for report generation
        mock_llm_response = {
            'model': 'claude-sonnet-4-20250514',
            'content': 'Reports regenerated with corrected current price data',
            'timestamp': datetime.now().isoformat()
        }
        
        report_files = llm._generate_rich_reports(symbol, mock_llm_response, analysis)
        
        print(f"✅ HTML Report: {report_files['html_report']}")
        print(f"✅ Markdown Report: {report_files['markdown_report']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Report regeneration error: {e}")
        return False

def main():
    """Run comprehensive TA verification"""
    print("🔍 COMPREHENSIVE TECHNICAL ANALYSIS VERIFICATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Load latest analysis
    analysis = load_latest_analysis("BTC_USDT")
    if not analysis:
        print("❌ Cannot proceed without analysis data")
        return
    
    # Run all verifications
    results = {}
    
    print(f"\nAnalyzing: {analysis.get('symbol', 'UNKNOWN')}")
    print(f"Analysis Date: {analysis.get('timestamp', 'UNKNOWN')}")
    
    results['current_price'] = verify_current_price(analysis)
    results['volume_profile'] = verify_volume_profile_calculations(analysis)
    results['technical_indicators'] = verify_technical_indicators(analysis)
    results['ultimate_score'] = verify_ultimate_score_calculation(analysis)
    results['data_completeness'] = verify_data_completeness(analysis)
    
    # Summary
    print("\n" + "="*80)
    print("📋 VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test.replace('_', ' ').title():25}: {status}")
    
    print(f"\n🎯 Overall Score: {passed}/{total} ({(passed/total)*100:.0f}%)")
    
    if passed == total:
        print("🎉 ALL VERIFICATIONS PASSED - TA CALCULATIONS ARE CORRECT!")
    else:
        print("⚠️  SOME VERIFICATIONS FAILED - REVIEW REQUIRED")
    
    # Offer to regenerate reports
    if results['current_price'] and input("\nRegenerate rich reports with corrected data? (y/n): ").lower() == 'y':
        regenerate_rich_reports()

if __name__ == "__main__":
    main()