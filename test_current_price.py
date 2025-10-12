#!/usr/bin/env python3
"""
Quick test to verify current price extraction is working correctly
"""

import sys
import logging
from complete_prompt_generator import CompletePromptGenerator
from enhanced_multi_timeframe_analyzer import EnhancedMultiTimeframeAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_current_price():
    """Test that current price is being extracted correctly (not $0.00)"""
    
    print("🧪 Testing Current Price Extraction...")
    print("=" * 50)
    
    try:
        # Initialize components
        analyzer = EnhancedMultiTimeframeAnalyzer()
        prompt_gen = CompletePromptGenerator()
        
        # Analyze BTC/USDT
        symbol = "BTC/USDT"
        print(f"📊 Analyzing {symbol}...")
        
        analysis_data = analyzer.analyze_symbol(symbol)
        
        # Check if we have price context
        if 'price_context' in analysis_data:
            price_context = analysis_data['price_context']
            current_price = price_context.get('current_price', 0)
            
            print(f"✅ Current Price Found: ${current_price:,.2f}")
            
            if current_price > 0:
                print("🎉 SUCCESS: Current price is correctly extracted!")
                return True
            else:
                print("❌ FAILED: Current price is still $0.00")
                return False
        else:
            print("❌ FAILED: No price_context in analysis data")
            print("Available keys:", list(analysis_data.keys()))
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_current_price()
    sys.exit(0 if success else 1)