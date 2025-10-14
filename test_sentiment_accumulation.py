"""
Quick test to verify sentiment and accumulation data flow
"""
import asyncio
from ultimate_crypto_analyzer import UltimateCryptoAnalyzer

async def test_single_pair():
    """Test HBAR/USD analysis with sentiment and accumulation"""
    analyzer = UltimateCryptoAnalyzer()
    
    symbol = "HBAR/USD"
    print(f"🔍 Testing {symbol} with enhanced analysis...")
    
    # Run ultimate analysis 
    result = await analyzer.run_ultimate_analysis(symbol)
    
    # Check what was returned
    print(f"✅ Analysis complete. Result keys: {list(result.keys())}")
    
    # Check for sentiment data
    if 'sentiment_analysis' in result:
        sentiment = result['sentiment_analysis']
        print(f"📊 Sentiment Analysis Found:")
        print(f"   - Fear & Greed: {sentiment.get('fear_greed_index', 'N/A')}")
        print(f"   - Overall: {sentiment.get('overall_sentiment', 'N/A')}")
        print(f"   - Score: {sentiment.get('sentiment_score', 'N/A')}")
    else:
        print("❌ No sentiment analysis found in result")
    
    # Check for accumulation data
    if 'accumulation_analysis' in result:
        accumulation = result['accumulation_analysis']
        print(f"💎 Accumulation Analysis Found:")
        print(f"   - 1M: {accumulation.get('one_month_score', 'N/A')}")
        print(f"   - 6M: {accumulation.get('six_month_score', 'N/A')}")
        print(f"   - 1Y+: {accumulation.get('one_year_plus_score', 'N/A')}")
    else:
        print("❌ No accumulation analysis found in result")
        
    # Check if prompt was generated
    if 'enhanced_ai_prompt' in result:
        prompt = result['enhanced_ai_prompt']
        print(f"📝 Prompt generated: {len(prompt)} characters")
        
        # Check if sentiment section exists in prompt
        if 'SENTIMENT ANALYSIS' in prompt:
            print("✅ Sentiment section found in prompt")
        else:
            print("❌ Sentiment section missing from prompt")
            
        # Check if accumulation section exists
        if 'ACCUMULATION ANALYSIS' in prompt or 'Multi-Horizon' in prompt:
            print("✅ Accumulation section found in prompt")
        else:
            print("❌ Accumulation section missing from prompt")
    else:
        print("❌ No prompt found in result")

if __name__ == "__main__":
    asyncio.run(test_single_pair())