#!/usr/bin/env python3
"""Quick test to check if indicators are calculating correctly"""

from enhanced_multi_timeframe_analyzer import EnhancedMultiTimeframeAnalyzer
import logging

logging.basicConfig(level=logging.INFO)

analyzer = EnhancedMultiTimeframeAnalyzer()
result = analyzer.analyze_multi_timeframe('BTC/USD')

print('Analysis completed')
print('Timeframes analyzed:')
for tf, data in result['timeframe_data'].items():
    indicators = data.get('indicators', {})
    print(f'  {tf}: {len(indicators)} indicators calculated')
    if 'rsi' in indicators:
        print(f'    RSI: {indicators["rsi"]}')
    if 'error' in indicators:
        print(f'    ERROR: {indicators["error"]}')
    
print(f'Sentiment analysis: {result.get("sentiment_analysis", {}).get("overall_sentiment", "Not found")}')

# Clean shutdown to prevent daemon thread issues
try:
    analyzer.alert_manager.stop()
except:
    pass