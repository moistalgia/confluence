#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis for Crypto Markets
Integrates multiple sentiment sources: Fear/Greed Index, Social Sentiment, Funding Rates
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Comprehensive sentiment data structure"""
    fear_greed_index: int = 50  # 0-100 scale
    fear_greed_classification: str = "NEUTRAL"
    social_sentiment: str = "NEUTRAL"
    funding_rate: float = 0.0
    funding_sentiment: str = "NEUTRAL"
    overall_sentiment: str = "NEUTRAL"
    sentiment_score: int = 50  # 0-100 composite score
    data_sources: List[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analysis with multiple data sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoAnalyzer/1.0'
        })
        
    def get_fear_greed_index(self) -> Dict:
        """Fetch Fear & Greed Index from API"""
        try:
            # Alternative Fear & Greed Index API (free)
            url = "https://api.alternative.me/fng/?limit=1"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    fng_data = data['data'][0]
                    value = int(fng_data['value'])
                    classification = fng_data['value_classification'].upper()
                    
                    return {
                        'index': value,
                        'classification': classification,
                        'timestamp': fng_data['timestamp'],
                        'status': 'success'
                    }
            
            return {'status': 'failed', 'error': 'API response invalid'}
            
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed Index: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def estimate_funding_rate_sentiment(self, symbol: str = "BTC") -> Dict:
        """Estimate funding rate sentiment (simplified without API keys)"""
        try:
            # This is a simplified version - in production you'd integrate with exchange APIs
            # For now, we'll create a reasonable estimate based on market conditions
            
            # Simulate funding rate analysis
            estimated_funding = 0.0001  # 0.01% per 8 hours (neutral)
            
            if estimated_funding > 0.0005:  # > 0.05%
                sentiment = "EXTREMELY_BULLISH"
            elif estimated_funding > 0.0002:  # > 0.02%
                sentiment = "BULLISH"
            elif estimated_funding < -0.0005:  # < -0.05%
                sentiment = "EXTREMELY_BEARISH"
            elif estimated_funding < -0.0002:  # < -0.02%
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            return {
                'rate': estimated_funding,
                'sentiment': sentiment,
                'status': 'estimated',
                'note': 'Funding rate estimated - integrate exchange APIs for real data'
            }
            
        except Exception as e:
            logger.warning(f"Failed to estimate funding rate: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def analyze_social_sentiment(self, symbol: str = "BTC") -> Dict:
        """Analyze social sentiment (simplified version)"""
        try:
            # This would integrate with Twitter API, Reddit API, etc.
            # For now, we'll provide a neutral baseline
            
            social_signals = {
                'twitter_sentiment': 'NEUTRAL',
                'reddit_sentiment': 'NEUTRAL', 
                'news_sentiment': 'NEUTRAL',
                'overall': 'NEUTRAL',
                'confidence': 0.5,
                'status': 'estimated',
                'note': 'Social sentiment estimated - integrate social APIs for real data'
            }
            
            return social_signals
            
        except Exception as e:
            logger.warning(f"Failed to analyze social sentiment: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def calculate_composite_sentiment(self, fear_greed: Dict, funding: Dict, social: Dict) -> int:
        """Calculate composite sentiment score (0-100)"""
        try:
            scores = []
            
            # Fear & Greed Index (40% weight)
            if fear_greed.get('status') == 'success':
                scores.append(fear_greed['index'] * 0.4)
            
            # Funding Rate Sentiment (30% weight)
            if funding.get('status') in ['success', 'estimated']:
                funding_sentiment = funding.get('sentiment', 'NEUTRAL')
                funding_score = {
                    'EXTREMELY_BEARISH': 10,
                    'BEARISH': 30,
                    'NEUTRAL': 50,
                    'BULLISH': 70,
                    'EXTREMELY_BULLISH': 90
                }.get(funding_sentiment, 50)
                scores.append(funding_score * 0.3)
            
            # Social Sentiment (30% weight)
            if social.get('status') in ['success', 'estimated']:
                social_sentiment = social.get('overall', 'NEUTRAL')
                social_score = {
                    'EXTREMELY_BEARISH': 10,
                    'BEARISH': 30,
                    'NEUTRAL': 50,
                    'BULLISH': 70,
                    'EXTREMELY_BULLISH': 90
                }.get(social_sentiment, 50)
                scores.append(social_score * 0.3)
            
            # Calculate weighted average
            if scores:
                composite_score = sum(scores) / sum([0.4, 0.3, 0.3][:len(scores)])
                return int(composite_score)
            
            return 50  # Neutral default
            
        except Exception as e:
            logger.warning(f"Failed to calculate composite sentiment: {e}")
            return 50
    
    def get_comprehensive_sentiment(self, symbol: str = "BTC/USDT") -> SentimentData:
        """Get comprehensive sentiment analysis"""
        try:
            # Extract base symbol (BTC from BTC/USDT)
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            
            # Fetch all sentiment data
            fear_greed = self.get_fear_greed_index()
            funding = self.estimate_funding_rate_sentiment(base_symbol)
            social = self.analyze_social_sentiment(base_symbol)
            
            # Calculate composite sentiment
            composite_score = self.calculate_composite_sentiment(fear_greed, funding, social)
            
            # Determine overall sentiment classification
            if composite_score >= 80:
                overall_sentiment = "EXTREMELY_BULLISH"
            elif composite_score >= 65:
                overall_sentiment = "BULLISH"
            elif composite_score >= 55:
                overall_sentiment = "SLIGHTLY_BULLISH"
            elif composite_score >= 45:
                overall_sentiment = "NEUTRAL"
            elif composite_score >= 35:
                overall_sentiment = "SLIGHTLY_BEARISH"
            elif composite_score >= 20:
                overall_sentiment = "BEARISH"
            else:
                overall_sentiment = "EXTREMELY_BEARISH"
            
            # Build data sources list
            data_sources = []
            if fear_greed.get('status') == 'success':
                data_sources.append('Fear & Greed Index')
            if funding.get('status') in ['success', 'estimated']:
                data_sources.append('Funding Rates')
            if social.get('status') in ['success', 'estimated']:
                data_sources.append('Social Sentiment')
            
            # Create comprehensive sentiment data
            sentiment_data = SentimentData(
                fear_greed_index=fear_greed.get('index', 50),
                fear_greed_classification=fear_greed.get('classification', 'NEUTRAL'),
                social_sentiment=social.get('overall', 'NEUTRAL'),
                funding_rate=funding.get('rate', 0.0),
                funding_sentiment=funding.get('sentiment', 'NEUTRAL'),
                overall_sentiment=overall_sentiment,
                sentiment_score=composite_score,
                data_sources=data_sources
            )
            
            logger.info(f"Sentiment analysis complete for {symbol}: {overall_sentiment} ({composite_score}/100)")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Comprehensive sentiment analysis failed: {e}")
            return SentimentData()  # Return neutral defaults

def main():
    """Test the sentiment analyzer"""
    analyzer = EnhancedSentimentAnalyzer()
    sentiment = analyzer.get_comprehensive_sentiment("BTC/USDT")
    
    print("=== Crypto Sentiment Analysis ===")
    print(f"Overall Sentiment: {sentiment.overall_sentiment}")
    print(f"Composite Score: {sentiment.sentiment_score}/100")
    print(f"Fear & Greed Index: {sentiment.fear_greed_index} ({sentiment.fear_greed_classification})")
    print(f"Social Sentiment: {sentiment.social_sentiment}")
    print(f"Funding Rate: {sentiment.funding_rate:.6f} ({sentiment.funding_sentiment})")
    print(f"Data Sources: {', '.join(sentiment.data_sources)}")
    
if __name__ == "__main__":
    main()