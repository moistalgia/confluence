#!/usr/bin/env python3
"""
AI Integration Module - Bridge between technical analysis and LLM systems
Provides intelligent analysis enhancement and prompt optimization
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class AIIntegration:
    """AI-powered analysis integration and enhancement system"""
    
    def __init__(self):
        self.confidence_thresholds = {
            'high': 75,
            'medium': 50,
            'low': 25
        }
        
        self.analysis_weights = {
            'technical_indicators': 0.30,
            'volume_profile': 0.25,
            'multi_timeframe': 0.25,
            'sentiment': 0.10,
            'risk_factors': 0.10
        }
    
    def enhance_analysis_confidence(self, analysis_data: Dict) -> Dict:
        """Enhance analysis confidence using AI feedback methodology"""
        
        try:
            enhanced_analysis = analysis_data.copy()
            
            # Calculate base confidence from data quality
            base_confidence = self._calculate_base_confidence(analysis_data)
            
            # Apply AI feedback enhancements
            enhancements = self._apply_ai_feedback_enhancements(analysis_data)
            
            # Calculate final confidence score
            final_confidence = self._calculate_final_confidence(base_confidence, enhancements)
            
            enhanced_analysis['ai_enhancement'] = {
                'base_confidence': base_confidence,
                'enhancements_applied': enhancements,
                'final_confidence': final_confidence,
                'confidence_level': self._get_confidence_level(final_confidence),
                'enhancement_summary': self._generate_enhancement_summary(enhancements)
            }
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"AI enhancement error: {e}")
            return analysis_data
    
    def _calculate_base_confidence(self, analysis_data: Dict) -> float:
        """Calculate base confidence from raw analysis data"""
        
        confidence_factors = []
        
        # Data completeness factor
        if 'data_quality' in analysis_data:
            quality_score = analysis_data['data_quality'].get('overall_score', 50)
            confidence_factors.append(quality_score * 0.3)
        
        # Technical indicator convergence
        if 'enhanced_signals' in analysis_data:
            signal_strength = analysis_data['enhanced_signals'].get('signal_strength', 0)
            confidence_factors.append(signal_strength * 0.25)
        
        # Multi-timeframe confluence
        if 'confluence_analysis' in analysis_data:
            confluence_score = analysis_data['confluence_analysis'].get('overall_score', 0)
            confidence_factors.append(confluence_score * 0.25)
        
        # Volume confirmation
        if 'volume_profile' in analysis_data:
            volume_confidence = analysis_data['volume_profile'].get('confidence_score', 50)
            confidence_factors.append(volume_confidence * 0.2)
        
        return sum(confidence_factors) if confidence_factors else 50
    
    def _apply_ai_feedback_enhancements(self, analysis_data: Dict) -> Dict:
        """Apply specific AI feedback enhancements"""
        
        enhancements = {
            'multi_timeframe_boost': 0,
            'volume_profile_boost': 0,
            'pattern_recognition_boost': 0,
            'sentiment_integration_boost': 0,
            'risk_assessment_boost': 0
        }
        
        # Multi-timeframe enhancement (+25% expected improvement)
        if 'confluence_analysis' in analysis_data:
            confluence = analysis_data['confluence_analysis']
            if confluence.get('overall_score', 0) > 70:
                enhancements['multi_timeframe_boost'] = 25
            elif confluence.get('overall_score', 0) > 50:
                enhancements['multi_timeframe_boost'] = 15
            else:
                enhancements['multi_timeframe_boost'] = 5
        
        # Volume profile enhancement (+20% expected improvement)
        if 'volume_profile' in analysis_data:
            volume_data = analysis_data['volume_profile']
            if volume_data.get('signal_strength', 0) > 75:
                enhancements['volume_profile_boost'] = 20
            elif volume_data.get('signal_strength', 0) > 50:
                enhancements['volume_profile_boost'] = 12
            else:
                enhancements['volume_profile_boost'] = 5
        
        # Pattern recognition enhancement (+15% expected improvement)
        if 'patterns' in analysis_data:
            patterns = analysis_data['patterns']
            high_confidence_patterns = sum(1 for p in patterns if p.get('confidence') == 'HIGH')
            if high_confidence_patterns >= 2:
                enhancements['pattern_recognition_boost'] = 15
            elif high_confidence_patterns >= 1:
                enhancements['pattern_recognition_boost'] = 10
            else:
                enhancements['pattern_recognition_boost'] = 3
        
        # Sentiment integration boost
        if 'sentiment_analysis' in analysis_data:
            sentiment = analysis_data['sentiment_analysis']
            if sentiment.get('confidence', 'Low') == 'High':
                enhancements['sentiment_integration_boost'] = 8
            elif sentiment.get('confidence', 'Low') == 'Medium':
                enhancements['sentiment_integration_boost'] = 4
        
        # Risk assessment enhancement
        if 'risk_analysis' in analysis_data:
            risk = analysis_data['risk_analysis']
            if risk.get('overall_assessment', {}).get('risk_level') in ['Low', 'Medium']:
                enhancements['risk_assessment_boost'] = 5
        
        return enhancements
    
    def _calculate_final_confidence(self, base_confidence: float, enhancements: Dict) -> float:
        """Calculate final confidence score with enhancements"""
        
        # Apply enhancements as percentage improvements
        total_enhancement = sum(enhancements.values())
        
        # Cap total enhancement at 60% (as per AI feedback analysis)
        capped_enhancement = min(total_enhancement, 60)
        
        # Apply enhancement as multiplicative factor
        enhancement_factor = 1 + (capped_enhancement / 100)
        enhanced_confidence = base_confidence * enhancement_factor
        
        # Cap final confidence at 95% (never 100% certain)
        return min(enhanced_confidence, 95)
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """Convert confidence score to categorical level"""
        
        if confidence_score >= self.confidence_thresholds['high']:
            return 'HIGH'
        elif confidence_score >= self.confidence_thresholds['medium']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_enhancement_summary(self, enhancements: Dict) -> List[str]:
        """Generate human-readable enhancement summary"""
        
        summary = []
        
        for enhancement, boost in enhancements.items():
            if boost > 0:
                enhancement_name = enhancement.replace('_', ' ').title().replace('Boost', '')
                summary.append(f"{enhancement_name}: +{boost}% confidence")
        
        return summary
    
    def optimize_for_llm_analysis(self, analysis_data: Dict) -> Dict:
        """Optimize analysis data for LLM consumption"""
        
        try:
            optimized = {
                'analysis_timestamp': datetime.now().isoformat(),
                'symbol': analysis_data.get('symbol', 'UNKNOWN'),
                'confidence_assessment': analysis_data.get('ai_enhancement', {}),
                'key_insights': self._extract_key_insights(analysis_data),
                'risk_reward_profile': self._generate_risk_reward_profile(analysis_data),
                'trading_context': self._generate_trading_context(analysis_data),
                'data_sources': self._identify_data_sources(analysis_data)
            }
            
            return optimized
            
        except Exception as e:
            logger.error(f"LLM optimization error: {e}")
            return analysis_data
    
    def _extract_key_insights(self, analysis_data: Dict) -> List[str]:
        """Extract the most important insights for LLM analysis"""
        
        insights = []
        
        # Signal insights
        if 'enhanced_signals' in analysis_data:
            signals = analysis_data['enhanced_signals']
            primary_signal = signals.get('primary_signal', 'NEUTRAL')
            signal_strength = signals.get('signal_strength', 0)
            
            insights.append(f"Primary Signal: {primary_signal} with {signal_strength}% strength")
            
            if signals.get('entry_conditions'):
                insights.append(f"Entry Conditions Met: {len(signals['entry_conditions'])}")
            
            if signals.get('risk_factors'):
                insights.append(f"Risk Factors Identified: {len(signals['risk_factors'])}")
        
        # Confluence insights
        if 'confluence_analysis' in analysis_data:
            confluence = analysis_data['confluence_analysis']
            score = confluence.get('overall_score', 0)
            insights.append(f"Multi-timeframe Confluence: {score}/100")
            
            if 'trend_alignment' in confluence:
                trend_data = confluence['trend_alignment']
                dominant_trend = trend_data.get('dominant_trend', 'MIXED')
                alignment = trend_data.get('alignment_strength', 0)
                insights.append(f"Trend Alignment: {dominant_trend} ({alignment:.1%} agreement)")
        
        # Volume insights
        if 'volume_profile' in analysis_data:
            volume = analysis_data['volume_profile']
            insights.append(f"Volume Profile Signal: {volume.get('primary_signal', 'NEUTRAL')}")
            
            if 'value_area' in volume:
                va = volume['value_area']
                insights.append(f"Price vs Value Area: {va.get('price_position', 'UNKNOWN')}")
        
        return insights[:10]  # Limit to top 10 insights
    
    def _generate_risk_reward_profile(self, analysis_data: Dict) -> Dict:
        """Generate risk/reward profile for the analysis"""
        
        profile = {
            'overall_risk': 'MEDIUM',
            'potential_reward': 'MEDIUM',
            'risk_factors': [],
            'reward_catalysts': []
        }
        
        try:
            # Extract risk factors
            if 'enhanced_signals' in analysis_data:
                risk_factors = analysis_data['enhanced_signals'].get('risk_factors', [])
                profile['risk_factors'].extend(risk_factors)
            
            # Extract reward catalysts (entry conditions)
            if 'enhanced_signals' in analysis_data:
                entry_conditions = analysis_data['enhanced_signals'].get('entry_conditions', [])
                profile['reward_catalysts'].extend(entry_conditions)
            
            # Assess overall risk level
            total_risks = len(profile['risk_factors'])
            if total_risks == 0:
                profile['overall_risk'] = 'LOW'
            elif total_risks <= 2:
                profile['overall_risk'] = 'MEDIUM'
            else:
                profile['overall_risk'] = 'HIGH'
            
            # Assess potential reward
            total_catalysts = len(profile['reward_catalysts'])
            if total_catalysts >= 3:
                profile['potential_reward'] = 'HIGH'
            elif total_catalysts >= 1:
                profile['potential_reward'] = 'MEDIUM'
            else:
                profile['potential_reward'] = 'LOW'
            
        except Exception as e:
            logger.error(f"Risk/reward profile generation error: {e}")
        
        return profile
    
    def _generate_trading_context(self, analysis_data: Dict) -> Dict:
        """Generate trading context for better LLM understanding"""
        
        context = {
            'market_structure': 'UNKNOWN',
            'volatility_regime': 'MEDIUM',
            'volume_environment': 'NORMAL',
            'trend_context': 'NEUTRAL'
        }
        
        try:
            # Market structure from confluence
            if 'confluence_analysis' in analysis_data:
                confluence_score = analysis_data['confluence_analysis'].get('overall_score', 0)
                if confluence_score > 70:
                    context['market_structure'] = 'STRONG_CONFLUENCE'
                elif confluence_score > 40:
                    context['market_structure'] = 'MODERATE_CONFLUENCE'
                else:
                    context['market_structure'] = 'WEAK_CONFLUENCE'
            
            # Volatility regime from technical indicators
            if 'timeframe_data' in analysis_data:
                daily_data = analysis_data['timeframe_data'].get('1d', {}).get('indicators', {})
                volatility = daily_data.get('volatility', 'MEDIUM')
                context['volatility_regime'] = volatility
            
            # Volume environment
            if 'volume_profile' in analysis_data:
                volume_strength = analysis_data['volume_profile'].get('volume_strength', 'MEDIUM')
                context['volume_environment'] = volume_strength
            
            # Trend context
            if 'confluence_analysis' in analysis_data:
                trend_alignment = analysis_data['confluence_analysis'].get('trend_alignment', {})
                dominant_trend = trend_alignment.get('dominant_trend', 'MIXED')
                context['trend_context'] = dominant_trend
            
        except Exception as e:
            logger.error(f"Trading context generation error: {e}")
        
        return context
    
    def _identify_data_sources(self, analysis_data: Dict) -> List[str]:
        """Identify what data sources were used in the analysis"""
        
        sources = []
        
        if 'timeframe_data' in analysis_data:
            timeframes = list(analysis_data['timeframe_data'].keys())
            sources.append(f"Multi-timeframe OHLCV data ({', '.join(timeframes)})")
        
        if 'volume_profile' in analysis_data:
            sources.append("Volume-at-Price analysis")
        
        if 'patterns' in analysis_data:
            sources.append("Chart pattern recognition")
        
        if 'sentiment_analysis' in analysis_data:
            sources.append("Market sentiment data")
        
        if 'risk_analysis' in analysis_data:
            sources.append("Risk management analysis")
        
        sources.append("Technical indicators (RSI, MACD, Bollinger Bands, etc.)")
        
        return sources

def main():
    """Test AI Integration module"""
    
    print("🤖 AI INTEGRATION MODULE TEST")
    print("=" * 50)
    
    # Mock analysis data
    test_analysis = {
        'symbol': 'BTC/USDT',
        'data_quality': {'overall_score': 85},
        'enhanced_signals': {
            'primary_signal': 'BULLISH',
            'signal_strength': 75,
            'entry_conditions': ['High volume', 'Trend alignment'],
            'risk_factors': ['Resistance nearby']
        },
        'confluence_analysis': {'overall_score': 80},
        'volume_profile': {'confidence_score': 70, 'signal_strength': 65}
    }
    
    ai = AIIntegration()
    
    # Test confidence enhancement
    enhanced = ai.enhance_analysis_confidence(test_analysis)
    enhancement = enhanced.get('ai_enhancement', {})
    
    print(f"Base Confidence: {enhancement.get('base_confidence', 'N/A'):.1f}")
    print(f"Final Confidence: {enhancement.get('final_confidence', 'N/A'):.1f}")
    print(f"Confidence Level: {enhancement.get('confidence_level', 'N/A')}")
    
    # Test LLM optimization
    optimized = ai.optimize_for_llm_analysis(enhanced)
    insights = optimized.get('key_insights', [])
    
    print(f"\nKey Insights: {len(insights)}")
    for insight in insights[:3]:
        print(f"  - {insight}")

if __name__ == "__main__":
    main()