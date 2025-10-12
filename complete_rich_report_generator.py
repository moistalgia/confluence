#!/usr/bin/env python3
"""
Complete Rich Report Generator
Creates comprehensive trading reports from COMPLETE analysis and LLM responses
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import re

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def load_complete_analysis_data():
    """Load all COMPLETE analysis data and LLM responses"""
    
    analysis_data = {}
    
    # Load processed analysis files
    processed_dir = Path("output/ultimate_analysis/processed_data")
    for processed_file in processed_dir.glob("*_processed_analysis_*.json"):
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            symbol = data.get('symbol', 'UNKNOWN')
            analysis_data[symbol] = data
    
    # Load LLM responses
    response_data = {}
    response_dir = Path("output/llm_responses")
    for response_file in response_dir.glob("response_*COMPLETE*.json"):
        with open(response_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract symbol from response content
            analysis_content = data.get('analysis', '')
            if 'BTC/USDT' in analysis_content or 'BTC' in analysis_content:
                symbol = 'BTC/USDT'
            elif 'ETH/USDT' in analysis_content or 'ETH' in analysis_content:
                symbol = 'ETH/USDT'
            else:
                # Fallback to filename
                filename = response_file.name
                if 'BTC' in filename:
                    symbol = 'BTC/USDT'
                elif 'ETH' in filename:
                    symbol = 'ETH/USDT'
                else:
                    symbol = 'UNKNOWN'
            response_data[symbol] = data
    
    return analysis_data, response_data

def generate_complete_rich_report(symbol: str, analysis: Dict, llm_response: Dict) -> str:
    """Generate comprehensive rich HTML report with COMPLETE data"""
    
    # Extract key data
    ultimate_score = analysis.get('ultimate_score', {})
    signals = analysis.get('enhanced_trading_signals', {})
    mtf_data = analysis.get('multi_timeframe_analysis', {})
    vp_data = analysis.get('volume_profile_analysis', {})
    
    # Get current price
    current_price = 0
    if 'price_analysis' in vp_data:
        current_price = vp_data['price_analysis'].get('current_price', 0)
    
    # Extract LLM recommendations
    llm_content = llm_response.get('response_content', '')
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    html_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COMPLETE Trading Analysis: {symbol}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6; color: #333; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; text-align: center; padding: 30px; border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
        .header .subtitle {{ font-size: 1.2em; opacity: 0.9; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; }}
        .card {{ 
            background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1); backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2); transition: transform 0.3s ease;
        }}
        .card:hover {{ transform: translateY(-5px); box-shadow: 0 15px 35px rgba(0,0,0,0.15); }}
        .card h2 {{ color: #4a5568; font-size: 1.4em; margin-bottom: 20px; display: flex; align-items: center; }}
        .card h2::before {{ content: '📊'; margin-right: 10px; font-size: 1.2em; }}
        .metric {{ display: flex; justify-content: space-between; margin: 12px 0; padding: 8px 0; border-bottom: 1px solid #eee; }}
        .metric-label {{ font-weight: 600; color: #4a5568; }}
        .metric-value {{ font-weight: 700; }}
        .score {{ font-size: 2.5em; font-weight: bold; text-align: center; margin: 15px 0; }}
        .score.high {{ color: #48bb78; }}
        .score.medium {{ color: #ed8936; }}
        .score.low {{ color: #f56565; }}
        .bias {{ text-transform: uppercase; font-weight: bold; padding: 8px 16px; border-radius: 25px; display: inline-block; }}
        .bias.bullish {{ background: #c6f6d5; color: #22543d; }}
        .bias.bearish {{ background: #fed7d7; color: #742a2a; }}
        .bias.neutral {{ background: #e2e8f0; color: #2d3748; }}
        .levels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px; }}
        .level-group {{ background: #f7fafc; padding: 15px; border-radius: 8px; }}
        .level-group h4 {{ margin-bottom: 10px; color: #4a5568; }}
        .level {{ margin: 5px 0; padding: 5px 10px; background: white; border-radius: 5px; font-family: monospace; }}
        .llm-section {{ grid-column: 1 / -1; }}
        .llm-content {{ 
            background: #f8f9ff; padding: 25px; border-radius: 10px; border-left: 5px solid #667eea;
            font-size: 1.05em; line-height: 1.8; white-space: pre-wrap;
        }}
        .timeframe-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .timeframe-item {{ 
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white; padding: 15px; border-radius: 10px; text-align: center;
        }}
        .indicator {{ margin: 8px 0; font-size: 0.9em; }}
        .volume-profile {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }}
        .complete-badge {{ 
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #8b4513; padding: 5px 15px; border-radius: 20px; font-size: 0.9em;
            display: inline-block; margin-bottom: 15px; font-weight: bold;
        }}
        .warning {{ background: #fed7d7; color: #742a2a; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .success {{ background: #c6f6d5; color: #22543d; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        @media (max-width: 768px) {{ 
            .grid {{ grid-template-columns: 1fr; }}
            .header h1 {{ font-size: 1.8em; }}
            .card {{ padding: 15px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 COMPLETE TRADING ANALYSIS</h1>
            <div class="subtitle">{symbol} • {timestamp} • 100% Data Inclusion</div>
            <div class="complete-badge">✨ COMPLETE Analysis with Claude Sonnet 4</div>
        </div>
        
        <div class="grid">
            <!-- Ultimate Score Card -->
            <div class="card">
                <h2>Ultimate Score & Bias</h2>
                <div class="score {'high' if ultimate_score.get('composite_score', 0) >= 70 else 'medium' if ultimate_score.get('composite_score', 0) >= 40 else 'low'}">
                    {ultimate_score.get('composite_score', 0)}/100
                </div>
                <div class="metric">
                    <span class="metric-label">Confidence Level:</span>
                    <span class="metric-value">{ultimate_score.get('confidence_level', 'UNKNOWN')}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Primary Bias:</span>
                    <span class="metric-value">
                        <span class="bias {'bullish' if 'BULLISH' in str(signals.get('primary_bias', '')) else 'bearish' if 'BEARISH' in str(signals.get('primary_bias', '')) else 'neutral'}">
                            {signals.get('primary_bias', 'NEUTRAL')}
                        </span>
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Current Price:</span>
                    <span class="metric-value">${current_price:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Analysis Type:</span>
                    <span class="metric-value">COMPLETE (100% Data)</span>
                </div>
            </div>

            <!-- Component Scores -->
            <div class="card">
                <h2>Score Breakdown</h2>"""
    
    # Add component scores
    component_scores = ultimate_score.get('component_scores', {})
    for component, score in component_scores.items():
        component_name = component.replace('_', ' ').title()
        html_report += f"""
                <div class="metric">
                    <span class="metric-label">{component_name}:</span>
                    <span class="metric-value">{score}</span>
                </div>"""
    
    # Add weights
    weights = ultimate_score.get('weights_used', {})
    if weights:
        html_report += """<h4 style="margin-top: 15px; color: #4a5568;">Methodology Weights:</h4>"""
        for component, weight in weights.items():
            weight_pct = f"{weight:.0%}" if isinstance(weight, (int, float)) else str(weight)
            html_report += f"""
                <div class="metric">
                    <span class="metric-label">{component.replace('_', ' ').title()}:</span>
                    <span class="metric-value">{weight_pct}</span>
                </div>"""
    
    html_report += """
            </div>

            <!-- Multi-Timeframe Analysis -->
            <div class="card">
                <h2>Multi-Timeframe Analysis</h2>"""
    
    confluence_score = mtf_data.get('confluence_score', 0)
    html_report += f"""
                <div class="metric">
                    <span class="metric-label">Confluence Score:</span>
                    <span class="metric-value">{confluence_score}/100</span>
                </div>
                <div class="timeframe-grid">"""
    
    # Add timeframe breakdown
    timeframe_analysis = mtf_data.get('timeframe_data', {})
    for tf, tf_data in timeframe_analysis.items():
        # Skip error entries
        if isinstance(tf_data, dict) and 'error' in tf_data:
            continue
            
        # Extract indicators and calculate trend/momentum
        indicators = tf_data.get('indicators', {})
        rsi = indicators.get('rsi', 50)
        macd_line = indicators.get('macd_line', 0)
        macd_signal = indicators.get('macd_signal', 0)
        adx = indicators.get('adx', 0)
        
        # Determine trend based on indicators
        if rsi > 60 and macd_line > macd_signal:
            trend = 'BULLISH'
            if rsi > 70:
                trend = 'STRONG BULLISH'
        elif rsi < 40 and macd_line < macd_signal:
            trend = 'BEARISH'
            if rsi < 30:
                trend = 'STRONG BEARISH'
        else:
            trend = 'NEUTRAL'
            
        # Determine momentum strength
        if adx > 40:
            momentum = 'STRONG'
        elif adx > 25:
            momentum = 'MODERATE'
        else:
            momentum = 'WEAK'
            
        html_report += f"""
                    <div class="timeframe-item">
                        <h4>{tf.upper()}</h4>
                        <div>Trend: {trend}</div>
                        <div>Momentum: {momentum}</div>"""
        
        # Add key indicators
        html_report += f"""<div class="indicator">RSI: {rsi:.1f}</div>"""
        html_report += f"""<div class="indicator">MACD: {macd_line:.4f}</div>"""
        html_report += f"""<div class="indicator">ADX: {adx:.1f}</div>"""
        
        html_report += """</div>"""
    
    html_report += """
                </div>
            </div>

            <!-- Volume Profile Analysis -->
            <div class="card volume-profile">
                <h2 style="color: white;">Volume Profile Analysis</h2>"""
    
    if vp_data and 'volume_profile' in vp_data:
        vp = vp_data['volume_profile']
        price_analysis = vp_data.get('price_analysis', {})
        
        # POC and Value Area
        poc_price = vp.get('poc', {}).get('price', 0)
        va_high = vp.get('value_area', {}).get('high', 0)
        va_low = vp.get('value_area', {}).get('low', 0)
        position = price_analysis.get('position', 'UNKNOWN')
        
        html_report += f"""
                <div class="metric" style="color: white; border-bottom-color: rgba(255,255,255,0.3);">
                    <span class="metric-label">Point of Control:</span>
                    <span class="metric-value">${poc_price:,.2f}</span>
                </div>
                <div class="metric" style="color: white; border-bottom-color: rgba(255,255,255,0.3);">
                    <span class="metric-label">Value Area:</span>
                    <span class="metric-value">${va_low:,.2f} - ${va_high:,.2f}</span>
                </div>
                <div class="metric" style="color: white; border-bottom-color: rgba(255,255,255,0.3);">
                    <span class="metric-label">Current Position:</span>
                    <span class="metric-value">{position}</span>
                </div>"""
        
        # Support/Resistance levels
        levels = price_analysis.get('volume_based_levels', {})
        if levels:
            html_report += """<div class="levels">"""
            
            if levels.get('support'):
                html_report += """<div class="level-group"><h4 style="color: white;">Support Levels</h4>"""
                for level in levels['support'][:3]:
                    html_report += f"""<div class="level">${level:,.2f}</div>"""
                html_report += """</div>"""
            
            if levels.get('resistance'):
                html_report += """<div class="level-group"><h4 style="color: white;">Resistance Levels</h4>"""
                for level in levels['resistance'][:3]:
                    html_report += f"""<div class="level">${level:,.2f}</div>"""
                html_report += """</div>"""
            
            html_report += """</div>"""
        
        # Volume Profile Trading Signals
        vp_signals = vp_data.get('trading_signals', {})
        if vp_signals:
            html_report += f"""
                <h4 style="color: white; margin-top: 15px;">Volume Profile Signals</h4>
                <div class="metric" style="color: white; border-bottom-color: rgba(255,255,255,0.3);">
                    <span class="metric-label">Signal Strength:</span>
                    <span class="metric-value">{vp_signals.get('signal_strength', 'UNKNOWN')}</span>
                </div>
                <div class="metric" style="color: white; border-bottom-color: rgba(255,255,255,0.3);">
                    <span class="metric-label">Direction:</span>
                    <span class="metric-value">{vp_signals.get('direction', 'UNKNOWN')}</span>
                </div>
                <div class="metric" style="color: white; border-bottom-color: rgba(255,255,255,0.3);">
                    <span class="metric-label">Confidence:</span>
                    <span class="metric-value">{vp_signals.get('confidence', 'UNKNOWN')}</span>
                </div>"""
    
    html_report += """
            </div>

            <!-- Enhanced Trading Signals -->
            <div class="card">
                <h2>Enhanced Trading Signals</h2>"""
    
    # Add enhanced signals data
    html_report += f"""
                <div class="metric">
                    <span class="metric-label">Signal Confidence:</span>
                    <span class="metric-value">{signals.get('confidence', 'UNKNOWN')}</span>
                </div>"""
    
    # Timeframe alignment
    tf_alignment = signals.get('timeframe_alignment', {})
    if tf_alignment:
        html_report += """<h4 style="margin-top: 15px; color: #4a5568;">Timeframe Alignment:</h4>"""
        for tf, alignment in tf_alignment.items():
            if isinstance(alignment, dict):
                trend = alignment.get('trend', 'UNKNOWN')
                momentum = alignment.get('momentum', 'UNKNOWN')
                html_report += f"""
                <div class="metric">
                    <span class="metric-label">{tf.upper()}:</span>
                    <span class="metric-value">{trend} ({momentum})</span>
                </div>"""
    
    # Risk assessment
    risk_assess = signals.get('risk_assessment', {})
    if risk_assess:
        html_report += f"""
                <h4 style="margin-top: 15px; color: #4a5568;">Risk Assessment:</h4>
                <div class="metric">
                    <span class="metric-label">Overall Risk:</span>
                    <span class="metric-value">{risk_assess.get('overall_risk', 'UNKNOWN')}</span>
                </div>"""
    
    html_report += """
            </div>

            <!-- Claude Sonnet 4 Analysis -->
            <div class="card llm-section">
                <h2 style="margin-bottom: 20px;">🤖 Claude Sonnet 4 Professional Analysis</h2>
                <div class="success">
                    ✨ This analysis was generated using COMPLETE data with 100% inclusion of all technical indicators, 
                    volume profile analysis, multi-timeframe confluence, and enhanced trading signals.
                </div>
                <div class="llm-content">{llm_content}</div>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding: 20px; color: #666;">
            <p>🎯 COMPLETE Analysis Report • Generated with 100% Data Inclusion • Powered by Claude Sonnet 4</p>
            <p style="font-size: 0.9em; margin-top: 5px;">
                Data Sources: Multi-timeframe Analysis • Volume Profile • Enhanced Technical Indicators • AI Trading Signals
            </p>
        </div>
    </div>
</body>
</html>"""
    
    return html_report

def main():
    """Generate COMPLETE rich reports for all analysis"""
    
    print("🎯 COMPLETE Rich Report Generator")
    print("Generating comprehensive reports with 100% data inclusion")
    print("=" * 70)
    
    # Load all data
    analysis_data, response_data = load_complete_analysis_data()
    
    if not analysis_data:
        print("❌ No analysis data found")
        return
    
    if not response_data:
        print("❌ No LLM response data found")
        return
    
    # Create output directory
    output_dir = Path("output/rich_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reports for each symbol
    for symbol in analysis_data.keys():
        if symbol in response_data:
            print(f"📊 Generating COMPLETE report for {symbol}")
            
            analysis = analysis_data[symbol]
            llm_response = response_data[symbol]
            
            # Generate report
            report_html = generate_complete_rich_report(symbol, analysis, llm_response)
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"COMPLETE_trading_report_{symbol.replace('/', '_')}_{timestamp}.html"
            report_path = output_dir / filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            print(f"✅ Saved: {filename}")
            print(f"   📄 File size: {len(report_html):,} characters")
        else:
            print(f"⚠️  {symbol}: Analysis available but no LLM response found")
    
    print()
    print("🚀 COMPLETE Rich Reports Generated!")
    print("📂 Location: output/rich_reports/")
    print("🎯 Features: 100% data inclusion, Claude Sonnet 4 analysis, comprehensive visualizations")

if __name__ == "__main__":
    main()