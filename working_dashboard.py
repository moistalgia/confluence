"""
Working Professional Trading Dashboard - Simplified Web Interface
===============================================================

Simple, reliable HTML generation that actually renders properly.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import webbrowser
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class WorkingDashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the working dashboard"""
    
    def __init__(self, *args, dashboard_manager=None, **kwargs):
        self.dashboard_manager = dashboard_manager
        super().__init__(*args, **kwargs)
        
    def do_GET(self):
        if self.path == '/dashboard' or self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            try:
                # Get dashboard data
                if self.dashboard_manager:
                    dashboard_data = self.dashboard_manager.get_dashboard_summary()
                else:
                    dashboard_data = self._get_mock_data()
                
                html = self._generate_simple_html(dashboard_data)
                self.wfile.write(html.encode('utf-8'))
                
            except Exception as e:
                logger.error(f"Error generating dashboard: {e}")
                error_html = self._generate_error_html(str(e))
                self.wfile.write(error_html.encode('utf-8'))
        else:
            self.send_error(404)
    
    def _generate_simple_html(self, data: Dict[str, Any]) -> str:
        """Generate simple HTML that actually works"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get basic stats
        perf = data.get('performance_stats', {})
        total_signals = perf.get('total_signals', 0)
        executed_trades = perf.get('executed_trades', 0)
        win_rate = perf.get('win_rate', 0.0)
        
        # Get validation stats  
        val_stats = data.get('validation_stats', {})
        avg_score = val_stats.get('average_score', 0.0)
        
        # Get price data
        price_monitor = data.get('price_monitor', {})
        monitored_pairs = price_monitor.get('monitored_pairs', 0)
        prices = price_monitor.get('prices', {})
        
        # Get recent signals (limit to 5)
        recent_signals = data.get('recent_signals', [])[:5]
        
        # Get active trades
        active_trades = data.get('active_trades', {})
        
        # Build HTML step by step
        html_parts = []
        
        # HTML header
        html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Professional Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .timestamp {
            font-size: 1.1em;
            opacity: 0.8;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .card h3 {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #87CEEB;
        }
        .stat {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stat:last-child {
            border-bottom: none;
        }
        .value {
            font-weight: bold;
        }
        .positive { color: #4CAF50; }
        .negative { color: #F44336; }
        .neutral { color: #FFC107; }
        .refresh-info {
            text-align: center;
            margin-top: 20px;
            opacity: 0.7;
        }
        .signal-item, .trade-item, .price-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
            font-size: 0.9em;
        }
        .expandable {
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .expandable:hover {
            background: rgba(0, 0, 0, 0.3);
        }
        .signal-header, .trade-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .expand-icon {
            transition: transform 0.3s;
            font-size: 0.8em;
        }
        .signal-details, .trade-details {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }
        .factor-breakdown {
            margin-top: 10px;
            font-size: 0.9em;
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 5px;
        }
        .factor {
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .factor:last-child {
            border-bottom: none;
        }
        .factor-details {
            margin-top: 5px;
            font-size: 0.85em;
            color: #B0B0B0;
            padding-left: 15px;
            line-height: 1.4;
        }
        .rejection-reason {
            margin-top: 10px;
            padding: 8px;
            background: rgba(244, 67, 54, 0.1);
            border-left: 3px solid #f44336;
            border-radius: 3px;
            font-size: 0.9em;
        }
        .completed-trade-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
            font-size: 0.9em;
            border-left: 3px solid transparent;
        }
        .completed-trade-item.profit {
            border-left-color: #4CAF50;
        }
        .completed-trade-item.loss {
            border-left-color: #f44336;
        }
        .trade-summary {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        .trade-pnl {
            font-weight: bold;
            font-size: 1.1em;
        }
        .detail-row {
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
        }
        .factor-breakdown {
            margin-top: 10px;
            padding: 8px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 3px;
        }
        .factor {
            margin: 3px 0;
            font-size: 0.85em;
        }
        .price-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .pair-name {
            font-weight: bold;
            font-size: 1.1em;
        }
        .pair-price {
            font-size: 1.2em;
            font-weight: bold;
            margin: 5px 0;
        }
        .pair-change {
            font-size: 0.9em;
        }
        .pair-status {
            font-size: 0.8em;
            text-transform: uppercase;
            margin-top: 5px;
        }
        .live {
            color: #4CAF50;
        }
        .stale {
            color: #FFC107;
        }
    </style>
</head>
<body>''')
        
        # Container and header
        html_parts.append('<div class="container">')
        html_parts.append('<div class="header">')
        html_parts.append('<h1>🚀 Professional Trading Dashboard</h1>')
        html_parts.append(f'<div class="timestamp">Last Updated: {timestamp}</div>')
        html_parts.append('</div>')
        
        # Performance stats card
        html_parts.append('<div class="grid">')
        html_parts.append('<div class="card">')
        html_parts.append('<h3>📊 Performance Statistics</h3>')
        html_parts.append(f'<div class="stat"><span>Total Signals:</span><span class="value">{total_signals}</span></div>')
        html_parts.append(f'<div class="stat"><span>Executed Trades:</span><span class="value">{executed_trades}</span></div>')
        html_parts.append(f'<div class="stat"><span>Win Rate:</span><span class="value positive">{win_rate:.1f}%</span></div>')
        html_parts.append(f'<div class="stat"><span>Avg Validation Score:</span><span class="value">{avg_score:.1f}%</span></div>')
        html_parts.append('</div>')
        
        # Price monitor card - Show ALL pairs
        html_parts.append('<div class="card">')
        html_parts.append('<h3>💰 Live Price Monitor</h3>')
        html_parts.append(f'<div class="stat"><span>Tracked Pairs:</span><span class="value">{monitored_pairs}</span></div>')
        
        if prices:
            html_parts.append('<div class="price-grid">')
            for pair, price_data in prices.items():
                price = price_data.get('current_price', price_data.get('price', 0.0))
                change = price_data.get('change_24h', price_data.get('24h_change', 0.0))
                change_class = 'positive' if change >= 0 else 'negative'
                status = price_data.get('status', 'unknown')
                
                # Format price based on asset type
                if 'BTC' in pair and price > 1000:
                    price_str = f'${price:,.2f}'
                elif 'ETH' in pair and price > 100:
                    price_str = f'${price:,.2f}'  
                elif price < 1:
                    price_str = f'${price:.6f}'
                else:
                    price_str = f'${price:.4f}'
                
                html_parts.append(f'<div class="price-item">')
                html_parts.append(f'<div class="pair-name">{pair}</div>')
                html_parts.append(f'<div class="pair-price">{price_str}</div>')
                html_parts.append(f'<div class="pair-change {change_class}">({change:+.2f}%)</div>')
                html_parts.append(f'<div class="pair-status {status}">{status.upper()}</div>')
                html_parts.append(f'</div>')
            html_parts.append('</div>')
        else:
            html_parts.append('<div class="stat"><span>No price data available</span></div>')
            
        html_parts.append('</div>')
        
        # Active trades card with expandable details
        html_parts.append('<div class="card">')
        html_parts.append('<h3>⚡ Active Trades</h3>')
        if active_trades:
            for idx, (trade_id, trade_info) in enumerate(list(active_trades.items())[:5]):  # Show max 5
                symbol = trade_info.get('symbol', 'Unknown')
                side = trade_info.get('action', trade_info.get('side', 'Unknown'))
                size = trade_info.get('position_size', trade_info.get('size', 0.0))
                entry = trade_info.get('entry_price', 0.0)
                current_pnl = trade_info.get('current_pnl', 0.0)
                pnl_class = 'positive' if current_pnl >= 0 else 'negative'
                
                # Calculate position value in USD
                position_usd = size * entry
                
                # Get current price for live position value
                current_price = entry  # Default to entry price
                if prices and symbol in prices:
                    current_price = prices[symbol].get('current_price', prices[symbol].get('price', entry))
                current_value_usd = size * current_price
                
                html_parts.append(f'<div class="trade-item expandable" onclick="toggleTrade({idx})">')
                html_parts.append(f'<div class="trade-header">')
                html_parts.append(f'<strong>{symbol}</strong> {side} | Size: {size:.4f} (${position_usd:.2f}) | ')
                html_parts.append(f'Current: ${current_value_usd:.2f} | P&L: <span class="{pnl_class}">${current_pnl:.2f}</span>')
                html_parts.append(f'<span class="expand-icon">▼</span>')
                html_parts.append(f'</div>')
                
                # Expandable trade details
                html_parts.append(f'<div id="trade-{idx}" class="trade-details" style="display: none;">')
                html_parts.append(f'<div class="detail-row"><strong>Trade ID:</strong> {trade_id}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Entry Price:</strong> ${entry:.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Current Price:</strong> ${current_price:.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Position Size:</strong> {size:.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Position Value (Entry):</strong> ${position_usd:.2f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Position Value (Current):</strong> ${current_value_usd:.2f}</div>')
                
                # Calculate strategy details
                stop_loss = trade_info.get("stop_loss", 0)
                take_profit = trade_info.get("take_profit", 0)
                
                if entry > 0 and stop_loss > 0 and take_profit > 0:
                    risk_pct = abs((entry - stop_loss) / entry) * 100
                    reward_pct = abs((take_profit - entry) / entry) * 100
                    rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                    
                    # Determine trade strategy based on targets
                    if reward_pct >= 5.0:
                        strategy_type = "Long-term Swing"
                        expected_duration = "2-7 days"
                    elif reward_pct >= 3.0:
                        strategy_type = "Swing Trade"
                        expected_duration = "1-3 days"
                    elif reward_pct >= 1.5:
                        strategy_type = "Short-term Swing"
                        expected_duration = "4-24 hours"
                    else:
                        strategy_type = "Scalp Trade"
                        expected_duration = "1-4 hours"
                    
                    html_parts.append(f'<div class="detail-row"><strong>Strategy Type:</strong> {strategy_type}</div>')
                    html_parts.append(f'<div class="detail-row"><strong>Expected Duration:</strong> {expected_duration}</div>')
                    html_parts.append(f'<div class="detail-row"><strong>Risk:</strong> {risk_pct:.1f}% | <strong>Reward:</strong> {reward_pct:.1f}% | <strong>R:R:</strong> {rr_ratio:.1f}:1</div>')
                
                html_parts.append(f'<div class="detail-row"><strong>Stop Loss:</strong> ${stop_loss:.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Take Profit:</strong> ${take_profit:.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Max Favorable:</strong> <span class="positive">${trade_info.get("max_favorable", 0):.2f}</span></div>')
                html_parts.append(f'<div class="detail-row"><strong>Max Adverse:</strong> <span class="negative">${trade_info.get("max_adverse", 0):.2f}</span></div>')
                
                # Related signal info if available
                signal_id = trade_info.get('trade_id', '')
                if signal_id:
                    html_parts.append(f'<div class="detail-row"><strong>Related Signal:</strong> {signal_id}</div>')
                
                html_parts.append(f'</div>')
                html_parts.append(f'</div>')
        else:
            html_parts.append('<div class="stat"><span>No active trades</span></div>')
        html_parts.append('</div>')
        
        # Completed trades card with expandable details
        closed_trades = data.get('closed_trades', {})
        html_parts.append('<div class="card">')
        html_parts.append('<h3>✅ Completed Trades</h3>')
        if closed_trades:
            for idx, (trade_id, trade_info) in enumerate(list(closed_trades.items())[:10]):  # Show max 10
                symbol = trade_info.get('symbol', 'Unknown')
                side = trade_info.get('action', trade_info.get('side', 'Unknown'))
                size = trade_info.get('position_size', trade_info.get('size', 0.0))
                entry = trade_info.get('entry_price', 0.0)
                close_price = trade_info.get('close_price', 0.0)
                final_pnl = trade_info.get('final_pnl', trade_info.get('current_pnl', 0.0))
                close_reason = trade_info.get('close_reason', 'Unknown')
                
                pnl_class = 'positive' if final_pnl >= 0 else 'negative'
                trade_class = 'profit' if final_pnl >= 0 else 'loss'
                
                html_parts.append(f'<div class="completed-trade-item expandable {trade_class}" onclick="toggleCompletedTrade({idx})">')
                html_parts.append(f'<div class="trade-header">')
                html_parts.append(f'<div class="trade-summary">')
                html_parts.append(f'<span><strong>{symbol}</strong> {side} | Size: {size:.4f}</span>')
                html_parts.append(f'<span class="trade-pnl {pnl_class}">${final_pnl:+.2f}</span>')
                html_parts.append(f'</div>')
                html_parts.append(f'<div style="font-size: 0.8em; opacity: 0.8;">Reason: {close_reason}</div>')
                html_parts.append(f'<span class="expand-icon">▼</span>')
                html_parts.append(f'</div>')
                
                # Expandable completed trade details
                html_parts.append(f'<div id="completed-trade-{idx}" class="trade-details" style="display: none;">')
                html_parts.append(f'<div class="detail-row"><strong>Trade ID:</strong> {trade_id}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Entry Price:</strong> ${entry:.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Close Price:</strong> ${close_price:.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Position Size:</strong> {size:.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Final P&L:</strong> <span class="{pnl_class}">${final_pnl:+.2f}</span></div>')
                html_parts.append(f'<div class="detail-row"><strong>Stop Loss:</strong> ${trade_info.get("stop_loss", 0):.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Take Profit:</strong> ${trade_info.get("take_profit", 0):.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Max Favorable:</strong> <span class="positive">${trade_info.get("max_favorable", 0):.2f}</span></div>')
                html_parts.append(f'<div class="detail-row"><strong>Max Adverse:</strong> <span class="negative">${trade_info.get("max_adverse", 0):.2f}</span></div>')
                html_parts.append(f'<div class="detail-row"><strong>Close Reason:</strong> {close_reason}</div>')
                
                # Calculate trade performance metrics
                if entry > 0 and close_price > 0:
                    if side.upper() in ['BUY', 'LONG']:
                        price_change = ((close_price - entry) / entry) * 100
                    else:  # SELL/SHORT
                        price_change = ((entry - close_price) / entry) * 100
                    
                    change_class = 'positive' if price_change >= 0 else 'negative'
                    html_parts.append(f'<div class="detail-row"><strong>Price Change:</strong> <span class="{change_class}">{price_change:+.2f}%</span></div>')
                
                # Related signal info if available
                signal_id = trade_info.get('trade_id', '')
                if signal_id:
                    html_parts.append(f'<div class="detail-row"><strong>Related Signal:</strong> {signal_id}</div>')
                
                html_parts.append(f'</div>')
                html_parts.append(f'</div>')
        else:
            html_parts.append('<div class="stat"><span>No completed trades</span></div>')
        html_parts.append('</div>')
        
        html_parts.append('</div>')  # End grid
        
        # Recent signals section with expandable details
        html_parts.append('<div class="card">')
        html_parts.append('<h3>🎯 Recent Signals</h3>')
        
        if recent_signals:
            for idx, signal in enumerate(recent_signals):
                symbol = signal.get('symbol', 'Unknown')
                action = signal.get('action', 'Unknown')
                score = signal.get('validation_score', 0.0)
                status = signal.get('execution_status', signal.get('validation_result', 'Unknown'))
                timestamp = signal.get('timestamp', '')
                confidence = signal.get('confidence', 0.0) * 100 if signal.get('confidence', 0.0) <= 1 else signal.get('confidence', 0.0)
                
                status_class = 'positive' if status == 'executed' else 'negative' if status in ['rejected', 'REJECTED'] else 'neutral'
                
                html_parts.append(f'<div class="signal-item expandable" onclick="toggleSignal({idx})">')
                html_parts.append(f'<div class="signal-header">')
                # Convert score to percentage if it's a decimal (0.415 -> 41.5%)
                display_score = score * 100 if score <= 1.0 else score
                html_parts.append(f'<strong>{symbol}</strong> {action} | Score: {display_score:.1f}% | ')
                html_parts.append(f'<span class="{status_class}">{status}</span> | {timestamp}')
                html_parts.append(f'<span class="expand-icon">▼</span>')
                html_parts.append(f'</div>')
                
                # Expandable details
                html_parts.append(f'<div id="signal-{idx}" class="signal-details" style="display: none;">')
                html_parts.append(f'<div class="detail-row"><strong>Confidence:</strong> {confidence:.1f}%</div>')
                html_parts.append(f'<div class="detail-row"><strong>Entry Price:</strong> ${signal.get("entry_price", 0):.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Stop Loss:</strong> ${signal.get("stop_loss", 0):.6f}</div>')
                html_parts.append(f'<div class="detail-row"><strong>Take Profit:</strong> ${signal.get("take_profit", 0):.6f}</div>')
                
                # Enhanced 5-Factor Breakdown with detailed justification
                html_parts.append(f'<div class="factor-breakdown">')
                html_parts.append(f'<strong>5-Factor Analysis (Total: {display_score:.1f}%):</strong>')
                
                # Get detailed breakdown if available
                breakdown = signal.get('detailed_validation_breakdown', {})
                raw_scores = breakdown.get('raw_scores', {})
                weighted_scores = breakdown.get('weighted_scores', {})
                
                # Indicator Confluence (30% weight)
                indicator_score = signal.get("indicator_confluence_score", 0)
                indicator_details = breakdown.get('indicator_details', [])
                html_parts.append(f'<div class="factor">')
                html_parts.append(f'  • <strong>Indicator Confluence:</strong> {indicator_score:.1f}% (Weight: 30%)')
                html_parts.append(f'    <div class="factor-details">')
                html_parts.append(f'      Raw Score: {raw_scores.get("indicator_raw", 0):.1%} → Weighted: {weighted_scores.get("indicator_weighted", 0)*100:.1f}%')
                if indicator_details:
                    html_parts.append(f'      <br>Details: {", ".join(indicator_details)}')
                html_parts.append(f'    </div>')
                html_parts.append(f'</div>')
                
                # Timeframe Alignment (25% weight)
                timeframe_score = signal.get("timeframe_alignment_score", 0)
                timeframe_details = breakdown.get('timeframe_details', [])
                html_parts.append(f'<div class="factor">')
                html_parts.append(f'  • <strong>Timeframe Alignment:</strong> {timeframe_score:.1f}% (Weight: 25%)')
                html_parts.append(f'    <div class="factor-details">')
                html_parts.append(f'      Raw Score: {raw_scores.get("timeframe_raw", 0):.1%} → Weighted: {weighted_scores.get("timeframe_weighted", 0)*100:.1f}%')
                if timeframe_details:
                    html_parts.append(f'      <br>Details: {", ".join(timeframe_details)}')
                html_parts.append(f'    </div>')
                html_parts.append(f'</div>')
                
                # Volume Confirmation (20% weight)
                volume_score = signal.get("volume_confirmation_score", 0)
                volume_details = breakdown.get('volume_details', [])
                html_parts.append(f'<div class="factor">')
                html_parts.append(f'  • <strong>Volume Confirmation:</strong> {volume_score:.1f}% (Weight: 20%)')
                html_parts.append(f'    <div class="factor-details">')
                html_parts.append(f'      Raw Score: {raw_scores.get("volume_raw", 0):.1%} → Weighted: {weighted_scores.get("volume_weighted", 0)*100:.1f}%')
                if volume_details:
                    html_parts.append(f'      <br>Details: {", ".join(volume_details)}')
                html_parts.append(f'    </div>')
                html_parts.append(f'</div>')
                
                # Market Structure (15% weight)
                structure_score = signal.get("market_structure_score", 0)
                structure_details = breakdown.get('structure_details', [])
                html_parts.append(f'<div class="factor">')
                html_parts.append(f'  • <strong>Market Structure:</strong> {structure_score:.1f}% (Weight: 15%)')
                html_parts.append(f'    <div class="factor-details">')
                html_parts.append(f'      Raw Score: {raw_scores.get("structure_raw", 0):.1%} → Weighted: {weighted_scores.get("structure_weighted", 0)*100:.1f}%')
                if structure_details:
                    html_parts.append(f'      <br>Details: {", ".join(structure_details)}')
                html_parts.append(f'    </div>')
                html_parts.append(f'</div>')
                
                # Risk/Reward (10% weight)
                rr_score = signal.get("risk_reward_score", 0)
                rr_details = breakdown.get('risk_reward_details', [])
                html_parts.append(f'<div class="factor">')
                html_parts.append(f'  • <strong>Risk/Reward Ratio:</strong> {rr_score:.1f}% (Weight: 10%)')
                html_parts.append(f'    <div class="factor-details">')
                html_parts.append(f'      Raw Score: {raw_scores.get("risk_reward_raw", 0):.1%} → Weighted: {weighted_scores.get("risk_reward_weighted", 0)*100:.1f}%')
                if rr_details:
                    html_parts.append(f'      <br>Details: {", ".join(rr_details)}')
                html_parts.append(f'    </div>')
                html_parts.append(f'</div>')
                
                html_parts.append(f'</div>')
                
                # Add rejection reasons if signal was rejected
                rejection_reason = signal.get('rejection_reason', '')
                if rejection_reason and status in ['rejected', 'REJECTED']:
                    html_parts.append(f'<div class="rejection-reason">')
                    html_parts.append(f'<strong>⚠️ Rejection Reason:</strong> {rejection_reason}')
                    html_parts.append(f'</div>')
                
                # Rejection reason if applicable
                rejection_reason = signal.get('rejection_reason', '')
                if rejection_reason:
                    html_parts.append(f'<div class="detail-row"><strong>Rejection Reason:</strong> {rejection_reason}</div>')
                
                html_parts.append(f'</div>')
                html_parts.append(f'</div>')
        else:
            html_parts.append('<div class="stat"><span>No recent signals</span></div>')
            
        html_parts.append('</div>')
        
        # Auto-refresh script and footer
        html_parts.append('''
        <div class="refresh-info">
            <p>🔴 LIVE: Dashboard auto-refreshes every 5 seconds | WebSocket data streaming</p>
        </div>
        
        <script>
            // State preservation for dashboard refreshes
            let expandedStates = JSON.parse(localStorage.getItem('dashboardExpanded') || '{}');
            
            // Restore expanded states on page load
            function restoreStates() {
                for (const [id, isExpanded] of Object.entries(expandedStates)) {
                    const details = document.getElementById(id);
                    const button = document.querySelector(`[onclick*="${id}"]`);
                    if (details && button) {
                        const icon = button.querySelector('.expand-icon');
                        if (isExpanded) {
                            details.style.display = 'block';
                            if (icon) icon.style.transform = 'rotate(180deg)';
                        } else {
                            details.style.display = 'none';
                            if (icon) icon.style.transform = 'rotate(0deg)';
                        }
                    }
                }
            }
            
            // Save state to localStorage
            function saveState(elementId, isExpanded) {
                expandedStates[elementId] = isExpanded;
                localStorage.setItem('dashboardExpanded', JSON.stringify(expandedStates));
            }
            
            // Expand/collapse signal details
            function toggleSignal(idx) {
                const details = document.getElementById('signal-' + idx);
                const icon = event.currentTarget.querySelector('.expand-icon');
                
                if (details.style.display === 'none') {
                    details.style.display = 'block';
                    icon.style.transform = 'rotate(180deg)';
                    saveState('signal-' + idx, true);
                } else {
                    details.style.display = 'none';
                    icon.style.transform = 'rotate(0deg)';
                    saveState('signal-' + idx, false);
                }
            }
            
            // Expand/collapse trade details
            function toggleTrade(idx) {
                const details = document.getElementById('trade-' + idx);
                const icon = event.currentTarget.querySelector('.expand-icon');
                
                if (details.style.display === 'none') {
                    details.style.display = 'block';
                    icon.style.transform = 'rotate(180deg)';
                    saveState('trade-' + idx, true);
                } else {
                    details.style.display = 'none';
                    icon.style.transform = 'rotate(0deg)';
                    saveState('trade-' + idx, false);
                }
            }
            
            // Expand/collapse completed trade details
            function toggleCompletedTrade(idx) {
                const details = document.getElementById('completed-trade-' + idx);
                const icon = event.currentTarget.querySelector('.expand-icon');
                
                if (details.style.display === 'none') {
                    details.style.display = 'block';
                    icon.style.transform = 'rotate(180deg)';
                    saveState('completed-trade-' + idx, true);
                } else {
                    details.style.display = 'none';
                    icon.style.transform = 'rotate(0deg)';
                    saveState('completed-trade-' + idx, false);
                }
            }
            
            // Restore states when page loads
            restoreStates();
            
            // Refresh every 30 seconds for live WebSocket data (preserving states)
            setTimeout(function() {
                location.reload();
            }, 30000);
        </script>
        
        </div>
        </body>
        </html>''')
        
        return ''.join(html_parts)
    
    def _generate_error_html(self, error: str) -> str:
        """Generate error page"""
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>Dashboard Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }}
        .error {{ background: #ffebee; color: #c62828; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="error">
        <h2>Dashboard Error</h2>
        <p>{error}</p>
        <p>Refreshing in 10 seconds...</p>
    </div>
    <script>setTimeout(() => location.reload(), 10000);</script>
</body>
</html>'''
    
    def _get_mock_data(self) -> Dict[str, Any]:
        """Return empty data structure - NO MOCK DATA EVER"""
        return {
            'performance_stats': {},
            'validation_stats': {},
            'price_monitor': {'monitored_pairs': 0, 'prices': {}},
            'recent_signals': [],
            'active_trades': {},
            'closed_trades': {}
        }

class WorkingDashboardServer:
    """Simple, working dashboard server"""
    
    def __init__(self, dashboard_manager=None, port=8080):
        self.dashboard_manager = dashboard_manager
        self.port = port
        self.server = None
        self.server_thread = None
        
    def start_server(self):
        """Start the dashboard server"""
        try:
            # Create handler with dashboard_manager reference
            def handler_factory(*args, **kwargs):
                return WorkingDashboardHandler(*args, dashboard_manager=self.dashboard_manager, **kwargs)
            
            self.server = HTTPServer(('localhost', self.port), handler_factory)
            
            print(f"🚀 Working Dashboard Server starting on port {self.port}")
            print(f"📊 Dashboard URL: http://localhost:{self.port}/dashboard")
            print(f"🔄 Auto-refresh: Every 30 seconds")
            print(f"⏹️  Call stop_server() to stop")
            
            # Run server in thread
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            return False
    
    def stop_server(self):
        """Stop the dashboard server"""
        if self.server:
            print("🛑 Stopping dashboard server...")
            self.server.shutdown()
            self.server.server_close()
            if self.server_thread:
                self.server_thread.join(timeout=2)
            print("✅ Server stopped")

def run_standalone_test():
    """Run standalone dashboard test"""
    dashboard = WorkingDashboardServer()
    
    if dashboard.start_server():
        print("\n🎯 Dashboard is running! Open http://localhost:8080/dashboard")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            dashboard.stop_server()

if __name__ == "__main__":
    run_standalone_test()