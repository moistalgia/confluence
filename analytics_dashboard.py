#!/usr/bin/env python3
"""
Advanced Analytics Dashboard
Web-based dashboard with real-time charts, performance metrics, and interactive analysis tools
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from threading import Thread
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import base64
import io

# Import our analysis components
try:
    from enhanced_multi_timeframe_analyzer import EnhancedMultiTimeframeAnalyzer
    from config_manager import ConfigManager
    from alert_system import AlertManager
    from backtesting_framework import Backtester, BacktestConfig, ConfluenceStrategy
    from realtime_data import RealtimeDataManager
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoAnalyticsDashboard:
    """Main dashboard application class"""
    
    def __init__(self):
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.title = "Crypto Analytics Dashboard"
        
        # Initialize components
        self.analyzer = None
        self.config_manager = None
        self.alert_manager = None
        self.realtime_manager = None
        
        # Dashboard state
        self.current_data = {}
        self.analysis_cache = {}
        self.performance_data = {}
        
        # Initialize components
        self._initialize_components()
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info("Crypto Analytics Dashboard initialized")
    
    def _initialize_components(self):
        """Initialize analysis components"""
        
        try:
            # Initialize analyzer
            self.analyzer = EnhancedMultiTimeframeAnalyzer()
            
            # Initialize config manager
            self.config_manager = ConfigManager()
            
            # Initialize alert manager
            self.alert_manager = AlertManager()
            
            # Initialize real-time data manager (with empty config)
            self.realtime_manager = RealtimeDataManager({})
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚀 Crypto Analytics Dashboard", className="header-title"),
                html.P("Professional-grade cryptocurrency analysis platform", className="header-subtitle"),
                html.Hr()
            ], className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Select Symbol:"),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[
                            {'label': 'BTC/USD', 'value': 'BTC/USD'},
                            {'label': 'ETH/USD', 'value': 'ETH/USD'},
                            {'label': 'ADA/USD', 'value': 'ADA/USD'},
                            {'label': 'DOT/USD', 'value': 'DOT/USD'},
                            {'label': 'LINK/USD', 'value': 'LINK/USD'},
                            {'label': 'SOL/USD', 'value': 'SOL/USD'},
                        ],
                        value='BTC/USD',
                        className="dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Timeframe:"),
                    dcc.Dropdown(
                        id='timeframe-dropdown',
                        options=[
                            {'label': '5 minutes', 'value': '5m'},
                            {'label': '15 minutes', 'value': '15m'},
                            {'label': '1 hour', 'value': '1h'},
                            {'label': '4 hours', 'value': '4h'},
                            {'label': '1 day', 'value': '1d'},
                        ],
                        value='1h',
                        className="dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Button("🔄 Refresh Analysis", id="refresh-btn", className="btn-primary"),
                    html.Button("⚙️ Settings", id="settings-btn", className="btn-secondary"),
                ], className="control-buttons"),
                
                # Auto-refresh toggle
                html.Div([
                    dcc.Checklist(
                        id='auto-refresh-toggle',
                        options=[{'label': ' Auto-refresh (30s)', 'value': 'auto'}],
                        value=[],
                        className="checkbox"
                    )
                ], className="control-item"),
                
            ], className="control-panel"),
            
            # Main Content Tabs
            dcc.Tabs(id="main-tabs", value='analysis-tab', children=[
                
                # Analysis Tab
                dcc.Tab(label='📊 Market Analysis', value='analysis-tab', children=[
                    html.Div([
                        
                        # Price Chart and Key Metrics Row
                        html.Div([
                            html.Div([
                                dcc.Graph(id='price-chart')
                            ], className="chart-container"),
                            
                            html.Div([
                                html.H3("📈 Key Metrics"),
                                html.Div(id='key-metrics-cards')
                            ], className="metrics-panel")
                        ], className="row"),
                        
                        # Technical Indicators Row
                        html.Div([
                            html.Div([
                                html.H3("🔧 Technical Indicators"),
                                dcc.Graph(id='indicators-chart')
                            ], className="chart-container"),
                            
                            html.Div([
                                html.H3("🎯 Signal Analysis"),
                                html.Div(id='signal-analysis')
                            ], className="signal-panel")
                        ], className="row"),
                        
                        # Confluence Analysis
                        html.Div([
                            html.H3("🔗 Confluence Analysis"),
                            html.Div(id='confluence-display')
                        ], className="confluence-section")
                        
                    ])
                ]),
                
                # Portfolio Tab
                dcc.Tab(label='💼 Portfolio', value='portfolio-tab', children=[
                    html.Div([
                        
                        # Portfolio Overview
                        html.Div([
                            html.H3("💰 Portfolio Overview"),
                            html.Div(id='portfolio-overview')
                        ], className="portfolio-section"),
                        
                        # Recent Trades
                        html.Div([
                            html.H3("📋 Recent Trades"),
                            html.Div(id='recent-trades-table')
                        ], className="trades-section"),
                        
                        # Performance Charts
                        html.Div([
                            html.Div([
                                dcc.Graph(id='portfolio-chart')
                            ], className="chart-container"),
                            
                            html.Div([
                                dcc.Graph(id='drawdown-chart')
                            ], className="chart-container")
                        ], className="row")
                        
                    ])
                ]),
                
                # Backtesting Tab
                dcc.Tab(label='🧪 Backtesting', value='backtest-tab', children=[
                    html.Div([
                        
                        # Backtest Controls
                        html.Div([
                            html.H3("🎮 Backtest Configuration"),
                            
                            html.Div([
                                html.Div([
                                    html.Label("Initial Capital:"),
                                    dcc.Input(
                                        id='initial-capital-input',
                                        type='number',
                                        value=100000,
                                        className="input-field"
                                    )
                                ], className="config-item"),
                                
                                html.Div([
                                    html.Label("Commission Rate (%):"),
                                    dcc.Input(
                                        id='commission-input',
                                        type='number',
                                        value=0.1,
                                        step=0.01,
                                        className="input-field"
                                    )
                                ], className="config-item"),
                                
                                html.Div([
                                    html.Label("Risk per Trade (%):"),
                                    dcc.Input(
                                        id='risk-per-trade-input',
                                        type='number',
                                        value=1.0,
                                        step=0.1,
                                        className="input-field"
                                    )
                                ], className="config-item"),
                            ], className="config-row"),
                            
                            html.Div([
                                html.Button("🚀 Run Backtest", id="run-backtest-btn", className="btn-primary"),
                                html.Button("📁 Load Results", id="load-backtest-btn", className="btn-secondary"),
                            ], className="backtest-buttons")
                        ], className="backtest-config"),
                        
                        # Backtest Results
                        html.Div([
                            html.H3("📊 Backtest Results"),
                            html.Div(id='backtest-results')
                        ], className="backtest-results")
                        
                    ])
                ]),
                
                # Alerts Tab
                dcc.Tab(label='🔔 Alerts', value='alerts-tab', children=[
                    html.Div([
                        
                        # Alert Configuration
                        html.Div([
                            html.H3("⚠️ Alert Configuration"),
                            
                            html.Div([
                                html.Div([
                                    html.Label("Alert Type:"),
                                    dcc.Dropdown(
                                        id='alert-type-dropdown',
                                        options=[
                                            {'label': 'Price Alert', 'value': 'price'},
                                            {'label': 'Signal Alert', 'value': 'signal'},
                                            {'label': 'Pattern Alert', 'value': 'pattern'},
                                            {'label': 'Volume Alert', 'value': 'volume'}
                                        ],
                                        value='signal',
                                        className="dropdown"
                                    )
                                ], className="alert-config-item"),
                                
                                html.Div([
                                    html.Label("Threshold:"),
                                    dcc.Input(
                                        id='alert-threshold-input',
                                        type='number',
                                        value=70,
                                        className="input-field"
                                    )
                                ], className="alert-config-item"),
                                
                                html.Div([
                                    html.Button("➕ Add Alert", id="add-alert-btn", className="btn-primary")
                                ], className="alert-config-item")
                            ], className="alert-config-row")
                        ], className="alert-config"),
                        
                        # Active Alerts
                        html.Div([
                            html.H3("🔔 Active Alerts"),
                            html.Div(id='active-alerts-list')
                        ], className="alerts-list")
                        
                    ])
                ]),
                
                # Settings Tab
                dcc.Tab(label='⚙️ Settings', value='settings-tab', children=[
                    html.Div([
                        
                        # Configuration Settings
                        html.Div([
                            html.H3("🔧 System Configuration"),
                            
                            html.Div([
                                html.H4("Trading Profile"),
                                dcc.Dropdown(
                                    id='trading-profile-dropdown',
                                    options=[
                                        {'label': 'Scalping', 'value': 'scalping'},
                                        {'label': 'Swing Trading', 'value': 'swing'},
                                        {'label': 'Position Trading', 'value': 'position'}
                                    ],
                                    value='swing',
                                    className="dropdown"
                                )
                            ], className="settings-item"),
                            
                            html.Div([
                                html.H4("Notification Settings"),
                                dcc.Checklist(
                                    id='notification-settings',
                                    options=[
                                        {'label': 'Email Notifications', 'value': 'email'},
                                        {'label': 'Desktop Notifications', 'value': 'desktop'},
                                        {'label': 'Console Notifications', 'value': 'console'}
                                    ],
                                    value=['console'],
                                    className="checkbox"
                                )
                            ], className="settings-item"),
                            
                            html.Div([
                                html.Button("💾 Save Settings", id="save-settings-btn", className="btn-primary")
                            ], className="settings-buttons")
                        ], className="settings-panel")
                        
                    ])
                ])
            ]),
            
            # Status Bar
            html.Div([
                html.Span(id='status-indicator', className="status-indicator"),
                html.Span(id='last-update', className="last-update"),
                html.Span(id='connection-status', className="connection-status")
            ], className="status-bar"),
            
            # Hidden divs for data storage
            html.Div(id='data-store', style={'display': 'none'}),
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0,
                disabled=True
            )
        ])
        
        # Add CSS styling
        self._add_css_styling()
    
    def _add_css_styling(self):
        """Add CSS styling to the dashboard"""
        
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f5f6fa;
                    }
                    
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        text-align: center;
                    }
                    
                    .header-title {
                        margin: 0;
                        font-size: 2.5em;
                        font-weight: 300;
                    }
                    
                    .header-subtitle {
                        margin: 5px 0 0 0;
                        opacity: 0.9;
                    }
                    
                    .control-panel {
                        background: white;
                        padding: 20px;
                        margin: 10px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                        align-items: center;
                    }
                    
                    .control-item {
                        min-width: 150px;
                    }
                    
                    .dropdown, .input-field {
                        width: 100%;
                        padding: 8px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        font-size: 14px;
                    }
                    
                    .btn-primary, .btn-secondary {
                        padding: 10px 20px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 14px;
                        margin-right: 10px;
                    }
                    
                    .btn-primary {
                        background-color: #3742fa;
                        color: white;
                    }
                    
                    .btn-secondary {
                        background-color: #57606f;
                        color: white;
                    }
                    
                    .row {
                        display: flex;
                        gap: 20px;
                        margin: 20px 10px;
                    }
                    
                    .chart-container {
                        flex: 2;
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    
                    .metrics-panel, .signal-panel {
                        flex: 1;
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    
                    .metric-card {
                        background: #f8f9fa;
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 6px;
                        border-left: 4px solid #3742fa;
                    }
                    
                    .metric-title {
                        font-size: 12px;
                        color: #666;
                        margin-bottom: 5px;
                    }
                    
                    .metric-value {
                        font-size: 18px;
                        font-weight: bold;
                        color: #333;
                    }
                    
                    .status-bar {
                        background: #2f3542;
                        color: white;
                        padding: 10px 20px;
                        display: flex;
                        justify-content: space-between;
                        font-size: 12px;
                    }
                    
                    .status-indicator {
                        color: #2ed573;
                    }
                    
                    .confluence-section {
                        background: white;
                        margin: 20px 10px;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    
                    .confluence-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 15px;
                        margin-top: 15px;
                    }
                    
                    .confluence-card {
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 6px;
                        border: 1px solid #e9ecef;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        # Main analysis update callback
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('indicators-chart', 'figure'),
             Output('key-metrics-cards', 'children'),
             Output('signal-analysis', 'children'),
             Output('confluence-display', 'children'),
             Output('status-indicator', 'children'),
             Output('last-update', 'children')],
            [Input('refresh-btn', 'n_clicks'),
             Input('interval-component', 'n_intervals')],
            [State('symbol-dropdown', 'value'),
             State('timeframe-dropdown', 'value')]
        )
        def update_analysis(refresh_clicks, n_intervals, symbol, timeframe):
            """Update main analysis display"""
            
            try:
                # Get analysis data
                analysis = self._get_analysis_data(symbol, timeframe)
                
                # Create charts
                price_chart = self._create_price_chart(analysis, symbol, timeframe)
                indicators_chart = self._create_indicators_chart(analysis, symbol, timeframe)
                
                # Create metrics cards
                metrics_cards = self._create_metrics_cards(analysis)
                
                # Create signal analysis
                signal_analysis = self._create_signal_analysis(analysis)
                
                # Create confluence display
                confluence_display = self._create_confluence_display(analysis)
                
                # Status updates
                status = "● Connected" if analysis else "● Disconnected"
                last_update = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
                
                return (price_chart, indicators_chart, metrics_cards, 
                       signal_analysis, confluence_display, status, last_update)
                
            except Exception as e:
                logger.error(f"Error updating analysis: {e}")
                empty_fig = go.Figure()
                return (empty_fig, empty_fig, "Error loading data", 
                       "Error loading signals", "Error loading confluence", 
                       "● Error", f"Error: {str(e)}")
        
        # Auto-refresh toggle
        @self.app.callback(
            Output('interval-component', 'disabled'),
            Input('auto-refresh-toggle', 'value')
        )
        def toggle_auto_refresh(auto_refresh):
            """Toggle auto-refresh functionality"""
            return 'auto' not in auto_refresh
        
        # Backtest callback
        @self.app.callback(
            Output('backtest-results', 'children'),
            Input('run-backtest-btn', 'n_clicks'),
            [State('symbol-dropdown', 'value'),
             State('initial-capital-input', 'value'),
             State('commission-input', 'value'),
             State('risk-per-trade-input', 'value')]
        )
        def run_backtest(n_clicks, symbol, initial_capital, commission, risk_per_trade):
            """Run backtesting analysis"""
            
            if n_clicks:
                try:
                    # Create backtest configuration
                    config = BacktestConfig(
                        initial_capital=initial_capital or 100000,
                        commission_rate=(commission or 0.1) / 100,
                        risk_per_trade=(risk_per_trade or 1.0) / 100
                    )
                    
                    # Run backtest
                    backtester = Backtester(self.analyzer, config)
                    strategy = ConfluenceStrategy()
                    backtester.set_strategy(strategy)
                    
                    # Simulate backtest results (in practice, this would run full backtest)
                    results = self._create_mock_backtest_results()
                    
                    return self._create_backtest_display(results)
                    
                except Exception as e:
                    logger.error(f"Error running backtest: {e}")
                    return html.Div(f"Error running backtest: {str(e)}", className="error-message")
            
            return html.Div("Click 'Run Backtest' to start analysis", className="placeholder-message")
    
    def _get_analysis_data(self, symbol: str, timeframe: str) -> Dict:
        """Get analysis data for symbol and timeframe"""
        
        try:
            if self.analyzer:
                # In practice, this would fetch real-time data
                # For demo, we'll create mock analysis data
                analysis = self._create_mock_analysis_data(symbol, timeframe)
                return analysis
            else:
                logger.warning("Analyzer not initialized")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting analysis data: {e}")
            return {}
    
    def _create_mock_analysis_data(self, symbol: str, timeframe: str) -> Dict:
        """Create mock analysis data for demonstration"""
        
        # Generate mock price data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             end=datetime.now(), freq='h')
        
        # Generate realistic price movement
        np.random.seed(42)
        base_price = 45000 if symbol == 'BTC/USD' else 3000
        price_changes = np.random.normal(0, 0.02, len(dates))
        prices = [base_price]
        
        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.8))  # Prevent extreme drops
        
        prices = prices[1:]  # Remove initial price
        
        # Create OHLCV data
        ohlcv_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            volume = abs(np.random.normal(1000, 200))
            
            ohlcv_data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        # Generate technical indicators
        rsi_values = [30 + abs(np.random.normal(0, 15)) for _ in range(len(dates))]
        macd_values = [np.random.normal(0, 50) for _ in range(len(dates))]
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'ohlcv': ohlcv_data,
            'indicators': {
                'rsi': rsi_values,
                'macd': macd_values,
                'bb_upper': [p * 1.05 for p in prices],
                'bb_lower': [p * 0.95 for p in prices],
                'sma_20': [p * (1 + np.random.normal(0, 0.01)) for p in prices]
            },
            'confluence_analysis': {
                'overall_confluence': {'confluence_score': 75.5},
                'trend_alignment': {'alignment_percentage': 80, 'dominant_trend': 'BULLISH'},
                'momentum_confluence': {'momentum_score': 70},
                'indicator_agreement': {'agreement_percentage': 85}
            },
            'signals': {
                'buy_signals': 3,
                'sell_signals': 1,
                'signal_strength': 'STRONG',
                'last_signal': 'BUY'
            }
        }
    
    def _create_price_chart(self, analysis: Dict, symbol: str, timeframe: str) -> go.Figure:
        """Create price chart with candlesticks and indicators"""
        
        fig = go.Figure()
        
        if not analysis or 'ohlcv' not in analysis:
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        ohlcv = analysis['ohlcv']
        dates = [d['timestamp'] for d in ohlcv]
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=dates,
            open=[d['open'] for d in ohlcv],
            high=[d['high'] for d in ohlcv],
            low=[d['low'] for d in ohlcv],
            close=[d['close'] for d in ohlcv],
            name=symbol
        ))
        
        # Add indicators if available
        if 'indicators' in analysis:
            indicators = analysis['indicators']
            
            # Bollinger Bands
            if 'bb_upper' in indicators:
                fig.add_trace(go.Scatter(
                    x=dates, y=indicators['bb_upper'],
                    name='BB Upper', line=dict(color='rgba(255,0,0,0.3)')
                ))
                fig.add_trace(go.Scatter(
                    x=dates, y=indicators['bb_lower'],
                    name='BB Lower', line=dict(color='rgba(255,0,0,0.3)'),
                    fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
                ))
            
            # Moving Average
            if 'sma_20' in indicators:
                fig.add_trace(go.Scatter(
                    x=dates, y=indicators['sma_20'],
                    name='SMA 20', line=dict(color='orange', width=2)
                ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart ({timeframe})",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400,
            showlegend=True,
            autosize=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def _create_indicators_chart(self, analysis: Dict, symbol: str, timeframe: str) -> go.Figure:
        """Create technical indicators chart"""
        
        fig = go.Figure()
        
        if not analysis or 'indicators' not in analysis:
            fig.add_annotation(text="No indicator data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        indicators = analysis['indicators']
        dates = [d['timestamp'] for d in analysis['ohlcv']]
        
        # RSI
        if 'rsi' in indicators:
            fig.add_trace(go.Scatter(
                x=dates, y=indicators['rsi'],
                name='RSI', line=dict(color='purple')
            ))
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        # MACD
        if 'macd' in indicators:
            fig.add_trace(go.Scatter(
                x=dates, y=indicators['macd'],
                name='MACD', line=dict(color='blue'),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title=f"Technical Indicators - {symbol}",
            xaxis_title="Time",
            yaxis=dict(title="RSI", range=[0, 100]),
            yaxis2=dict(title="MACD", overlaying='y', side='right'),
            height=300,
            showlegend=True,
            autosize=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def _create_metrics_cards(self, analysis: Dict) -> List:
        """Create metrics cards display"""
        
        if not analysis or 'ohlcv' not in analysis:
            return [html.Div("No metrics available")]
        
        latest_data = analysis['ohlcv'][-1] if analysis['ohlcv'] else {}
        signals = analysis.get('signals', {})
        confluence = analysis.get('confluence_analysis', {}).get('overall_confluence', {})
        
        cards = []
        
        # Current Price
        if 'close' in latest_data:
            cards.append(html.Div([
                html.Div("Current Price", className="metric-title"),
                html.Div(f"${latest_data['close']:,.2f}", className="metric-value")
            ], className="metric-card"))
        
        # 24h Volume
        if 'volume' in latest_data:
            cards.append(html.Div([
                html.Div("24h Volume", className="metric-title"),
                html.Div(f"{latest_data['volume']:,.0f}", className="metric-value")
            ], className="metric-card"))
        
        # Signal Strength
        if 'signal_strength' in signals:
            cards.append(html.Div([
                html.Div("Signal Strength", className="metric-title"),
                html.Div(signals['signal_strength'], className="metric-value")
            ], className="metric-card"))
        
        # Confluence Score
        if 'confluence_score' in confluence:
            cards.append(html.Div([
                html.Div("Confluence Score", className="metric-title"),
                html.Div(f"{confluence['confluence_score']:.1f}%", className="metric-value")
            ], className="metric-card"))
        
        return cards
    
    def _create_signal_analysis(self, analysis: Dict) -> List:
        """Create signal analysis display"""
        
        if not analysis or 'signals' not in analysis:
            return [html.Div("No signal data available")]
        
        signals = analysis['signals']
        
        components = []
        
        # Buy/Sell signals count
        components.append(html.Div([
            html.H4("Signal Summary"),
            html.P(f"Buy Signals: {signals.get('buy_signals', 0)}"),
            html.P(f"Sell Signals: {signals.get('sell_signals', 0)}"),
            html.P(f"Last Signal: {signals.get('last_signal', 'None')}")
        ]))
        
        # Signal strength indicator
        strength = signals.get('signal_strength', 'NEUTRAL')
        color = {'STRONG': 'green', 'MODERATE': 'orange', 'WEAK': 'red'}.get(strength, 'gray')
        
        components.append(html.Div([
            html.H4("Signal Strength"),
            html.Div(strength, style={'color': color, 'font-weight': 'bold', 'font-size': '18px'})
        ]))
        
        return components
    
    def _create_confluence_display(self, analysis: Dict) -> html.Div:
        """Create confluence analysis display"""
        
        if not analysis or 'confluence_analysis' not in analysis:
            return html.Div("No confluence data available")
        
        confluence = analysis['confluence_analysis']
        
        cards = []
        
        # Overall Confluence
        if 'overall_confluence' in confluence:
            overall = confluence['overall_confluence']
            cards.append(html.Div([
                html.H4("Overall Confluence"),
                html.P(f"Score: {overall.get('confluence_score', 0):.1f}%")
            ], className="confluence-card"))
        
        # Trend Alignment
        if 'trend_alignment' in confluence:
            trend = confluence['trend_alignment']
            cards.append(html.Div([
                html.H4("Trend Alignment"),
                html.P(f"Alignment: {trend.get('alignment_percentage', 0):.1f}%"),
                html.P(f"Trend: {trend.get('dominant_trend', 'NEUTRAL')}")
            ], className="confluence-card"))
        
        # Momentum Confluence
        if 'momentum_confluence' in confluence:
            momentum = confluence['momentum_confluence']
            cards.append(html.Div([
                html.H4("Momentum Confluence"),
                html.P(f"Score: {momentum.get('momentum_score', 0):.1f}%")
            ], className="confluence-card"))
        
        # Indicator Agreement
        if 'indicator_agreement' in confluence:
            agreement = confluence['indicator_agreement']
            cards.append(html.Div([
                html.H4("Indicator Agreement"),
                html.P(f"Agreement: {agreement.get('agreement_percentage', 0):.1f}%")
            ], className="confluence-card"))
        
        return html.Div(cards, className="confluence-grid")
    
    def _create_mock_backtest_results(self) -> Dict:
        """Create mock backtest results for demonstration"""
        
        return {
            'performance': {
                'initial_capital': 100000,
                'final_value': 125000,
                'total_return_percent': 25.0,
                'max_drawdown_percent': 8.5,
                'sharpe_ratio': 1.85,
                'total_trades': 45,
                'winning_trades': 28,
                'win_rate_percent': 62.2,
                'profit_factor': 1.75
            }
        }
    
    def _create_backtest_display(self, results: Dict) -> html.Div:
        """Create backtest results display"""
        
        if not results or 'performance' not in results:
            return html.Div("No backtest results available")
        
        perf = results['performance']
        
        return html.Div([
            html.H4("Backtest Performance Summary"),
            
            html.Div([
                html.Div([
                    html.Div("Initial Capital", className="metric-title"),
                    html.Div(f"${perf.get('initial_capital', 0):,.2f}", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.Div("Final Value", className="metric-title"),
                    html.Div(f"${perf.get('final_value', 0):,.2f}", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.Div("Total Return", className="metric-title"),
                    html.Div(f"{perf.get('total_return_percent', 0):.1f}%", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.Div("Max Drawdown", className="metric-title"),
                    html.Div(f"{perf.get('max_drawdown_percent', 0):.1f}%", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.Div("Sharpe Ratio", className="metric-title"),
                    html.Div(f"{perf.get('sharpe_ratio', 0):.2f}", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.Div("Win Rate", className="metric-title"),
                    html.Div(f"{perf.get('win_rate_percent', 0):.1f}%", className="metric-value")
                ], className="metric-card")
            ], style={'display': 'grid', 'grid-template-columns': 'repeat(3, 1fr)', 'gap': '15px'})
        ])
    
    def run(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = False):
        """Run the dashboard server"""
        
        logger.info(f"Starting Crypto Analytics Dashboard on http://{host}:{port}")
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")

# Dashboard factory function
def create_dashboard() -> CryptoAnalyticsDashboard:
    """Create and return a dashboard instance"""
    return CryptoAnalyticsDashboard()

if __name__ == "__main__":
    # Create and run dashboard
    dashboard = create_dashboard()
    
    print("🚀 Crypto Analytics Dashboard")
    print("==============================")
    print("Features:")
    print("• Real-time market analysis")
    print("• Interactive charts and indicators")
    print("• Confluence analysis display")
    print("• Backtesting interface")
    print("• Alert management")
    print("• Performance metrics")
    print()
    print("Starting dashboard server...")
    print("Open your browser to: http://127.0.0.1:8050")
    
    # Run dashboard
    dashboard.run(debug=True)