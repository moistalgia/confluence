#!/usr/bin/env python3
"""
LLM Integration System for Ultimate Crypto Analysis
Supports multiple LLM providers: Claude, OpenAI, Anthropic
"""

import os
import json
import logging
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env')

def format_price(price):
    """Format price with appropriate precision based on value"""
    if price == 0:
        return "$0.00"
    
    # For very small prices (< $0.01), use 6-8 decimal places
    if price < 0.01:
        return f"${price:.8f}"
    # For small prices ($0.01 - $1), use 4-6 decimal places  
    elif price < 1:
        return f"${price:.6f}"
    # For mid prices ($1 - $100), use 4 decimal places
    elif price < 100:
        return f"${price:.4f}"
    # For higher prices ($100+), use 2 decimal places with comma separation
    else:
        return f"${price:,.2f}"

def format_distance_percent(percent_value, decimals=1):
    """Format distance percentages clearly without confusing negative signs"""
    if percent_value > 0:
        return f"+{percent_value:.{decimals}f}% above"
    elif percent_value < 0:
        return f"{abs(percent_value):.{decimals}f}% below" 
    else:
        return "at level"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMIntegration:
    """
    Multi-provider LLM integration with intelligent routing
    Supports: Claude Sonnet 4, GPT-4, GPT-3.5-turbo, and more
    """
    
    def __init__(self):
        """Initialize LLM integration with environment configuration"""
        self.load_configuration()
        self.setup_output_directories()
        
    def load_configuration(self):
        """Load LLM configuration from environment variables"""
        
        # Primary Model: Claude Sonnet 4
        self.claude_api_key = os.getenv('ANTHROPIC_API_KEY', os.getenv('CLAUDE_API_KEY', ''))
        self.claude_model = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514')
        
        # Secondary Models
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
        
        # Model Selection Priority  
        default_model = os.getenv('DEFAULT_LLM_MODEL', 'claude-sonnet')
        if 'claude' in default_model.lower():
            self.primary_model = 'claude'
        elif 'openai' in default_model.lower() or 'gpt' in default_model.lower():
            self.primary_model = 'openai'
        else:
            self.primary_model = 'claude'
        
        # Request Configuration
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '4000'))
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
        self.timeout = int(os.getenv('LLM_TIMEOUT', '120'))
        
        logger.info(f"LLM Configuration loaded. Primary model: {self.primary_model}")
        
    def setup_output_directories(self):
        """Create output directories for LLM responses and reports"""
        self.base_dir = Path("output/ultimate_analysis")
        self.llm_responses_dir = self.base_dir / "llm_responses"
        self.rich_reports_dir = self.base_dir / "rich_reports"
        
        for directory in [self.llm_responses_dir, self.rich_reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info("LLM output directories created")
    
    async def process_crypto_analysis(self, symbol: str, analysis_data: Dict) -> Dict:
        """
        Main function to process crypto analysis through LLM
        Returns comprehensive professional analysis
        """
        
        try:
            logger.info(f"🤖 Processing {symbol} analysis through {self.primary_model.upper()}")
            
            # Generate enhanced prompt
            prompt = self._generate_enhanced_prompt(symbol, analysis_data)
            
            # Process through LLM
            llm_response = await self._send_to_llm(prompt, symbol)
            
            if llm_response:
                # Save raw LLM response
                response_file = self._save_llm_response(symbol, llm_response, analysis_data)
                
                # Generate rich report
                report_files = self._generate_rich_reports(symbol, llm_response, analysis_data)
                
                logger.info(f"✅ {symbol} LLM processing complete")
                
                return {
                    'status': 'success',
                    'symbol': symbol,
                    'llm_model': self.primary_model,
                    'response_file': str(response_file),
                    'report_files': report_files,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error(f"❌ LLM processing failed for {symbol}")
                return {'status': 'error', 'symbol': symbol, 'error': 'LLM request failed'}
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            return {'status': 'error', 'symbol': symbol, 'error': str(e)}
    
    def _generate_enhanced_prompt(self, symbol: str, analysis_data: Dict) -> str:
        """Generate the enhanced prompt from analysis data"""
        
        # Import the prompt generator function
        from complete_prompt_generator import generate_complete_ultimate_prompt
        
        prompt = generate_complete_ultimate_prompt(symbol, analysis_data)
        
        # Add model-specific instructions
        enhanced_prompt = f"""
{prompt}

## ANALYSIS REQUIREMENTS

As an **Elite Institutional Trading Professional**, provide a comprehensive analysis structured as follows:

### EXECUTIVE SUMMARY
- **Trade Direction**: Clear LONG/SHORT/NEUTRAL recommendation
- **Confidence Level**: HIGH/MEDIUM/LOW with reasoning
- **Time Horizon**: Scalp/Swing/Position trade recommendation
- **Risk Assessment**: Quantified risk/reward ratio

### TECHNICAL CONFLUENCE ANALYSIS
- **Multi-timeframe Alignment**: How 1W/1D/4H/1H timeframes align
- **Volume Profile Significance**: POC and Value Area implications
- **Key Support/Resistance**: Most critical levels with confluence reasoning
- **Market Structure**: Current phase and expected evolution

### INSTITUTIONAL PERSPECTIVE
- **Volume-Based Strategy**: Where institutions are positioned
- **High-Probability Zones**: Entry/exit areas with volume confirmation
- **Smart Money Indicators**: What the volume profile reveals about positioning

### ACTIONABLE TRADING PLAN
- **Entry Strategy**: Specific price levels and confirmation signals
- **Position Sizing**: Based on volatility and risk parameters
- **Stop Loss**: Logical placement using technical and volume levels
- **Profit Targets**: Multiple targets with scaling strategy
- **Risk Management**: Specific protocols for this trade setup

### MARKET MONITORING
- **Key Levels to Watch**: Critical price points for trend confirmation/reversal
- **Volume Confirmation**: What volume patterns to monitor
- **Exit Triggers**: Specific conditions for trade management

Provide actionable, institutional-grade analysis with specific price levels and clear reasoning.
"""
        
        return enhanced_prompt
    
    async def _send_to_llm(self, prompt: str, symbol: str) -> Optional[Dict]:
        """Send prompt to configured LLM and return response"""
        
        if self.primary_model == 'claude' and self.claude_api_key:
            return await self._send_to_claude(prompt, symbol)
        elif self.primary_model == 'openai' and self.openai_api_key:
            return await self._send_to_openai(prompt, symbol)
        else:
            logger.error(f"No valid LLM configuration found for {self.primary_model}")
            return None
    
    async def _send_to_claude(self, prompt: str, symbol: str) -> Optional[Dict]:
        """Send request to Claude API"""
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.claude_api_key,
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': self.claude_model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    'https://api.anthropic.com/v1/messages',
                    headers=headers,
                    json=data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Claude API success for {symbol}")
                        
                        return {
                            'model': self.claude_model,
                            'content': result['content'][0]['text'],
                            'usage': result.get('usage', {}),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Claude API error {response.status}: {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"Claude API timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return None
    
    async def _send_to_openai(self, prompt: str, symbol: str) -> Optional[Dict]:
        """Send request to OpenAI API"""
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.openai_api_key}'
        }
        
        data = {
            'model': self.openai_model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an elite institutional trading professional with deep expertise in technical analysis, volume profile analysis, and multi-timeframe confluence trading.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ OpenAI API success for {symbol}")
                        
                        return {
                            'model': self.openai_model,
                            'content': result['choices'][0]['message']['content'],
                            'usage': result.get('usage', {}),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error {response.status}: {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"OpenAI API timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return None
    
    def _save_llm_response(self, symbol: str, llm_response: Dict, analysis_data: Dict) -> Path:
        """Save LLM response with metadata"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol.replace('/', '_')}_llm_analysis_{timestamp}.json"
        filepath = self.llm_responses_dir / filename
        
        # Combine response with metadata
        complete_response = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'llm_response': llm_response,
            'analysis_metadata': {
                'ultimate_score': analysis_data.get('ultimate_score', {}),
                'primary_bias': analysis_data.get('enhanced_trading_signals', {}).get('primary_bias', 'UNKNOWN'),
                'confidence': analysis_data.get('enhanced_trading_signals', {}).get('confidence', 'UNKNOWN'),
                'poc_price': analysis_data.get('volume_profile_analysis', {}).get('volume_profile', {}).get('poc', {}).get('price', 0),
                'current_price': analysis_data.get('volume_profile_analysis', {}).get('metadata', {}).get('current_price', 0)
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(complete_response, f, indent=2, default=str)
        
        logger.info(f"💾 LLM response saved: {filepath}")
        return filepath
    
    def _generate_rich_reports(self, symbol: str, llm_response: Dict, analysis_data: Dict) -> Dict[str, str]:
        """Generate rich HTML and Markdown reports"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{symbol.replace('/', '_')}_trading_report_{timestamp}"
        
        # Generate HTML report
        html_file = self._generate_html_report(symbol, llm_response, analysis_data, base_filename)
        
        # Generate Markdown report  
        md_file = self._generate_markdown_report(symbol, llm_response, analysis_data, base_filename)
        
        return {
            'html_report': str(html_file),
            'markdown_report': str(md_file)
        }
    
    def _convert_markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown text to HTML with proper formatting"""
        if not markdown_text:
            return "<p>No content available</p>"
        
        # Clean up the text first
        lines = markdown_text.split('\n')
        processed_lines = []
        in_list = False
        list_type = None  # Track 'ul' or 'ol'
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and separators
            if not line or line == '---':
                if in_list:
                    processed_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                elif processed_lines and not processed_lines[-1].endswith('>'):
                    processed_lines.append('<br>')
                continue
            
            # Headers
            if line.startswith('### '):
                if in_list:
                    processed_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                processed_lines.append(f'<h4>{line[4:]}</h4>')
            elif line.startswith('## '):
                if in_list:
                    processed_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                processed_lines.append(f'<h3>{line[3:]}</h3>')
            elif line.startswith('# '):
                if in_list:
                    processed_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                processed_lines.append(f'<h2>{line[2:]}</h2>')
            
            # List items
            elif line.startswith('- '):
                if not in_list or list_type != 'ul':
                    if in_list:  # Close previous list if different type
                        processed_lines.append(f'</{list_type}>')
                    processed_lines.append('<ul>')
                    in_list = True
                    list_type = 'ul'
                # Process bold text in list items
                item_content = line[2:]
                item_content = self._process_bold_text(item_content)
                processed_lines.append(f'<li>{item_content}</li>')
            
            # Numbered lists
            elif line[0].isdigit() and '. ' in line[:4]:
                if not in_list or list_type != 'ol':
                    if in_list:  # Close previous list if different type
                        processed_lines.append(f'</{list_type}>')
                    processed_lines.append('<ol>')
                    in_list = True
                    list_type = 'ol'
                item_content = line.split('. ', 1)[1]
                item_content = self._process_bold_text(item_content)
                processed_lines.append(f'<li>{item_content}</li>')
            
            # Regular paragraphs
            else:
                if in_list:
                    processed_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                # Process bold text
                line_content = self._process_bold_text(line)
                processed_lines.append(f'<p>{line_content}</p>')
        
        # Close any open list
        if in_list:
            processed_lines.append(f'</{list_type}>')
        
        return '\n'.join(processed_lines)
    
    def _process_bold_text(self, text: str) -> str:
        """Process bold text markers in a string"""
        import re
        # Replace **text** with <strong>text</strong>
        return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    def _generate_html_report(self, symbol: str, llm_response: Dict, analysis_data: Dict, base_filename: str) -> Path:
        """Generate comprehensive HTML trading report"""
        
        filepath = self.rich_reports_dir / f"{base_filename}.html"
        
        # Extract key data
        current_price = analysis_data.get('volume_profile_analysis', {}).get('metadata', {}).get('current_price', 0)
        ultimate_score = analysis_data.get('ultimate_score', {}).get('composite_score', 0)
        primary_bias = analysis_data.get('enhanced_trading_signals', {}).get('primary_bias', 'UNKNOWN')
        confidence = analysis_data.get('enhanced_trading_signals', {}).get('confidence', 'UNKNOWN')
        
        # POC and Value Area
        volume_profile = analysis_data.get('volume_profile_analysis', {}).get('volume_profile', {})
        poc_price = volume_profile.get('poc', {}).get('price', 0)
        value_area = volume_profile.get('value_area', {})
        va_high = value_area.get('high', 0)
        va_low = value_area.get('low', 0)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} Professional Trading Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9ff;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .content {{
            padding: 30px;
        }}
        .analysis-section {{
            margin-bottom: 30px;
            padding: 20px;
            border-left: 4px solid #667eea;
            background: #f8f9ff;
            border-radius: 0 8px 8px 0;
        }}
        .analysis-section h2 {{
            margin-top: 0;
            color: #333;
        }}
        .bias-bullish {{ color: #28a745; }}
        .bias-bearish {{ color: #dc3545; }}
        .bias-neutral {{ color: #ffc107; }}
        .confidence-high {{ color: #28a745; }}
        .confidence-medium {{ color: #ffc107; }}
        .confidence-low {{ color: #dc3545; }}
        .footer {{
            background: #333;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }}
        .llm-content {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .llm-content h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 25px;
        }}
        .llm-content h3 {{
            color: #34495e;
            margin-top: 20px;
        }}
        .llm-content h4 {{
            color: #7f8c8d;
            margin-top: 15px;
        }}
        .llm-content ul, .llm-content ol {{
            margin-left: 20px;
        }}
        .llm-content li {{
            margin-bottom: 5px;
        }}
        .llm-content strong {{
            color: #2c3e50;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{symbol} Professional Trading Analysis</h1>
            <div class="subtitle">
                AI-Enhanced Multi-Timeframe & Volume Profile Analysis<br>
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{format_price(current_price)}</div>
                <div class="metric-label">Current Price</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{ultimate_score:.1f}/100</div>
                <div class="metric-label">Ultimate Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value bias-{primary_bias.lower()}">{primary_bias}</div>
                <div class="metric-label">Primary Bias</div>
            </div>
            <div class="metric-card">
                <div class="metric-value confidence-{confidence.lower()}">{confidence}</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{format_price(poc_price)}</div>
                <div class="metric-label">POC Price</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{format_price(va_high)} - {format_price(va_low)}</div>
                <div class="metric-label">Value Area</div>
            </div>
        </div>
        
        <div class="content">
            <div class="analysis-section">
                <h2>🎯 Professional LLM Analysis</h2>
                <p><strong>Model:</strong> {llm_response.get('model', 'Unknown')}</p>
                <p><strong>Generated:</strong> {llm_response.get('timestamp', 'Unknown')}</p>
                <div class="llm-content">
                    {self._convert_markdown_to_html(llm_response.get('content', 'No analysis content available'))}
                </div>
            </div>
            
            <div class="analysis-section">
                <h2>📊 Technical Analysis Summary</h2>
                <ul>
                    <li><strong>Ultimate Score:</strong> {ultimate_score:.1f}/100</li>
                    <li><strong>Primary Bias:</strong> {primary_bias}</li>
                    <li><strong>Confidence Level:</strong> {confidence}</li>
                    <li><strong>Current Position vs Value Area:</strong> {"Above" if current_price > va_high else "Below" if current_price < va_low else "Inside"}</li>
                    <li><strong>Distance from POC:</strong> {format_distance_percent((current_price - poc_price) / poc_price * 100, 2)}</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Professional Trading Analysis | AI-Enhanced Technical & Volume Profile Analysis</p>
            <p>⚠️ This analysis is for informational purposes only. Trading involves risk of loss.</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📄 HTML report generated: {filepath}")
        return filepath
    
    def _generate_markdown_report(self, symbol: str, llm_response: Dict, analysis_data: Dict, base_filename: str) -> Path:
        """Generate comprehensive Markdown trading report"""
        
        filepath = self.rich_reports_dir / f"{base_filename}.md"
        
        # Extract key data
        current_price = analysis_data.get('volume_profile_analysis', {}).get('metadata', {}).get('current_price', 0)
        ultimate_score = analysis_data.get('ultimate_score', {}).get('composite_score', 0)
        primary_bias = analysis_data.get('enhanced_trading_signals', {}).get('primary_bias', 'UNKNOWN')
        confidence = analysis_data.get('enhanced_trading_signals', {}).get('confidence', 'UNKNOWN')
        
        # POC and Value Area
        volume_profile = analysis_data.get('volume_profile_analysis', {}).get('volume_profile', {})
        poc_price = volume_profile.get('poc', {}).get('price', 0)
        value_area = volume_profile.get('value_area', {})
        va_high = value_area.get('high', 0)
        va_low = value_area.get('low', 0)
        
        md_content = f"""# {symbol} Professional Trading Analysis

**AI-Enhanced Multi-Timeframe & Volume Profile Analysis**  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

---

## 📊 Key Metrics

| Metric | Value |
|--------|--------|
| **Current Price** | {format_price(current_price)} |
| **Ultimate Score** | {ultimate_score:.1f}/100 |
| **Primary Bias** | **{primary_bias}** |
| **Confidence** | **{confidence}** |
| **POC Price** | {format_price(poc_price)} |
| **Value Area** | {format_price(va_low)} - {format_price(va_high)} |

---

## 🎯 Professional LLM Analysis

**Model:** {llm_response.get('model', 'Unknown')}  
**Generated:** {llm_response.get('timestamp', 'Unknown')}

```
{llm_response.get('content', 'No analysis content available')}
```

---

## 📈 Technical Analysis Summary

- **Ultimate Score:** {ultimate_score:.1f}/100
- **Primary Bias:** {primary_bias}
- **Confidence Level:** {confidence}
- **Current Position vs Value Area:** {"Above" if current_price > va_high else "Below" if current_price < va_low else "Inside"}
- **Distance from POC:** {format_distance_percent((current_price - poc_price) / poc_price * 100, 2)}

---

## ⚠️ Risk Disclaimer

This analysis is for informational purposes only. Trading involves risk of loss. Always conduct your own research and consider your risk tolerance before making trading decisions.

---

*Professional Trading Analysis | AI-Enhanced Technical & Volume Profile Analysis*
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"📝 Markdown report generated: {filepath}")
        return filepath


async def main():
    """Test the LLM integration system"""
    
    print("🤖 LLM Integration System")
    print("=" * 50)
    
    # Initialize LLM integration
    llm = LLMIntegration()
    
    # Find the latest analysis file
    analysis_dir = Path("output/ultimate_analysis")
    analysis_files = list(analysis_dir.glob("ultimate_ETH_USDT_*.json"))
    
    if not analysis_files:
        print("❌ No analysis files found. Run the ultimate analyzer first.")
        return
    
    # Use the most recent analysis file
    latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Using analysis file: {latest_file.name}")
    
    # Load analysis data
    with open(latest_file, 'r') as f:
        analysis_data = json.load(f)
    
    symbol = analysis_data.get('symbol', 'ETH/USDT')
    
    # Process through LLM (if API keys are available)
    result = await llm.process_crypto_analysis(symbol, analysis_data)
    
    print(f"\\n🎉 LLM Integration Result:")
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"Response File: {result.get('response_file')}")
        print(f"HTML Report: {result.get('report_files', {}).get('html_report')}")
        print(f"Markdown Report: {result.get('report_files', {}).get('markdown_report')}")
    else:
        print(f"Error: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())