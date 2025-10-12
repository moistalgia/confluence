#!/usr/bin/env python3
"""
Rich Report Regenerator - Fix LLM Analysis Formatting
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from llm_integration import LLMIntegration

async def regenerate_reports_for_symbol(symbol_pattern="BTC_USDT"):
    """Regenerate rich reports for the latest analysis"""
    
    print(f"🔄 Regenerating reports for {symbol_pattern}")
    
    # Find latest LLM response file
    llm_responses_dir = Path("output/ultimate_analysis/llm_responses")
    response_files = list(llm_responses_dir.glob(f"{symbol_pattern}_llm_analysis_*.json"))
    
    if not response_files:
        print(f"❌ No LLM response files found for {symbol_pattern}")
        return
    
    latest_response_file = max(response_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Using: {latest_response_file.name}")
    
    # Load LLM response data
    with open(latest_response_file, 'r') as f:
        response_data = json.load(f)
    
    # Find corresponding ultimate analysis file
    timestamp_part = latest_response_file.stem.split('_')[-2:]  # e.g., ['20251011', '222944']
    timestamp_str = "_".join(timestamp_part)
    
    ultimate_files = list(Path("output/ultimate_analysis").glob(f"ultimate_{symbol_pattern}_{timestamp_str}.json"))
    
    if not ultimate_files:
        print(f"❌ No ultimate analysis file found for timestamp {timestamp_str}")
        return
    
    # Load analysis data
    with open(ultimate_files[0], 'r') as f:
        analysis_data = json.load(f)
    
    # Initialize LLM integration for report generation
    llm = LLMIntegration()
    
    # Extract data
    symbol = response_data['symbol']
    llm_response = response_data['llm_response']
    
    # Generate new reports with corrected formatting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{symbol.replace('/', '_')}_trading_report_CORRECTED_{timestamp}"
    
    # Generate reports
    html_file = llm._generate_html_report(symbol, llm_response, analysis_data, base_filename)
    md_file = llm._generate_markdown_report(symbol, llm_response, analysis_data, base_filename)
    
    print(f"✅ Corrected HTML Report: {html_file}")
    print(f"✅ Corrected Markdown Report: {md_file}")
    
    return {
        'html_report': str(html_file),
        'markdown_report': str(md_file)
    }

async def main():
    """Regenerate reports for both symbols"""
    
    print("🔧 RICH REPORT FORMATTER - LLM Analysis Fix")
    print("=" * 60)
    
    symbols = ["BTC_USDT", "ETH_USDT"]
    
    for symbol in symbols:
        try:
            await regenerate_reports_for_symbol(symbol)
            print()
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
    
    print("🎉 Report regeneration complete!")

if __name__ == "__main__":
    asyncio.run(main())