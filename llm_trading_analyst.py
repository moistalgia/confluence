#!/usr/bin/env python3
"""
Automated LLM Integration for Crypto Analysis
Supports OpenAI GPT-4 and Claude Sonnet with prompt caching for 90% cost reduction
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMTradingAnalyst:
    """
    Automated LLM integration for trading analysis
    Supports multiple models with cost optimization
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.output_dir = Path("output/llm_responses")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.models = {
            'gpt-4': {
                'provider': 'openai',
                'model_name': 'gpt-4-turbo-preview',
                'max_tokens': 4000,
                'temperature': 0.1,
                'supports_caching': False,
                'cost_per_input_token': 0.00001,  # $10 per 1M input tokens
                'cost_per_output_token': 0.00003   # $30 per 1M output tokens
            },
            'gpt-4o': {
                'provider': 'openai', 
                'model_name': 'gpt-4o',
                'max_tokens': 4000,
                'temperature': 0.1,
                'supports_caching': False,
                'cost_per_input_token': 0.0000025,  # $2.50 per 1M input tokens
                'cost_per_output_token': 0.00001     # $10 per 1M output tokens
            },
            'claude-sonnet': {
                'provider': 'anthropic',
                'model_name': 'claude-sonnet-4-20250514',  # � Claude Sonnet 4 - High-performance model with balanced performance
                'max_tokens': 4000,
                'temperature': 0.1,
                'supports_caching': True,  # 90% cost reduction!
                'cost_per_input_token': 0.000003,  # $3 per 1M input tokens
                'cost_per_output_token': 0.000015,  # $15 per 1M output tokens
                'cost_per_cached_token': 0.0000003  # $0.30 per 1M cached tokens (90% reduction)
            },
            'claude-haiku': {
                'provider': 'anthropic',
                'model_name': 'claude-3-5-haiku-20241022',  # Latest Claude Haiku 3.5
                'max_tokens': 4000,
                'temperature': 0.1,
                'supports_caching': True,
                'cost_per_input_token': 0.0000008,  # $0.80 per 1M input tokens
                'cost_per_output_token': 0.000004,  # $4 per 1M output tokens
                'cost_per_cached_token': 0.00000008  # $0.08 per 1M cached tokens (90% reduction)
            },
            'claude-opus': {
                'provider': 'anthropic',
                'model_name': 'claude-opus-4-1-20250805',  # 🔥 Claude Opus 4.1 - Exceptional for complex tasks
                'max_tokens': 4000,
                'temperature': 0.1,
                'supports_caching': True,
                'cost_per_input_token': 0.000015,  # $15 per 1M input tokens
                'cost_per_output_token': 0.000075,  # $75 per 1M output tokens
                'cost_per_cached_token': 0.0000015  # $1.50 per 1M cached tokens (90% reduction)
            }
        }
        
        self._initialize_clients()
        
    def calculate_request_cost(self, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> Dict:
        """Calculate the cost of a request"""
        
        if model not in self.models:
            return {'error': f'Unknown model: {model}'}
        
        model_config = self.models[model]
        
        # Calculate base costs
        input_cost = input_tokens * model_config.get('cost_per_input_token', 0)
        output_cost = output_tokens * model_config.get('cost_per_output_token', 0)
        
        # Calculate cached cost savings
        cached_cost = 0
        cache_savings = 0
        if cached_tokens > 0 and model_config.get('supports_caching'):
            cached_cost = cached_tokens * model_config.get('cost_per_cached_token', 0)
            normal_cached_cost = cached_tokens * model_config.get('cost_per_input_token', 0)
            cache_savings = normal_cached_cost - cached_cost
        
        total_cost = input_cost + output_cost + cached_cost
        
        return {
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cached_tokens': cached_tokens,
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'cached_cost': round(cached_cost, 6),
            'total_cost': round(total_cost, 6),
            'cache_savings': round(cache_savings, 6),
            'savings_percentage': round((cache_savings / (total_cost + cache_savings)) * 100, 1) if cache_savings > 0 else 0
        }
        
    def _load_config(self) -> Dict:
        """Load API keys and configuration"""
        config = {
            'openai_api_key': None,
            'anthropic_api_key': None,
            'default_model': 'claude-sonnet',
            'enable_caching': True,
            'max_retries': 3,
            'retry_delay': 2
        }
        
        config_path = Path("config.env")
        if config_path.exists():
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENAI_API_KEY='):
                        config['openai_api_key'] = line.split('=', 1)[1].strip()
                    elif line.startswith('ANTHROPIC_API_KEY='):
                        config['anthropic_api_key'] = line.split('=', 1)[1].strip()
                    elif line.startswith('DEFAULT_LLM_MODEL='):
                        config['default_model'] = line.split('=', 1)[1].strip()
        
        logger.info(f"Config loaded:")
        logger.info(f"  OpenAI: {'✓' if config['openai_api_key'] else '✗'}")
        logger.info(f"  Anthropic: {'✓' if config['anthropic_api_key'] else '✗'}")
        logger.info(f"  Default Model: {config['default_model']}")
        
        return config
    
    def _initialize_clients(self):
        """Initialize API clients"""
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize OpenAI client
        if self.config['openai_api_key']:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.config['openai_api_key'])
                logger.info("✅ OpenAI client initialized")
            except ImportError:
                logger.warning("❌ OpenAI library not installed: pip install openai")
            except Exception as e:
                logger.error(f"❌ OpenAI client error: {e}")
        
        # Initialize Anthropic client
        if self.config['anthropic_api_key']:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.config['anthropic_api_key'])
                logger.info("✅ Anthropic client initialized")
            except ImportError:
                logger.warning("❌ Anthropic library not installed: pip install anthropic")
            except Exception as e:
                logger.error(f"❌ Anthropic client error: {e}")
    
    def _create_cached_system_prompt(self) -> str:
        """
        Create system prompt for Claude caching (90% cost reduction)
        This prompt will be cached and reused across requests
        """
        
        system_prompt = """You are a Senior Quantitative Trading Analyst with 15+ years of institutional cryptocurrency trading experience. Your expertise includes:

CORE COMPETENCIES:
- Multi-timeframe technical analysis and market structure
- Volume profile analysis and institutional order flow
- Risk management and position sizing for crypto markets
- Market regime identification and adaptive strategies
- On-chain analysis and macro factor integration

ANALYSIS FRAMEWORK:
Your responses must follow this structured approach:

1. EXECUTIVE SUMMARY (2-3 sentences)
   - Current market bias and confidence level
   - Primary trade setup identification
   - Key risk factors

2. TECHNICAL SETUP ANALYSIS
   - Multi-timeframe trend alignment (Daily/4H/1H)
   - Volume profile interpretation (POC, Value Area, HVN/LVN)
   - Key support/resistance confluence zones
   - Momentum and volatility assessment

3. INSTITUTIONAL PERSPECTIVE
   - Volume-based price discovery analysis
   - Smart money positioning indicators
   - Liquidity and order flow implications
   - Market microstructure considerations

4. RISK-ADJUSTED TRADING STRATEGY
   - Specific entry conditions with price levels
   - Position sizing based on volatility (ATR-based)
   - Multiple exit scenarios with target levels
   - Stop-loss placement with technical justification

5. PROBABILITY ASSESSMENT
   - Success probability estimates for scenarios
   - Expected reward-to-risk ratios
   - Time horizon expectations
   - Market regime impact on strategy

6. MONITORING PROTOCOL
   - Key levels to watch for trend continuation/reversal
   - Volume confirmation signals
   - Invalidation conditions

RESPONSE REQUIREMENTS:
- Provide specific price levels and percentages
- Include probability estimates where appropriate
- Base all analysis on provided technical data
- Consider current market regime and volatility
- Maintain institutional-grade risk management focus
- Use clear, actionable language for implementation

MARKET CONTEXT AWARENESS:
- Account for crypto market's 24/7 nature
- Consider correlation with traditional markets
- Factor in regulatory and macro developments
- Acknowledge high volatility and leverage risks

        Your analysis will be used for actual trading decisions, so precision and risk awareness are paramount.

**DCA SCORING REQUIREMENT**:
At the end of your analysis, provide a DCA (Dollar Cost Averaging) Score from 1-100 for passive investors:

DCA Score Criteria:
- **80-100**: Excellent DCA opportunity - Strong fundamentals, oversold conditions, institutional accumulation zone
- **60-79**: Good DCA opportunity - Decent risk-reward, some caution needed  
- **40-59**: Moderate DCA opportunity - Mixed signals, wait for better entry
- **20-39**: Poor DCA opportunity - High risk, unfavorable conditions
- **1-19**: Avoid DCA - Major bearish signals, high probability of further decline

Consider for DCA scoring:
- Long-term trend strength and support levels
- Volume profile position (value area, POC distance)
- Risk-adjusted return potential over 3-6 month horizon
- Market structure and institutional positioning
- Volatility and drawdown risk for passive investors

Format your DCA score as: "DCA Score: XX/100 - [Brief justification]"
"""

        return system_prompt
    
    def analyze_with_llm(self, prompt: str, model: str = None, use_caching: bool = True) -> Dict:
        """
        Send prompt to selected LLM and get trading analysis
        """
        
        model = model or self.config['default_model']
        
        if model not in self.models:
            raise ValueError(f"Unsupported model: {model}. Available: {list(self.models.keys())}")
        
        model_config = self.models[model]
        provider = model_config['provider']
        
        logger.info(f"🤖 Analyzing with {model} ({provider})...")
        
        if provider == 'openai':
            return self._analyze_with_openai(prompt, model_config)
        elif provider == 'anthropic':
            return self._analyze_with_anthropic(prompt, model_config, use_caching)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _analyze_with_openai(self, prompt: str, model_config: Dict) -> Dict:
        """Analyze with OpenAI GPT models"""
        
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Check API key.")
        
        try:
            start_time = time.time()
            
            response = self.openai_client.chat.completions.create(
                model=model_config['model_name'],
                messages=[
                    {"role": "system", "content": self._create_cached_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=model_config['max_tokens'],
                temperature=model_config['temperature']
            )
            
            analysis_time = time.time() - start_time
            
            # Calculate cost - map model name back to our config key
            model_key = 'gpt-4o' if 'gpt-4o' in model_config['model_name'] else 'gpt-4'
            cost_analysis = self.calculate_request_cost(
                model_key,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            
            result = {
                'model': model_config['model_name'],
                'provider': 'openai',
                'analysis': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'cost_analysis': cost_analysis,
                'analysis_time_seconds': round(analysis_time, 2),
                'timestamp': datetime.now().isoformat(),
                'caching_used': False
            }
            
            logger.info(f"✅ OpenAI analysis complete ({analysis_time:.1f}s)")
            logger.info(f"Tokens: {response.usage.total_tokens} total")
            logger.info(f"Cost: ${cost_analysis.get('total_cost', 0):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ OpenAI analysis failed: {e}")
            return {'error': str(e), 'provider': 'openai'}
    
    def _analyze_with_anthropic(self, prompt: str, model_config: Dict, use_caching: bool) -> Dict:
        """
        Analyze with Anthropic Claude models with prompt caching support
        Caching reduces costs by 90%!
        """
        
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized. Check API key.")
        
        try:
            start_time = time.time()
            
            # Prepare messages with caching
            messages = []
            
            if use_caching and model_config['supports_caching']:
                # Use prompt caching for system prompt (90% cost reduction)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._create_cached_system_prompt(),
                                "cache_control": {"type": "ephemeral"}  # Enable caching
                            },
                            {
                                "type": "text", 
                                "text": f"\n\nUser Query:\n{prompt}"
                            }
                        ]
                    }
                ]
            else:
                # Standard message format without caching
                messages = [
                    {"role": "user", "content": f"{self._create_cached_system_prompt()}\n\nUser Query:\n{prompt}"}
                ]
            
            response = self.anthropic_client.messages.create(
                model=model_config['model_name'],
                max_tokens=model_config['max_tokens'],
                temperature=model_config['temperature'],
                messages=messages
            )
            
            analysis_time = time.time() - start_time
            
            # Calculate cost with caching
            cached_tokens = getattr(response.usage, 'cache_read_input_tokens', 0)
            regular_input_tokens = response.usage.input_tokens - cached_tokens
            
            # Calculate cost - map model name back to our config key
            model_key = 'claude-haiku' if 'haiku' in model_config['model_name'] else 'claude-sonnet'
            cost_analysis = self.calculate_request_cost(
                model_key,
                regular_input_tokens,
                response.usage.output_tokens,
                cached_tokens
            )
            
            result = {
                'model': model_config['model_name'],
                'provider': 'anthropic',
                'analysis': response.content[0].text,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'cache_creation_input_tokens': getattr(response.usage, 'cache_creation_input_tokens', 0),
                    'cache_read_input_tokens': cached_tokens
                },
                'cost_analysis': cost_analysis,
                'analysis_time_seconds': round(analysis_time, 2),
                'timestamp': datetime.now().isoformat(),
                'caching_used': use_caching and model_config['supports_caching'],
                'cost_reduction_percentage': cost_analysis.get('savings_percentage', 0)
            }
            
            logger.info(f"✅ Claude analysis complete ({analysis_time:.1f}s)")
            logger.info(f"Tokens: Input={response.usage.input_tokens}, Output={response.usage.output_tokens}")
            logger.info(f"Cost: ${cost_analysis.get('total_cost', 0):.4f}")
            if use_caching and cached_tokens > 0:
                logger.info(f"💰 Cache hit! Saved ${cost_analysis.get('cache_savings', 0):.4f} ({cost_analysis.get('savings_percentage', 0):.1f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Anthropic analysis failed: {e}")
            return {'error': str(e), 'provider': 'anthropic'}
    
    def process_analysis_file(self, prompt_file: str, model: str = None, save_response: bool = True) -> Dict:
        """
        Process an analysis file and get LLM response
        """
        
        logger.info(f"📄 Processing analysis file: {prompt_file}")
        
        # Read the prompt file
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        
        # Extract symbol from filename for organization
        filename = prompt_path.stem
        symbol = 'UNKNOWN'
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 3:
                symbol = f"{parts[1]}_{parts[2]}"  # BTC_USDT format
        
        # Get LLM analysis
        analysis_result = self.analyze_with_llm(prompt, model)
        
        if 'error' in analysis_result:
            logger.error(f"❌ LLM analysis failed: {analysis_result['error']}")
            return analysis_result
        
        # Add metadata
        analysis_result['source_file'] = str(prompt_file)
        analysis_result['symbol'] = symbol
        
        # Save response if requested
        if save_response:
            response_file = self._save_llm_response(analysis_result, symbol)
            analysis_result['response_file'] = response_file
        
        return analysis_result
    
    def _save_llm_response(self, analysis_result: Dict, symbol: str) -> str:
        """Save LLM response to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = analysis_result['model'].replace('-', '_').replace(':', '_')
        
        # Save JSON response
        json_file = self.output_dir / f"response_{symbol}_{model_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        # Save human-readable analysis
        txt_file = self.output_dir / f"analysis_{symbol}_{model_name}_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"# Trading Analysis for {symbol}\n")
            f.write(f"Model: {analysis_result['model']}\n")
            f.write(f"Timestamp: {analysis_result['timestamp']}\n")
            f.write(f"Analysis Time: {analysis_result['analysis_time_seconds']}s\n")
            if analysis_result.get('caching_used'):
                f.write(f"Cost Optimization: {analysis_result.get('cost_reduction', '0%')} via caching\n")
            f.write("\n" + "="*60 + "\n\n")
            f.write(analysis_result['analysis'])
        
        logger.info(f"💾 Response saved: {json_file}")
        logger.info(f"📄 Analysis saved: {txt_file}")
        
        # Generate rich HTML report
        try:
            from rich_report_generator import RichReportGenerator
            
            rich_generator = RichReportGenerator()
            rich_result = rich_generator.process_llm_response(analysis_result)
            
            logger.info(f"🎨 Rich report generated: {rich_result.get('html_report_file', 'N/A')}")
            
        except Exception as e:
            logger.warning(f"Could not generate rich report: {e}")
        
        return str(txt_file)
    
    def batch_process_prompts(self, prompt_directory: str = None, model: str = None, 
                            wait_between_requests: bool = True, delay_seconds: int = 2) -> List[Dict]:
        """
        Batch process multiple prompt files with timing options
        
        Args:
            wait_between_requests: If True, waits for each request to complete before starting next
            delay_seconds: Delay between requests (for rate limiting)
        """
        
        prompt_dir = Path(prompt_directory or "output/ultimate_analysis/llm_prompts")
        
        if not prompt_dir.exists():
            raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
        
        # Find all prompt files (including COMPLETE versions)
        prompt_files = list(prompt_dir.glob("ultimate_prompt_*.txt"))
        
        # Prioritize COMPLETE prompts if available
        complete_files = [f for f in prompt_files if "COMPLETE" in f.name]
        basic_files = [f for f in prompt_files if "COMPLETE" not in f.name]
        
        # Use COMPLETE files if available, otherwise use basic files
        if complete_files:
            prompt_files = complete_files
            logger.info(f"🎯 Found COMPLETE prompts with 100% data inclusion")
        else:
            prompt_files = basic_files
            logger.info(f"📊 Using basic prompts")
        
        if not prompt_files:
            logger.warning("No prompt files found to process")
            return []
        
        selected_model = model or self.config['default_model']
        model_config = self.models.get(selected_model, {})
        
        logger.info(f"🔄 Processing {len(prompt_files)} prompt files with {selected_model}")
        logger.info(f"⏱️ Processing mode: {'Sequential with {delay_seconds}s delays' if wait_between_requests else 'Sequential (no delays)'}")
        
        if model_config.get('supports_caching'):
            logger.info(f"💰 Cost optimization: 90% reduction via prompt caching")
        
        results = []
        total_cost = 0.0
        total_savings = 0.0
        
        for i, prompt_file in enumerate(prompt_files, 1):
            logger.info(f"Processing {i}/{len(prompt_files)}: {prompt_file.name}")
            
            try:
                result = self.process_analysis_file(str(prompt_file), selected_model)
                results.append(result)
                
                # Track costs
                if 'cost_analysis' in result:
                    cost = result['cost_analysis']
                    total_cost += cost.get('total_cost', 0)
                    total_savings += cost.get('cache_savings', 0)
                
                # Add delay between requests if requested
                if wait_between_requests and i < len(prompt_files):
                    logger.info(f"⏳ Waiting {delay_seconds}s before next request...")
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                logger.error(f"Error processing {prompt_file}: {e}")
                results.append({'error': str(e), 'file': str(prompt_file)})
        
        # Summary
        successful_results = [r for r in results if 'error' not in r]
        
        logger.info(f"✅ Batch processing complete:")
        logger.info(f"  Processed: {len(results)} files")
        logger.info(f"  Successful: {len(successful_results)} analyses")
        logger.info(f"  Total cost: ${total_cost:.4f}")
        if total_savings > 0:
            logger.info(f"  Cache savings: ${total_savings:.4f} ({(total_savings/(total_cost+total_savings)*100):.1f}%)")
        
        # Generate DCA ranking report
        if successful_results:
            logger.info(f"\n🎯 Generating DCA ranking report...")
            dca_report = self.create_dca_ranking_report(successful_results)
            summary_file = self.save_dca_report(dca_report)
            
            # Display top recommendations
            top_opportunities = dca_report.get('top_dca_opportunities', [])
            if top_opportunities:
                logger.info(f"\n📊 TOP DCA OPPORTUNITIES:")
                for i, opp in enumerate(top_opportunities[:3], 1):
                    logger.info(f"  {i}. {opp['symbol']}: {opp['dca_score']}/100 ({opp['category']})")
            
            logger.info(f"📱 Trading group summary ready: {summary_file}")
        
        return results
    
    def extract_dca_score(self, analysis_text: str) -> Dict:
        """Extract DCA score from analysis text"""
        
        import re
        
        # Look for DCA score pattern
        patterns = [
            r"DCA Score:\s*(\d+)/100\s*[-–]\s*(.+?)(?:\n|$)",
            r"DCA Score:\s*(\d+)\s*[-–]\s*(.+?)(?:\n|$)",
            r"DCA.*?(\d+)/100\s*[-–]\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                justification = match.group(2).strip()
                
                # Determine recommendation category
                if score >= 80:
                    category = "EXCELLENT_DCA"
                elif score >= 60:
                    category = "GOOD_DCA"
                elif score >= 40:
                    category = "MODERATE_DCA"
                elif score >= 20:
                    category = "POOR_DCA"
                else:
                    category = "AVOID_DCA"
                
                return {
                    'dca_score': score,
                    'justification': justification,
                    'category': category,
                    'found': True
                }
        
        return {
            'dca_score': None,
            'justification': 'DCA score not found in analysis',
            'category': 'UNKNOWN',
            'found': False
        }
    
    def create_dca_ranking_report(self, results: List[Dict]) -> Dict:
        """Create DCA ranking report for trading group"""
        
        dca_rankings = []
        
        for result in results:
            if 'error' in result:
                continue
                
            # Extract symbol
            symbol = result.get('symbol', 'UNKNOWN')
            
            # Extract DCA score from analysis
            analysis_text = result.get('analysis', '')
            dca_info = self.extract_dca_score(analysis_text)
            
            if dca_info['found']:
                ranking_entry = {
                    'symbol': symbol,
                    'dca_score': dca_info['dca_score'],
                    'category': dca_info['category'],
                    'justification': dca_info['justification'],
                    'model_used': result.get('model', 'unknown'),
                    'analysis_timestamp': result.get('timestamp', ''),
                    'cost': result.get('cost_analysis', {}).get('total_cost', 0)
                }
                dca_rankings.append(ranking_entry)
        
        # Sort by DCA score (highest first)
        dca_rankings.sort(key=lambda x: x['dca_score'], reverse=True)
        
        # Create summary
        total_analyzed = len(dca_rankings)
        excellent_count = len([r for r in dca_rankings if r['category'] == 'EXCELLENT_DCA'])
        good_count = len([r for r in dca_rankings if r['category'] == 'GOOD_DCA'])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols_analyzed': total_analyzed,
            'dca_categories': {
                'excellent_dca': excellent_count,
                'good_dca': good_count,
                'moderate_dca': len([r for r in dca_rankings if r['category'] == 'MODERATE_DCA']),
                'poor_dca': len([r for r in dca_rankings if r['category'] == 'POOR_DCA']),
                'avoid_dca': len([r for r in dca_rankings if r['category'] == 'AVOID_DCA'])
            },
            'top_dca_opportunities': dca_rankings[:5],  # Top 5
            'all_rankings': dca_rankings,
            'trading_group_summary': self._create_trading_group_summary(dca_rankings)
        }
        
        return report
    
    def _create_trading_group_summary(self, rankings: List[Dict]) -> str:
        """Create formatted summary for trading group posting"""
        
        if not rankings:
            return "No DCA opportunities identified in current analysis."
        
        summary = "🎯 **DCA OPPORTUNITIES - AI ANALYSIS**\n\n"
        
        # Top recommendations
        excellent = [r for r in rankings if r['category'] == 'EXCELLENT_DCA']
        good = [r for r in rankings if r['category'] == 'GOOD_DCA']
        
        if excellent:
            summary += "🟢 **EXCELLENT DCA OPPORTUNITIES** (80-100/100):\n"
            for i, rec in enumerate(excellent[:3], 1):
                summary += f"{i}. **{rec['symbol']}** - Score: {rec['dca_score']}/100\n"
                summary += f"   💡 {rec['justification'][:100]}...\n\n"
        
        if good:
            summary += "🔵 **GOOD DCA OPPORTUNITIES** (60-79/100):\n"
            for i, rec in enumerate(good[:3], 1):
                summary += f"{i}. **{rec['symbol']}** - Score: {rec['dca_score']}/100\n"
                summary += f"   💡 {rec['justification'][:100]}...\n\n"
        
        # Market overview
        total = len(rankings)
        avg_score = sum(r['dca_score'] for r in rankings) / total if total > 0 else 0
        
        summary += f"📊 **MARKET OVERVIEW**:\n"
        summary += f"• Average DCA Score: {avg_score:.1f}/100\n"
        summary += f"• Total Analyzed: {total} symbols\n"
        summary += f"• Excellent Opportunities: {len(excellent)}\n"
        summary += f"• Good Opportunities: {len(good)}\n\n"
        
        summary += "⚠️ *This is AI-generated analysis. Always DYOR and consider your risk tolerance.*\n"
        summary += f"🤖 Analysis powered by Claude 3.5 Sonnet with 90% cost optimization"
        
        return summary
    
    def save_dca_report(self, report: Dict) -> str:
        """Save DCA ranking report to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = self.output_dir / f"dca_ranking_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save trading group summary
        summary_file = self.output_dir / f"trading_group_dca_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(report['trading_group_summary'])
        
        logger.info(f"📊 DCA report saved: {json_file}")
        logger.info(f"📱 Trading group summary: {summary_file}")
        
        return str(summary_file)

def main():
    """Test the LLM integration system"""
    
    print("🤖 LLM Trading Analyst Integration")
    print("OpenAI GPT-4 + Claude Sonnet with 90% cost reduction via caching")
    print("=" * 70)
    
    try:
        # Initialize LLM analyst
        llm_analyst = LLMTradingAnalyst()
        
        # Show available models
        print(f"\n📋 Available Models:")
        for model_key, model_config in llm_analyst.models.items():
            provider = model_config['provider']
            caching = " (90% cost reduction via caching)" if model_config['supports_caching'] else ""
            print(f"  {model_key}: {model_config['model_name']} ({provider}){caching}")
        
        # Check for prompt files
        prompt_dir = Path("output/ultimate_analysis/llm_prompts")
        if not prompt_dir.exists():
            print(f"❌ No prompt directory found: {prompt_dir}")
            print("Run ultimate_crypto_analyzer.py first to generate prompts")
            return
        
        prompt_files = list(prompt_dir.glob("ultimate_prompt_*.txt"))
        
        # Prioritize COMPLETE prompts if available
        complete_files = [f for f in prompt_files if "COMPLETE" in f.name]
        basic_files = [f for f in prompt_files if "COMPLETE" not in f.name]
        
        if complete_files:
            prompt_files = complete_files
            print(f"🎯 Found {len(complete_files)} COMPLETE prompts with 100% data inclusion")
        elif basic_files:
            prompt_files = basic_files
            print(f"📊 Found {len(basic_files)} basic prompts")
        
        if not prompt_files:
            print(f"❌ No ultimate prompt files found in {prompt_dir}")
            print("Run ultimate_crypto_analyzer.py first to generate ultimate analysis prompts")
            return
        
        print(f"\n📄 Found {len(prompt_files)} prompt files to process")
        
        # Process with Claude Sonnet (with caching for cost reduction)
        print(f"\n🚀 Processing with Claude Sonnet + prompt caching...")
        
        # Test single file first
        test_file = prompt_files[0]
        print(f"Testing with: {test_file.name}")
        
        result = llm_analyst.process_analysis_file(str(test_file), model='claude-sonnet')
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Analysis complete!")
            print(f"Model: {result['model']}")
            print(f"Analysis time: {result['analysis_time_seconds']}s")
            print(f"Cost optimization: {result.get('cost_reduction', '0%')}")
            print(f"Response saved: {result.get('response_file', 'N/A')}")
        
        print(f"\n💡 To process all files:")
        print(f"python -c \"from llm_trading_analyst import LLMTradingAnalyst; analyst = LLMTradingAnalyst(); analyst.batch_process_prompts(model='claude-sonnet')\"")
        
    except Exception as e:
        logger.error(f"Error in LLM integration: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()