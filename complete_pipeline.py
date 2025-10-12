#!/usr/bin/env python3
"""
Complete Crypto Analysis Pipeline
Analysis → Prompt → LLM → Rich Reports

Full integration of:
- Ultimate Crypto Analyzer (data collection & analysis)
- LLM Integration (Claude Sonnet 4 / OpenAI processing)
- Rich Report Generation (HTML/Markdown outputs)
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import our components
from ultimate_crypto_analyzer import UltimateCryptoAnalyzer
from llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoAnalysisPipeline:
    """
    Complete end-to-end crypto analysis pipeline
    """
    
    def __init__(self, symbols: List[str] = None):
        """Initialize the complete pipeline"""
        
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.setup_components()
        self.setup_output_tracking()
        
    def setup_components(self):
        """Initialize all pipeline components"""
        
        logger.info("🚀 Initializing Complete Crypto Analysis Pipeline")
        
        # Initialize Ultimate Crypto Analyzer
        self.analyzer = UltimateCryptoAnalyzer()
        logger.info("✅ Ultimate Crypto Analyzer initialized")
        
        # Initialize LLM Integration
        self.llm_integration = LLMIntegration()
        logger.info("✅ LLM Integration initialized")
        
    def setup_output_tracking(self):
        """Setup output directory tracking"""
        
        self.base_output_dir = Path("output/ultimate_analysis")
        self.pipeline_log_file = self.base_output_dir / "pipeline_log.json"
        
        # Load existing pipeline log
        if self.pipeline_log_file.exists():
            with open(self.pipeline_log_file, 'r') as f:
                self.pipeline_log = json.load(f)
        else:
            self.pipeline_log = {
                'pipeline_runs': [],
                'last_updated': None
            }
    
    async def run_complete_analysis(self, force_refresh: bool = False) -> Dict:
        """
        Run the complete analysis pipeline for all symbols
        
        Args:
            force_refresh: If True, regenerate all analysis even if recent data exists
            
        Returns:
            Complete pipeline results with file paths and status
        """
        
        pipeline_start_time = datetime.now()
        pipeline_id = pipeline_start_time.strftime('%Y%m%d_%H%M%S')
        
        logger.info("=" * 80)
        logger.info(f"🎯 STARTING COMPLETE CRYPTO ANALYSIS PIPELINE")
        logger.info(f"Pipeline ID: {pipeline_id}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Force Refresh: {force_refresh}")
        logger.info("=" * 80)
        
        pipeline_results = {
            'pipeline_id': pipeline_id,
            'start_time': pipeline_start_time.isoformat(),
            'symbols_processed': [],
            'total_symbols': len(self.symbols),
            'success_count': 0,
            'error_count': 0,
            'results': {}
        }
        
        # Process each symbol
        for symbol in self.symbols:
            logger.info(f"\\n🔄 Processing {symbol}...")
            
            try:
                # Step 1: Run Ultimate Analysis
                symbol_result = await self._process_symbol(symbol, force_refresh)
                
                pipeline_results['results'][symbol] = symbol_result
                pipeline_results['symbols_processed'].append(symbol)
                
                if symbol_result['status'] == 'success':
                    pipeline_results['success_count'] += 1
                    logger.info(f"✅ {symbol} pipeline complete")
                else:
                    pipeline_results['error_count'] += 1
                    logger.error(f"❌ {symbol} pipeline failed: {symbol_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"💥 Critical error processing {symbol}: {str(e)}")
                pipeline_results['error_count'] += 1
                pipeline_results['results'][symbol] = {
                    'status': 'critical_error',
                    'error': str(e),
                    'symbol': symbol
                }
        
        # Finalize pipeline results
        pipeline_results['end_time'] = datetime.now().isoformat()
        pipeline_results['duration_seconds'] = (datetime.now() - pipeline_start_time).total_seconds()
        
        # Save pipeline log
        self._save_pipeline_log(pipeline_results)
        
        # Print final summary
        self._print_pipeline_summary(pipeline_results)
        
        return pipeline_results
    
    async def _process_symbol(self, symbol: str, force_refresh: bool) -> Dict:
        """Process a single symbol through the complete pipeline"""
        
        symbol_start_time = datetime.now()
        
        try:
            # Step 1: Ultimate Crypto Analysis
            logger.info(f"📊 Step 1: Running ultimate analysis for {symbol}...")
            
            analysis_result = self.analyzer.run_ultimate_analysis(symbol)
            
            if not analysis_result or 'error' in analysis_result:
                return {
                    'status': 'analysis_error',
                    'error': 'Ultimate analysis failed',
                    'symbol': symbol,
                    'step_failed': 'ultimate_analysis'
                }
            
            # Find the generated analysis file
            analysis_files = list(self.base_output_dir.glob(f"ultimate_{symbol.replace('/', '_')}_*.json"))
            if not analysis_files:
                return {
                    'status': 'file_error',
                    'error': 'Analysis file not found',
                    'symbol': symbol,
                    'step_failed': 'file_location'
                }
            
            # Use the most recent analysis file
            latest_analysis_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            
            # Load analysis data
            with open(latest_analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            logger.info(f"✅ Ultimate analysis complete for {symbol}")
            
            # Step 2: LLM Processing
            logger.info(f"🤖 Step 2: Processing through LLM for {symbol}...")
            
            llm_result = await self.llm_integration.process_crypto_analysis(symbol, analysis_data)
            
            if llm_result.get('status') != 'success':
                # Even if LLM fails, we still have the analysis
                logger.warning(f"⚠️ LLM processing failed for {symbol}, but analysis is available")
                return {
                    'status': 'partial_success',
                    'symbol': symbol,
                    'analysis_file': str(latest_analysis_file),
                    'llm_error': llm_result.get('error', 'Unknown LLM error'),
                    'step_completed': 'ultimate_analysis',
                    'step_failed': 'llm_processing'
                }
            
            logger.info(f"✅ LLM processing complete for {symbol}")
            
            # Step 3: Success - All steps completed
            duration = (datetime.now() - symbol_start_time).total_seconds()
            
            return {
                'status': 'success',
                'symbol': symbol,
                'analysis_file': str(latest_analysis_file),
                'llm_response_file': llm_result.get('response_file'),
                'html_report': llm_result.get('report_files', {}).get('html_report'),
                'markdown_report': llm_result.get('report_files', {}).get('markdown_report'),
                'llm_model': llm_result.get('llm_model'),
                'duration_seconds': duration,
                'steps_completed': ['ultimate_analysis', 'llm_processing', 'report_generation']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol,
                'duration_seconds': (datetime.now() - symbol_start_time).total_seconds()
            }
    
    def _save_pipeline_log(self, pipeline_results: Dict):
        """Save pipeline execution log"""
        
        self.pipeline_log['pipeline_runs'].append(pipeline_results)
        self.pipeline_log['last_updated'] = datetime.now().isoformat()
        
        # Keep only the last 10 pipeline runs
        if len(self.pipeline_log['pipeline_runs']) > 10:
            self.pipeline_log['pipeline_runs'] = self.pipeline_log['pipeline_runs'][-10:]
        
        with open(self.pipeline_log_file, 'w') as f:
            json.dump(self.pipeline_log, f, indent=2, default=str)
        
        logger.info(f"💾 Pipeline log saved: {self.pipeline_log_file}")
    
    def _print_pipeline_summary(self, results: Dict):
        """Print a comprehensive pipeline summary"""
        
        logger.info("\\n" + "=" * 80)
        logger.info("🎉 CRYPTO ANALYSIS PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Pipeline ID: {results['pipeline_id']}")
        logger.info(f"Total Duration: {results['duration_seconds']:.1f} seconds")
        logger.info(f"Symbols Processed: {results['total_symbols']}")
        logger.info(f"✅ Successful: {results['success_count']}")
        logger.info(f"❌ Failed: {results['error_count']}")
        
        if results['success_count'] > 0:
            logger.info("\\n📁 Generated Files:")
            for symbol, result in results['results'].items():
                if result.get('status') in ['success', 'partial_success']:
                    logger.info(f"\\n🎯 {symbol}:")
                    if result.get('analysis_file'):
                        logger.info(f"   📊 Analysis: {result['analysis_file']}")
                    if result.get('html_report'):
                        logger.info(f"   📄 HTML Report: {result['html_report']}")
                    if result.get('markdown_report'):
                        logger.info(f"   📝 Markdown Report: {result['markdown_report']}")
                    if result.get('llm_response_file'):
                        logger.info(f"   🤖 LLM Response: {result['llm_response_file']}")
        
        if results['error_count'] > 0:
            logger.info("\\n❌ Errors:")
            for symbol, result in results['results'].items():
                if result.get('status') not in ['success', 'partial_success']:
                    logger.info(f"   {symbol}: {result.get('error', 'Unknown error')}")
        
        logger.info("\\n💡 Next Steps:")
        logger.info("   1. Review generated HTML reports in output/ultimate_analysis/rich_reports/")
        logger.info("   2. Copy LLM analysis for trading decisions")
        logger.info("   3. Monitor key levels identified in the reports")
        logger.info("   4. Set up alerts for confluence zones")
        
        logger.info("=" * 80)


async def main():
    """Run the complete crypto analysis pipeline"""
    
    # Configure symbols to analyze
    symbols = [
        'BTC/USDT',
        'ETH/USDT'
    ]
    
    # Initialize pipeline
    pipeline = CryptoAnalysisPipeline(symbols=symbols)
    
    # Run complete analysis
    results = await pipeline.run_complete_analysis(force_refresh=True)
    
    # Return results for programmatic use
    return results


if __name__ == "__main__":
    # Run the pipeline
    results = asyncio.run(main())
    
    print("\\n🏁 Pipeline execution complete!")
    print(f"Check the output/ultimate_analysis/ directory for all generated files.")