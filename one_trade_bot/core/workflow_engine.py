"""
Daily Workflow Engine
🎯 Orchestrates the complete 5-filter system for ONE GOOD TRADE PER DAY

Daily Process:
1. Load watchlist and configuration
2. Run Filter 1: Market Regime (eliminate bad days)  
3. Run Filter 2: Setup Scanner (find patterns)
4. Run Filter 3: Confluence Check (multi-timeframe)
5. Run Filter 4: Risk Validation (final safety)
6. Run Filter 5: Entry Execution (disciplined entry)

Result: Either THE ONE TRADE or "sit on hands" decision
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import yaml

from core.data_provider import DataProvider
from filters.regime_filter import filter_watchlist_by_regime
from filters.setup_scanner import scan_watchlist_for_setups
from filters.confluence_checker import check_setup_confluence
from filters.risk_check import perform_final_risk_validation
from filters.entry_execution import execute_entry_workflow

logger = logging.getLogger(__name__)

class DailyWorkflowEngine:
    """
    Orchestrates the complete 5-filter pipeline for daily trade selection
    Implements "One Good Trade Per Day" philosophy with strict quality control
    """
    
    def __init__(self, config: Dict, data_provider: DataProvider):
        """
        Initialize the workflow engine
        
        Args:
            config: Complete bot configuration
            data_provider: Data source for market data
        """
        self.config = config
        self.data_provider = data_provider
        
        # Extract configuration sections
        self.watchlist = config.get('watchlist', [])
        self.account_config = config.get('account', {})
        self.filter_configs = {
            'regime': config.get('regime_filter', {}),
            'setup': config.get('setup_scanner', {}), 
            'confluence': config.get('confluence_checker', {}),
            'risk': config.get('risk_check', {}),
            'entry': config.get('entry_execution', {})
        }
        
        # Workflow state
        self.workflow_results = {}
        self.daily_decision = None
        self.execution_plan = None
        
        logger.info("🚀 Daily Workflow Engine initialized")
        logger.info(f"   Watchlist: {len(self.watchlist)} symbols")
        logger.info(f"   Account balance: ${self.account_config.get('balance', 0):,.0f}")
        
    def run_daily_workflow(self, open_positions: List[Dict] = None) -> Dict:
        """
        Execute complete daily workflow to find THE ONE TRADE
        
        Args:
            open_positions: List of currently open positions
            
        Returns:
            Dict with complete workflow results and daily decision
        """
        if open_positions is None:
            open_positions = []
            
        workflow_start_time = datetime.utcnow()
        
        logger.info("🚀 STARTING DAILY WORKFLOW")
        logger.info("=" * 80)
        logger.info(f"   Date: {workflow_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info(f"   Watchlist: {len(self.watchlist)} symbols")
        logger.info(f"   Open positions: {len(open_positions)}")
        logger.info("=" * 80)
        
        try:
            # Check if we already have a position (one trade at a time rule)
            if len(open_positions) > 0:
                logger.info("⚠️ POSITION ALREADY OPEN - Skipping workflow (one trade at a time)")
                return self._create_workflow_result(
                    'POSITION_ALREADY_OPEN',
                    "Daily workflow skipped - already have open position",
                    workflow_start_time
                )
            
            # FILTER 1: Market Regime Analysis
            logger.info("\n🔍 FILTER 1: MARKET REGIME ANALYSIS")
            regime_results = filter_watchlist_by_regime(
                self.watchlist, self.data_provider, self.filter_configs['regime']
            )
            self.workflow_results['regime'] = regime_results
            
            tradeable_symbols = regime_results['tradeable_symbols']
            if len(tradeable_symbols) == 0:
                logger.warning("❌ WORKFLOW STOPPED: No symbols passed regime filter")
                return self._create_workflow_result(
                    'NO_TRADEABLE_REGIME',
                    "Market regime unfavorable - sitting on hands today",
                    workflow_start_time
                )
            
            # FILTER 2: Setup Scanner  
            logger.info("\n🎯 FILTER 2: SETUP SCANNER")
            setup_results = scan_watchlist_for_setups(
                tradeable_symbols, self.data_provider, self.filter_configs['setup']
            )
            self.workflow_results['setup'] = setup_results
            
            setup_symbols = setup_results['setup_symbols']
            if len(setup_symbols) == 0:
                logger.warning("❌ WORKFLOW STOPPED: No valid setups found")
                return self._create_workflow_result(
                    'NO_SETUPS_FOUND',
                    "No pullback setups identified - waiting for better patterns",
                    workflow_start_time
                )
            
            # FILTER 3: Confluence Check
            logger.info("\n🔄 FILTER 3: CONFLUENCE CHECKER")
            confluence_results = check_setup_confluence(
                setup_symbols, self.data_provider, 
                setup_results['results'], self.filter_configs['confluence']
            )
            self.workflow_results['confluence'] = confluence_results
            
            approved_symbols = confluence_results['approved_symbols']
            if len(approved_symbols) == 0:
                logger.warning("❌ WORKFLOW STOPPED: No multi-timeframe confluence")
                return self._create_workflow_result(
                    'NO_CONFLUENCE',
                    "Multi-timeframe analysis rejected all setups",
                    workflow_start_time
                )
            
            # FILTER 4: Final Risk Check
            logger.info("\n🛡️ FILTER 4: FINAL RISK CHECK")
            risk_results = perform_final_risk_validation(
                approved_symbols, confluence_results['results'],
                self.account_config['balance'], open_positions,
                self.data_provider, self.filter_configs['risk']
            )
            self.workflow_results['risk'] = risk_results
            
            final_approved = risk_results['final_approved_symbols']
            the_one_trade = risk_results['the_one_trade']
            
            if not the_one_trade:
                logger.warning("❌ WORKFLOW STOPPED: No symbols passed final risk check")
                return self._create_workflow_result(
                    'RISK_REJECTED',
                    "All symbols failed final risk validation",
                    workflow_start_time
                )
            
            # FILTER 5: Entry Execution Planning
            logger.info("\n⚡ FILTER 5: ENTRY EXECUTION")
            
            # Determine trade direction from confluence results
            trade_direction = self._determine_trade_direction(the_one_trade, confluence_results)
            
            entry_results = execute_entry_workflow(
                the_one_trade, trade_direction,
                self.account_config['balance'], 
                self.data_provider, self.filter_configs['entry']
            )
            self.workflow_results['entry'] = entry_results
            
            # Final workflow result
            if entry_results['workflow_status'] == 'SUCCESS':
                logger.info("✅ WORKFLOW COMPLETE: THE ONE TRADE identified and ready")
                return self._create_workflow_result(
                    'TRADE_READY',
                    f"THE ONE TRADE: {the_one_trade} {trade_direction}",
                    workflow_start_time,
                    the_one_trade=the_one_trade,
                    trade_direction=trade_direction,
                    entry_plan=entry_results.get('entry_plan'),
                    next_action=entry_results.get('next_action')
                )
            else:
                logger.error("❌ WORKFLOW FAILED: Entry execution planning failed")
                return self._create_workflow_result(
                    'EXECUTION_FAILED', 
                    f"Entry planning failed for {the_one_trade}: {entry_results.get('error', 'Unknown error')}",
                    workflow_start_time
                )
                
        except Exception as e:
            logger.error(f"🚨 WORKFLOW ERROR: {str(e)}")
            return self._create_workflow_result(
                'WORKFLOW_ERROR',
                f"Workflow failed with error: {str(e)}",
                workflow_start_time,
                error=str(e)
            )
    
    def _determine_trade_direction(self, symbol: str, confluence_results: Dict) -> str:
        """
        Determine trade direction from confluence results
        
        Returns:
            'LONG' or 'SHORT'
        """
        confluence_data = confluence_results['results'].get(symbol, {})
        return confluence_data.get('trade_direction', 'UNKNOWN')
    
    def _create_workflow_result(self, status: str, message: str, start_time: datetime,
                               **kwargs) -> Dict:
        """
        Create standardized workflow result
        
        Returns:
            Dict with workflow result
        """
        end_time = datetime.utcnow()
        duration_seconds = (end_time - start_time).total_seconds()
        
        result = {
            'workflow_status': status,
            'message': message,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(), 
            'duration_seconds': duration_seconds,
            'filter_results': self.workflow_results.copy(),
            'daily_decision': status,
            **kwargs
        }
        
        # Log final summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 DAILY WORKFLOW SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Status: {status}")
        logger.info(f"Decision: {message}")
        logger.info(f"Duration: {duration_seconds:.1f} seconds")
        
        # Log filter funnel
        if self.workflow_results:
            logger.info("\n📈 FILTER FUNNEL:")
            
            if 'regime' in self.workflow_results:
                regime = self.workflow_results['regime']['summary']
                logger.info(f"   Watchlist → Regime Filter: {regime['total_checked']} → {regime['tradeable_count']}")
                
            if 'setup' in self.workflow_results:  
                setup = self.workflow_results['setup']['summary']
                logger.info(f"   Regime → Setup Scanner: {setup['total_scanned']} → {setup['setups_count']}")
                
            if 'confluence' in self.workflow_results:
                confluence = self.workflow_results['confluence']['summary'] 
                logger.info(f"   Setup → Confluence: {confluence['total_checked']} → {confluence['approved_count']}")
                
            if 'risk' in self.workflow_results:
                risk = self.workflow_results['risk']['summary']
                logger.info(f"   Confluence → Risk Check: {risk['total_checked']} → {risk['final_approved_count']}")
        
        # Final recommendation
        if status == 'TRADE_READY':
            logger.info(f"\n🎯 RECOMMENDATION: Execute THE ONE TRADE")
            logger.info(f"   Symbol: {kwargs.get('the_one_trade', 'N/A')}")
            logger.info(f"   Direction: {kwargs.get('trade_direction', 'N/A')}")
            logger.info(f"   Next Action: {kwargs.get('next_action', 'N/A')}")
        else:
            logger.info(f"\n💺 RECOMMENDATION: Sit on hands today")
            logger.info(f"   Reason: {message}")
        
        logger.info("=" * 80)
        
        return result
    
    def get_workflow_summary(self) -> Dict:
        """
        Get summary of current workflow state
        
        Returns:
            Dict with workflow summary
        """
        return {
            'engine_status': 'READY',
            'watchlist_size': len(self.watchlist),
            'account_balance': self.account_config.get('balance', 0),
            'last_workflow_results': self.workflow_results,
            'daily_decision': self.daily_decision,
            'execution_plan': self.execution_plan
        }


def create_workflow_engine(config_path: str = 'config.yaml') -> DailyWorkflowEngine:
    """
    Factory function to create configured workflow engine
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured DailyWorkflowEngine instance
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data provider
    data_provider = DataProvider(config.get('data_provider', {}))
    
    # Create and return workflow engine
    return DailyWorkflowEngine(config, data_provider)


def run_daily_scan(config_path: str = 'config.yaml', open_positions: List[Dict] = None) -> Dict:
    """
    Convenience function to run complete daily scan
    
    Args:
        config_path: Path to configuration file
        open_positions: List of open positions
        
    Returns:
        Complete workflow results
    """
    logger.info("🚀 STARTING DAILY SCAN")
    
    # Create engine and run workflow
    engine = create_workflow_engine(config_path)
    results = engine.run_daily_workflow(open_positions or [])
    
    return results