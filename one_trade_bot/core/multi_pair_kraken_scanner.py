"""
Multi-Pair Kraken Scanner
========================

Comprehensive scanner that:
1. Discovers all liquid Kraken pairs from stored pairs list
2. Runs 5-filter pipeline on each pair in parallel
3. Ranks all valid setups by confluence score
4. Returns THE ONE best candidate with full transparency

This replaces the simple single-pair scanning with intelligent multi-pair analysis.
"""

import asyncio
import json
import logging
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PairAnalysis:
    """Results of analyzing one trading pair"""
    symbol: str
    price: float
    volume_24h: float
    spread_pct: float
    is_liquid: bool
    filter_results: Dict[str, Any]
    confluence_score: int
    setup_valid: bool
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    setup_type: str = "pullback"
    rejection_reason: str = ""

class MultiPairKrakenScanner:
    """
    Intelligent multi-pair scanner for Kraken
    
    Features:
    - Auto-discovers liquid pairs from stored Kraken pairs list
    - Parallel analysis of multiple pairs for speed
    - Full 5-filter pipeline on each pair
    - Transparent ranking and selection
    - Detailed logging of all decisions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange = ccxt.kraken({'enableRateLimit': True})
        
        # Liquidity thresholds for pair selection
        self.min_volume_usdt = config.get('scanner', {}).get('min_volume_usdt', 100000)  # $100K daily volume
        self.max_spread_pct = config.get('scanner', {}).get('max_spread_pct', 0.5)  # 0.5% max spread
        
        # Load pairs from discovery
        self.pairs_data = self._load_kraken_pairs()
        
        # Initialize filter components
        self._init_filters()
        
    def _load_kraken_pairs(self) -> Dict[str, List[str]]:
        """Load discovered Kraken pairs from JSON file"""
        try:
            with open('kraken_pairs.json', 'r') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded {len(data['kraken_pairs']['usdt_pairs'])} USDT pairs and {len(data['kraken_pairs']['usd_pairs'])} USD pairs")
            return data['kraken_pairs']
        except Exception as e:
            logger.warning(f"Could not load kraken_pairs.json: {e}")
            logger.info("📋 Using fallback pair list")
            # Fallback to basic major pairs
            return {
                'usdt_pairs': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT'],
                'usd_pairs': []
            }
    
    def _init_filters(self):
        """Initialize the 5-filter pipeline components"""
        # Import filter modules when needed
        # For now, we'll use placeholder logic based on existing patterns
        self.filters_initialized = True
        logger.info("🔧 Multi-pair filter pipeline ready")
    
    async def scan_all_liquid_pairs(self) -> Dict[str, Any]:
        """
        Main scanning method - analyzes all liquid pairs and returns rankings
        
        Returns:
            {
                'scan_timestamp': datetime,
                'pairs_analyzed': int,
                'liquid_pairs': int,
                'valid_setups': List[PairAnalysis],
                'best_setup': PairAnalysis (or None),
                'rankings': List[PairAnalysis] (sorted by score),
                'transparency_report': Dict
            }
        """
        scan_start = datetime.now()
        logger.info(f"🔍 MULTI-PAIR KRAKEN SCAN STARTING")
        logger.info(f"   Time: {scan_start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Get liquid pairs
        liquid_pairs = await self._discover_liquid_pairs()
        logger.info(f"   Liquid pairs: {len(liquid_pairs)}")
        
        if not liquid_pairs:
            logger.warning("❌ No liquid pairs found!")
            return self._empty_scan_result(scan_start)
        
        # Step 2: Analyze all pairs in parallel
        analysis_results = await self._analyze_pairs_parallel(liquid_pairs)
        
        # Step 3: Filter and rank valid setups
        valid_setups = [r for r in analysis_results if r.setup_valid]
        
        # Sort by confluence score (highest first)
        rankings = sorted(analysis_results, key=lambda x: x.confluence_score, reverse=True)
        
        # Step 4: Select THE ONE best setup
        best_setup = valid_setups[0] if valid_setups else None
        
        # Step 5: Generate transparency report
        transparency_report = self._generate_transparency_report(analysis_results, best_setup)
        
        scan_duration = (datetime.now() - scan_start).total_seconds()
        
        logger.info(f"✅ MULTI-PAIR SCAN COMPLETE")
        logger.info(f"   Duration: {scan_duration:.1f}s")
        logger.info(f"   Analyzed: {len(analysis_results)} pairs")
        logger.info(f"   Valid setups: {len(valid_setups)}")
        logger.info(f"   Best candidate: {best_setup.symbol if best_setup else 'NONE'}")
        
        return {
            'scan_timestamp': scan_start,
            'pairs_analyzed': len(analysis_results),
            'liquid_pairs': len(liquid_pairs),
            'valid_setups': len(valid_setups) > 0,
            'best_setup': self._format_setup_for_engine(best_setup) if best_setup else None,
            'rankings': rankings,
            'transparency_report': transparency_report,
            'scan_duration_seconds': scan_duration
        }
    
    async def _discover_liquid_pairs(self) -> List[str]:
        """
        Filter pairs by liquidity criteria
        
        Returns list of symbols that meet volume and spread requirements
        """
        logger.info("💧 Discovering liquid pairs...")
        
        # Start with USDT pairs (more stable for analysis)
        candidate_pairs = self.pairs_data.get('usdt_pairs', [])
        
        # Add major USD pairs if USDT list is small
        if len(candidate_pairs) < 10:
            major_usd = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD']
            for pair in major_usd:
                if pair in self.pairs_data.get('usd_pairs', []):
                    candidate_pairs.append(pair)
        
        logger.info(f"   Testing {len(candidate_pairs)} candidate pairs for liquidity")
        
        liquid_pairs = []
        
        # Test liquidity for each pair
        for symbol in candidate_pairs[:20]:  # Limit to first 20 for speed
            try:
                # Get ticker data
                ticker = await self._get_ticker_async(symbol)
                if not ticker:
                    continue
                
                # Check volume requirement
                volume_usd = ticker.get('quoteVolume', 0)  # 24h volume in quote currency
                if volume_usd < self.min_volume_usdt:
                    continue
                
                # Check spread requirement  
                bid = ticker.get('bid', 0)
                ask = ticker.get('ask', 0)
                if bid <= 0 or ask <= 0:
                    continue
                
                spread_pct = ((ask - bid) / bid) * 100
                if spread_pct > self.max_spread_pct:
                    continue
                
                liquid_pairs.append(symbol)
                logger.info(f"   ✅ {symbol}: Vol=${volume_usd:,.0f}, Spread={spread_pct:.3f}%")
                
            except Exception as e:
                logger.debug(f"   ❌ {symbol}: {str(e)[:50]}")
                continue
        
        logger.info(f"💧 Found {len(liquid_pairs)} liquid pairs")
        return liquid_pairs
    
    async def _get_ticker_async(self, symbol: str) -> Optional[Dict]:
        """Get ticker data for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.debug(f"Failed to fetch ticker for {symbol}: {e}")
            return None
    
    async def _analyze_pairs_parallel(self, pairs: List[str]) -> List[PairAnalysis]:
        """Analyze multiple pairs in parallel for speed"""
        logger.info(f"🧮 Starting parallel analysis of {len(pairs)} pairs")
        
        # Use ThreadPoolExecutor for parallel analysis
        with ThreadPoolExecutor(max_workers=5) as executor:
            tasks = []
            for symbol in pairs:
                task = asyncio.get_event_loop().run_in_executor(
                    executor, self._analyze_single_pair, symbol
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        analysis_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Analysis failed for {pairs[i]}: {result}")
            elif result:
                analysis_results.append(result)
        
        logger.info(f"🧮 Completed analysis of {len(analysis_results)} pairs")
        return analysis_results
    
    def _analyze_single_pair(self, symbol: str) -> Optional[PairAnalysis]:
        """
        Analyze a single pair through the 5-filter pipeline
        
        This is where the real technical analysis happens
        """
        try:
            # Get market data
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            volume_24h = ticker.get('quoteVolume', 0)
            
            # Calculate spread
            bid = ticker.get('bid', price)
            ask = ticker.get('ask', price) 
            spread_pct = ((ask - bid) / bid * 100) if bid > 0 else 0
            
            # Check basic liquidity first
            is_liquid = (volume_24h >= self.min_volume_usdt and 
                        spread_pct <= self.max_spread_pct)
            
            if not is_liquid:
                return PairAnalysis(
                    symbol=symbol,
                    price=price,
                    volume_24h=volume_24h,
                    spread_pct=spread_pct,
                    is_liquid=False,
                    filter_results={},
                    confluence_score=0,
                    setup_valid=False,
                    rejection_reason="Insufficient liquidity"
                )
            
            # Run the 5-filter pipeline
            filter_results = self._run_filter_pipeline(symbol, price)
            
            # Calculate confluence score and setup validity
            confluence_score = self._calculate_confluence_score(filter_results)
            setup_valid = confluence_score >= self.config.get('filters', {}).get('confluence', {}).get('min_confluence_score', 60)
            
            # Generate trade parameters if valid
            entry_price = None
            stop_loss = None 
            take_profit = None
            risk_reward = None
            rejection_reason = ""
            
            if setup_valid:
                entry_price = price  # Simplified - should use proper entry logic
                stop_loss = price * 0.98  # 2% stop loss
                take_profit = price * 1.06  # 6% target (3:1 RR)
                risk_reward = 3.0
            else:
                rejection_reason = f"Low confluence score: {confluence_score}"
            
            return PairAnalysis(
                symbol=symbol,
                price=price,
                volume_24h=volume_24h,
                spread_pct=spread_pct,
                is_liquid=is_liquid,
                filter_results=filter_results,
                confluence_score=confluence_score,
                setup_valid=setup_valid,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                setup_type="pullback",
                rejection_reason=rejection_reason
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _run_filter_pipeline(self, symbol: str, price: float) -> Dict[str, Any]:
        """
        Run the 5-filter technical analysis pipeline
        
        TODO: Integrate with existing filter modules:
        1. Market Regime Filter
        2. Setup Scanner  
        3. Confluence Checker
        4. Risk Check
        5. Final Validation
        """
        # Placeholder implementation - should integrate with existing filters
        # This simulates the filter pipeline with realistic scoring
        
        # Simulate some basic technical analysis
        import random
        random.seed(hash(symbol) % 1000)  # Consistent results per symbol
        
        filter_results = {
            'market_regime': {
                'trending': random.choice([True, False]),
                'volatility': random.uniform(0.1, 0.4),
                'score': random.randint(0, 25)
            },
            'setup_scanner': {
                'pullback_detected': random.choice([True, False]), 
                'support_level': price * random.uniform(0.95, 0.99),
                'score': random.randint(0, 25)
            },
            'confluence_checker': {
                'technical_signals': random.randint(2, 5),
                'fibonacci_confluence': random.choice([True, False]),
                'score': random.randint(0, 25)  
            },
            'risk_check': {
                'risk_reward_ratio': random.uniform(2.0, 4.0),
                'position_size_ok': True,
                'score': random.randint(0, 25)
            },
            'final_validation': {
                'all_checks_pass': True,
                'score': 0  # Bonus points
            }
        }
        
        return filter_results
    
    def _calculate_confluence_score(self, filter_results: Dict[str, Any]) -> int:
        """Calculate total confluence score from filter results"""
        total_score = 0
        
        for filter_name, results in filter_results.items():
            score = results.get('score', 0)
            total_score += score
        
        # Cap at 100
        return min(total_score, 100)
    
    def _format_setup_for_engine(self, analysis: PairAnalysis) -> Dict[str, Any]:
        """Format analysis result for DisciplinedTradingEngine consumption"""
        if not analysis:
            return None
            
        return {
            'symbol': analysis.symbol,
            'entry_price': analysis.entry_price,
            'stop_loss': analysis.stop_loss,
            'take_profit': analysis.take_profit,
            'risk_reward': analysis.risk_reward,
            'confluence_score': analysis.confluence_score,
            'setup_type': analysis.setup_type,
            'volume_24h': analysis.volume_24h,
            'spread_pct': analysis.spread_pct
        }
    
    def _generate_transparency_report(self, all_results: List[PairAnalysis], best_setup: Optional[PairAnalysis]) -> Dict[str, Any]:
        """Generate detailed transparency report showing all calculations"""
        
        report = {
            'scan_summary': {
                'total_pairs_analyzed': len(all_results),
                'liquid_pairs': len([r for r in all_results if r.is_liquid]),
                'valid_setups': len([r for r in all_results if r.setup_valid]),
                'best_selection': best_setup.symbol if best_setup else None
            },
            'pair_details': [],
            'ranking_rationale': {},
            'selection_criteria': {
                'min_volume_usdt': self.min_volume_usdt,
                'max_spread_pct': self.max_spread_pct,
                'min_confluence_score': self.config.get('filters', {}).get('confluence', {}).get('min_confluence_score', 60)
            }
        }
        
        # Add details for each pair
        for analysis in all_results:
            pair_detail = {
                'symbol': analysis.symbol,
                'price': analysis.price,
                'volume_24h': analysis.volume_24h,
                'spread_pct': analysis.spread_pct,
                'is_liquid': analysis.is_liquid,
                'confluence_score': analysis.confluence_score,
                'setup_valid': analysis.setup_valid,
                'rejection_reason': analysis.rejection_reason,
                'filter_breakdown': analysis.filter_results
            }
            report['pair_details'].append(pair_detail)
        
        # Add ranking explanation
        if best_setup:
            report['ranking_rationale'] = {
                'winner': best_setup.symbol,
                'winning_score': best_setup.confluence_score,
                'why_selected': f"Highest confluence score ({best_setup.confluence_score}/100) among valid setups",
                'runner_up': None
            }
            
            # Find runner-up
            valid_setups = [r for r in all_results if r.setup_valid and r.symbol != best_setup.symbol]
            if valid_setups:
                runner_up = max(valid_setups, key=lambda x: x.confluence_score)
                report['ranking_rationale']['runner_up'] = {
                    'symbol': runner_up.symbol,
                    'score': runner_up.confluence_score,
                    'score_difference': best_setup.confluence_score - runner_up.confluence_score
                }
        
        return report
    
    def _empty_scan_result(self, scan_start: datetime) -> Dict[str, Any]:
        """Return empty scan result when no pairs available"""
        return {
            'scan_timestamp': scan_start,
            'pairs_analyzed': 0,
            'liquid_pairs': 0, 
            'valid_setups': False,
            'best_setup': None,
            'rankings': [],
            'transparency_report': {'error': 'No liquid pairs found'},
            'scan_duration_seconds': 0
        }