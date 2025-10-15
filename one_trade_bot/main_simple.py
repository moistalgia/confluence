#!/usr/bin/env python3
"""
One Trade Bot - Simple Main Runner
🎯 Clean, simple runner for the complete DisciplinedTradingEngine

Usage:
    python main_simple.py                  # Run daily scan and entry monitoring
    python main_simple.py --test           # Run test suite
    python main_simple.py --live-data      # Force enable live market data
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime
import yaml

# Add the bot directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.disciplined_trading_engine import DisciplinedTradingEngine
from core.data_provider import DataProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config from {config_path}: {str(e)}")
        sys.exit(1)

async def run_disciplined_trading(config, enable_live_data=None):
    """Run the complete disciplined trading system"""
    
    # Override live data setting if requested
    if enable_live_data is not None:
        config['use_live_market_data'] = enable_live_data
        config['paper_trading']['use_live_market_data'] = enable_live_data
    
    try:
        # Initialize data provider (fallback for non-live data)
        data_provider = DataProvider({})
        
        # Initialize the complete disciplined trading engine
        logger.info("🎯 INITIALIZING DISCIPLINED TRADING ENGINE")
        logger.info("=" * 60)
        
        engine = DisciplinedTradingEngine(
            config=config.get('paper_trading', {}),
            data_provider=data_provider,
            db_path='paper_trading.db'
        )
        
        logger.info("🚀 STARTING DISCIPLINED TRADING CYCLE")
        logger.info("   This system maintains our core discipline:")
        logger.info("   - 8:00 AM daily scan picks THE ONE target")
        logger.info("   - Patient entry zone monitoring")
        logger.info("   - ONE trade maximum per day")
        logger.info("   - Professional execution & database tracking")
        logger.info("")
        
        # Run the disciplined cycle
        await engine.run_disciplined_cycle()
        
    except KeyboardInterrupt:
        logger.info("🛑 Trading system stopped by user")
    except Exception as e:
        logger.error(f"🚨 Trading system error: {str(e)}")
        raise

async def run_test_suite():
    """Run the test suite to validate the system"""
    logger.info("🧪 RUNNING DISCIPLINED ENGINE TEST SUITE")
    logger.info("=" * 60)
    
    try:
        # Import and run the test
        import subprocess
        result = subprocess.run([sys.executable, 'test_disciplined_engine.py'], 
                              capture_output=True, text=True, cwd='.')
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            logger.info("✅ All tests passed! System ready for live trading")
        else:
            logger.error("❌ Tests failed! Please fix issues before live trading")
            
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"🚨 Test execution failed: {str(e)}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='One Trade Bot - Disciplined Trading System')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    parser.add_argument('--live-data', action='store_true', help='Force enable live market data')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.test:
        # Run test suite
        success = asyncio.run(run_test_suite())
        sys.exit(0 if success else 1)
    else:
        # Run live trading system
        logger.info(f"🎯 ONE TRADE BOT - Starting {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Live Data: {'✅ ENABLED' if config.get('paper_trading', {}).get('use_live_market_data', False) or args.live_data else '❌ SIMULATED'}")
        logger.info(f"   Account: ${config.get('trading', {}).get('account_balance', 10000):,}")
        logger.info(f"   Max Risk: {config.get('trading', {}).get('max_risk_per_trade', 0.01)*100:.1f}% per trade")
        
        asyncio.run(run_disciplined_trading(config, args.live_data))

if __name__ == "__main__":
    main()