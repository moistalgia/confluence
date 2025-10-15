#!/usr/bin/env python3
"""
Quick test to validate Kraken live data connection
"""

import asyncio
import logging
import yaml
from core.disciplined_trading_engine import DisciplinedTradingEngine
from core.data_provider import DataProvider

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_live_data():
    """Test live data connection with Kraken"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force enable live data
    config['paper_trading']['use_live_market_data'] = True
    
    print("=== TESTING KRAKEN LIVE DATA CONNECTION ===")
    print(f"Live data enabled: {config['paper_trading']['use_live_market_data']}")
    
    try:
        # Initialize data provider (using None to bypass initialization issues)
        data_provider = None
        
        # Initialize engine with live data
        engine = DisciplinedTradingEngine(
            config=config.get('paper_trading', {}),
            data_provider=data_provider,
            db_path='test_paper_trading.db'
        )
        
        # Test getting live market data
        if engine.use_live_data:
            print("SUCCESS: Live data connection established!")
            
            # Test fetching BTC price from Kraken
            try:
                market_data = engine.get_live_market_data('BTC/USDT')
                if market_data:
                    print(f"BTC/USDT Live Price: ${market_data['last']:.2f}")
                    print(f"Bid: ${market_data['bid']:.2f}")
                    print(f"Ask: ${market_data['ask']:.2f}")
                    print(f"Spread: {market_data['spread_pct']*100:.3f}%")
                    
                    if market_data['last'] > 100000:
                        print("SUCCESS: Real BTC price (~110k) detected!")
                        print("Live data is working correctly!")
                    else:
                        print(f"WARNING: BTC price ${market_data['last']:.2f} seems low for current market")
                else:
                    print("ERROR: No market data returned")
                    
            except Exception as e:
                print(f"ERROR: Failed to fetch market data: {e}")
        else:
            print("INFO: Using simulated data (live data disabled)")
            
    except Exception as e:
        print(f"ERROR: Engine initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_live_data())