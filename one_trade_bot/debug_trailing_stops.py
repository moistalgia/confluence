#!/usr/bin/env python3
"""
Debug the trailing stop logic specifically
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Set working directory and path
project_dir = Path(r"c:\Dev\crypto-analyzer\one_trade_bot")
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

from core.intelligent_exit_manager import IntelligentExitManager, get_default_exit_config

def debug_trailing_stops():
    """Debug trailing stop logic step by step"""
    print("🔍 DEBUGGING TRAILING STOP LOGIC")
    print("=" * 50)
    
    # Create exit manager
    config = get_default_exit_config()
    exit_manager = IntelligentExitManager(config)
    
    # Test position
    test_position = {
        'symbol': 'ADA/USDT',
        'entry_price': 0.4500,
        'stop_loss': 0.4400,    # -2.2% stop
        'take_profit': 0.4800,  # +6.7% target
        'quantity': 100.0,
        'entry_time': datetime.now() - timedelta(hours=2)
    }
    
    print(f"🎯 Position: {test_position['symbol']}")
    print(f"   Entry: ${test_position['entry_price']:.4f}")
    print(f"   Original Stop: ${test_position['stop_loss']:.4f}")
    print(f"   Target: ${test_position['take_profit']:.4f}")
    
    # Test prices in sequence to show trailing evolution
    test_prices = [
        0.4545,  # +1.0% profit (should trigger breakeven)
        0.4590,  # +2.0% profit (should trigger 50% trail)
        0.4635,  # +3.0% profit (should trigger 70% trail)
        0.4620,  # Small pullback (trailing should protect)
    ]
    
    print(f"\n📊 Step-by-step trailing evolution:")
    print("-" * 50)
    
    for i, price in enumerate(test_prices, 1):
        profit_pct = (price - test_position['entry_price']) / test_position['entry_price']
        
        print(f"\n{i}. Price: ${price:.4f} ({profit_pct*100:+.1f}% profit)")
        
        # Check trailing stop manually
        current_stop = exit_manager.trailing_stops.get(test_position['symbol'], test_position['stop_loss'])
        
        trailing_result = exit_manager._check_trailing_stop(
            test_position['entry_price'], 
            price, 
            current_stop, 
            profit_pct
        )
        
        print(f"   Current stop before: ${current_stop:.4f}")
        print(f"   Trailing check: {trailing_result}")
        
        if trailing_result['updated']:
            exit_manager.trailing_stops[test_position['symbol']] = trailing_result['new_stop']
            print(f"   ✅ Stop updated to: ${trailing_result['new_stop']:.4f}")
        else:
            print(f"   ⏸️  No update: {trailing_result['reason']}")
        
        # Now check full exit conditions
        exit_signal = exit_manager.check_exit_conditions(test_position, price)
        
        if exit_signal.should_exit:
            print(f"   🚨 EXIT: {exit_signal.exit_type} - {exit_signal.reason}")
            break
        else:
            print(f"   📈 HOLD: Position continues")

if __name__ == "__main__":
    debug_trailing_stops()