#!/usr/bin/env python3
"""
Test the Intelligent Exit Manager
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Set working directory and path
project_dir = Path(r"c:\Dev\crypto-analyzer\one_trade_bot")
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))

import yaml
from core.intelligent_exit_manager import IntelligentExitManager, get_default_exit_config

def test_intelligent_exits():
    """Test the intelligent exit system with various scenarios"""
    print("🧪 Testing Intelligent Exit Manager")
    print("=" * 50)
    
    # Load config with intelligent exits
    config = get_default_exit_config()
    print(f"📋 Configuration loaded:")
    print(f"   Trailing stops: {'✅' if config['intelligent_exits']['trailing_stop']['enabled'] else '❌'}")
    print(f"   Momentum detection: {'✅' if config['intelligent_exits']['momentum_reversal']['enabled'] else '❌'}")
    print(f"   Resistance analysis: {'✅' if config['intelligent_exits']['resistance_rejection']['enabled'] else '❌'}")
    print(f"   Time decay: {'✅' if config['intelligent_exits']['time_decay']['enabled'] else '❌'}")
    
    # Create exit manager
    exit_manager = IntelligentExitManager(config)
    
    # Test position (similar to ADA/USDT scenario)
    test_position = {
        'symbol': 'ADA/USDT',
        'entry_price': 0.4500,
        'stop_loss': 0.4400,    # -2.2% stop
        'take_profit': 0.4800,  # +6.7% target
        'quantity': 100.0,
        'entry_time': datetime.now() - timedelta(hours=2)  # 2 hours in trade
    }
    
    print(f"\n🎯 Test Position: {test_position['symbol']}")
    print(f"   Entry: ${test_position['entry_price']:.4f}")
    print(f"   Stop: ${test_position['stop_loss']:.4f}")
    print(f"   Target: ${test_position['take_profit']:.4f}")
    print(f"   Time in trade: 2 hours")
    
    # Test scenarios
    test_scenarios = [
        {"price": 0.4450, "desc": "Small loss (-1.1%)"},
        {"price": 0.4545, "desc": "Small profit (+1.0% - should trigger breakeven)"},
        {"price": 0.4590, "desc": "Good profit (+2.0% - should trigger 50% trail)"},
        {"price": 0.4635, "desc": "Strong profit (+3.0% - should trigger 70% trail)"},
        {"price": 0.4800, "desc": "Target hit (+6.7%)"},
        {"price": 0.4390, "desc": "Stop loss hit (-2.4%)"}
    ]
    
    print(f"\n📊 Testing Exit Scenarios:")
    print("-" * 50)
    
    for i, scenario in enumerate(test_scenarios, 1):
        current_price = scenario['price']
        description = scenario['desc']
        
        print(f"\n{i}. {description}")
        print(f"   Current Price: ${current_price:.4f}")
        
        # Check exit conditions
        exit_signal = exit_manager.check_exit_conditions(test_position, current_price)
        
        if exit_signal.should_exit:
            print(f"   🚨 EXIT SIGNAL: {exit_signal.exit_type}")
            print(f"   Reason: {exit_signal.reason}")
            print(f"   Exit Price: ${exit_signal.exit_price:.4f}")
            print(f"   Confidence: {exit_signal.confidence}")
        else:
            print(f"   ✅ HOLD - {exit_signal.reason}")
        
        # Show position status
        status = exit_manager.get_position_status(test_position, current_price)
        print(f"   P&L: ${status.profit_usd:+.2f} ({status.profit_pct*100:+.1f}%)")
        print(f"   Progress: {status.progress_to_target*100:.0f}% to target")
    
    print(f"\n✅ Intelligent Exit Manager test completed")

if __name__ == "__main__":
    test_intelligent_exits()