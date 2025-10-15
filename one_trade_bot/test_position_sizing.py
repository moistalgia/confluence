#!/usr/bin/env python3
"""
Test position sizing calculation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_position_sizing():
    print("🧮 POSITION SIZING TEST - FIXED APPROACH")
    print("=" * 50)
    
    # Test parameters
    account_balance = 10000
    position_pct = 0.01  # 1% of account as position size
    entry_price = 94.92
    stop_loss = 93.02
    
    # NEW APPROACH: Fixed position size
    target_position_value = account_balance * position_pct
    position_size = target_position_value / entry_price
    position_value = position_size * entry_price
    
    # Calculate actual risk
    price_risk = abs(entry_price - stop_loss)
    risk_amount = position_size * price_risk
    
    print(f"Account Balance: ${account_balance:,.2f}")
    print(f"Position Percentage: {position_pct*100:.1f}%")
    print(f"Target Position Value: ${target_position_value:.2f}")
    print(f"Entry Price: ${entry_price:.4f}")
    print(f"Stop Loss: ${stop_loss:.4f}")
    print(f"Price Risk: ${price_risk:.4f} per share")
    print(f"Position Size: {position_size:.6f} shares")
    print(f"Actual Position Value: ${position_value:.2f}")
    print(f"Actual Risk Amount: ${risk_amount:.2f}")
    print()
    
    if abs(position_value - target_position_value) < 1:  # Should be ~$100
        print("✅ Position size looks correct!")
        print(f"   Target: ${target_position_value:.2f}, Actual: ${position_value:.2f}")
    else:
        print("❌ POSITION SIZE MISMATCH!")
        print(f"   Expected: ${target_position_value:.2f}, got ${position_value:.2f}")
    
    # Show what the calculation should be
    print("\n📝 NEW CALCULATION APPROACH:")
    print(f"Target Position: ${account_balance} × {position_pct} = ${target_position_value}")
    print(f"Position Size: ${target_position_value} ÷ ${entry_price} = {position_size:.6f} shares")
    print(f"Position Value: {position_size:.6f} × ${entry_price} = ${position_value:.2f}")
    print(f"Risk Amount: {position_size:.6f} × ${price_risk} = ${risk_amount:.2f}")

if __name__ == "__main__":
    test_position_sizing()