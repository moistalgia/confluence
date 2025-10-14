#!/usr/bin/env python3
"""
Singleton Trading System Manager
===============================

Eliminates the duplicate trading engine bug that was allocating $1000 
to a second engine while the main engine only had $9000.

Ensures ONLY ONE trading engine exists across the entire application.

Key fixes:
- Singleton pattern prevents multiple engine instances
- Centralized configuration management
- Single source of truth for all trading operations
- Eliminates cash allocation conflicts

Author: Professional Trading Team  
Date: October 13, 2025
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import threading

# Import our professional components
from professional_portfolio_tracker import ProfessionalPortfolioTracker

logger = logging.getLogger(__name__)

class TradingSystemConfig:
    """Centralized configuration for the trading system"""
    
    def __init__(self, 
                 starting_capital: float = 10000.0,
                 max_risk_per_trade: float = 0.015,  # 1.5%
                 max_portfolio_risk: float = 0.60,   # 60%
                 max_concurrent_positions: int = 8,
                 enable_shorting: bool = True,
                 enable_compounding: bool = True):
        
        self.starting_capital = starting_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_concurrent_positions = max_concurrent_positions
        self.enable_shorting = enable_shorting
        self.enable_compounding = enable_compounding
        
        # Validation
        if starting_capital <= 0:
            raise ValueError("Starting capital must be positive")
        if not (0 < max_risk_per_trade < 1):
            raise ValueError("Max risk per trade must be between 0 and 1")
        if not (0 < max_portfolio_risk <= 1):
            raise ValueError("Max portfolio risk must be between 0 and 1")
            
        logger.info(f"📊 Trading config: ${starting_capital:,.0f} capital, "
                   f"{max_risk_per_trade:.1%} risk/trade, "
                   f"{max_portfolio_risk:.0%} max deployment")

class SingletonTradingSystem:
    """
    Singleton trading system - ensures only ONE instance exists
    
    Prevents the duplicate engine bug by using thread-safe singleton pattern
    """
    
    _instance: Optional['SingletonTradingSystem'] = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        """Thread-safe singleton implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize only once"""
        if not self._initialized:
            self.config: Optional[TradingSystemConfig] = None
            self.portfolio: Optional[ProfessionalPortfolioTracker] = None
            self.creation_time = datetime.now()
            self.access_count = 0
            
            SingletonTradingSystem._initialized = True
            logger.info("🚀 Singleton Trading System created")
    
    def initialize(self, config: TradingSystemConfig) -> None:
        """Initialize the trading system with configuration"""
        
        if self.config is not None:
            raise RuntimeError("Trading system already initialized! "
                             "Cannot create multiple instances.")
        
        self.config = config
        self.portfolio = ProfessionalPortfolioTracker(config.starting_capital)
        
        logger.info(f"✅ Trading system initialized with ${config.starting_capital:,.0f}")
        logger.info(f"   📊 Max positions: {config.max_concurrent_positions}")
        logger.info(f"   ⚖️ Risk per trade: {config.max_risk_per_trade:.1%}")
        logger.info(f"   📈 Max deployment: {config.max_portfolio_risk:.0%}")
    
    def get_portfolio(self) -> ProfessionalPortfolioTracker:
        """Get the portfolio tracker"""
        if self.portfolio is None:
            raise RuntimeError("Trading system not initialized! Call initialize() first.")
        
        self.access_count += 1
        return self.portfolio
    
    def get_config(self) -> TradingSystemConfig:
        """Get the trading configuration"""
        if self.config is None:
            raise RuntimeError("Trading system not initialized! Call initialize() first.")
        
        return self.config
    
    def is_initialized(self) -> bool:
        """Check if system is initialized"""
        return self.config is not None and self.portfolio is not None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for debugging"""
        return {
            'initialized': self.is_initialized(),
            'creation_time': self.creation_time,
            'access_count': self.access_count,
            'config': {
                'starting_capital': self.config.starting_capital if self.config else None,
                'max_positions': self.config.max_concurrent_positions if self.config else None,
            } if self.config else None,
            'portfolio_summary': self.portfolio.get_portfolio_summary() if self.portfolio else None
        }

# Global functions to access the singleton
def get_trading_system() -> SingletonTradingSystem:
    """Get the singleton trading system instance"""
    return SingletonTradingSystem()

def get_portfolio() -> ProfessionalPortfolioTracker:
    """Get the portfolio tracker from singleton system"""
    return get_trading_system().get_portfolio()

def get_trading_config() -> TradingSystemConfig:
    """Get the trading configuration from singleton system"""
    return get_trading_system().get_config()

def initialize_trading_system(starting_capital: float = 10000.0,
                             max_risk_per_trade: float = 0.015,
                             max_portfolio_risk: float = 0.60,
                             max_concurrent_positions: int = 8) -> SingletonTradingSystem:
    """
    Initialize the trading system (call once at application start)
    
    Returns: The singleton trading system instance
    Raises: RuntimeError if already initialized
    """
    
    config = TradingSystemConfig(
        starting_capital=starting_capital,
        max_risk_per_trade=max_risk_per_trade,
        max_portfolio_risk=max_portfolio_risk,
        max_concurrent_positions=max_concurrent_positions
    )
    
    system = get_trading_system()
    system.initialize(config)
    
    return system

def reset_trading_system() -> None:
    """
    Reset singleton for testing purposes only
    
    ⚠️ WARNING: Only use this in unit tests!
    """
    SingletonTradingSystem._instance = None
    SingletonTradingSystem._initialized = False
    logger.warning("⚠️ Trading system reset (testing only)")

# Test the singleton pattern
def test_singleton_enforcement():
    """Test that singleton prevents multiple instances"""
    
    print("🧪 TESTING SINGLETON ENFORCEMENT")
    print("=" * 50)
    
    # Reset for clean test
    reset_trading_system()
    
    # Test 1: Initialize system
    print("Test 1: Initialize trading system")
    system1 = initialize_trading_system(
        starting_capital=10000.0,
        max_risk_per_trade=0.02,
        max_concurrent_positions=5
    )
    
    info1 = system1.get_system_info()
    print(f"   System 1 created at: {info1['creation_time']}")
    print(f"   Starting capital: ${info1['config']['starting_capital']:,.0f}")
    
    # Test 2: Try to create another instance
    print("\\nTest 2: Try to create second instance (should fail)")
    try:
        system2 = initialize_trading_system(
            starting_capital=5000.0,  # Different config
            max_risk_per_trade=0.05
        )
        print("❌ ERROR: Second system created! Singleton broken!")
    except RuntimeError as e:
        print(f"✅ SUCCESS: {e}")
    
    # Test 3: Get same instance from different calls
    print("\\nTest 3: Verify same instance returned")
    system3 = get_trading_system()
    system4 = get_trading_system()
    
    if system1 is system3 is system4:
        print("✅ SUCCESS: All calls return same instance")
    else:
        print("❌ ERROR: Different instances returned!")
    
    # Test 4: Check access counting
    portfolio1 = get_portfolio()
    portfolio2 = get_portfolio()
    
    final_info = system1.get_system_info()
    print(f"\\nAccess count: {final_info['access_count']}")
    
    if portfolio1 is portfolio2:
        print("✅ SUCCESS: Same portfolio instance returned")
    else:
        print("❌ ERROR: Different portfolio instances!")
    
    # Test 5: Show portfolio is working
    portfolio = get_portfolio()
    portfolio.update_price('BTC/USDT', 67000.0)
    
    result = portfolio.open_position(
        symbol='BTC/USDT',
        side='LONG', 
        quantity=0.1,
        entry_price=67000.0,
        fees=5.0
    )
    
    print(f"\\nTest position opened: {result['success']}")
    print("Final portfolio state:")
    print(portfolio.get_portfolio_summary())
    
    # Validation
    if portfolio.validate_portfolio_integrity():
        print("\\n✅ SINGLETON SYSTEM WORKING PERFECTLY!")
        print("   - Only one trading engine instance")
        print("   - Accurate portfolio tracking")  
        print("   - No cash allocation conflicts")
    else:
        print("\\n❌ Validation failed!")

if __name__ == "__main__":
    test_singleton_enforcement()