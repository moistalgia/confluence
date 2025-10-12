#!/usr/bin/env python3
"""
Configuration Management Demonstration
Shows how to use the advanced configuration system for crypto analysis
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from config_manager import ConfigManager, CryptoAnalyzerConfig, ConfigFormat
from enhanced_multi_timeframe_analyzer import EnhancedMultiTimeframeAnalyzer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_config_system():
    """Demonstrate the configuration management system"""
    
    print("=== Crypto Analyzer Configuration Management Demo ===\n")
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # 1. Create and save different trading profiles
    print("1. Creating Trading Profiles...")
    
    # Scalping profile - short timeframes, fast signals
    scalping_config = config_manager.create_trading_profile("scalping", "scalping")
    print(f"   ✅ Created scalping profile: {scalping_config.profile_name}")
    
    # Swing trading profile - medium timeframes
    swing_config = config_manager.create_trading_profile("swing_trading", "swing")
    print(f"   ✅ Created swing trading profile: {swing_config.profile_name}")
    
    # Position trading profile - long timeframes
    position_config = config_manager.create_trading_profile("position_trading", "position")
    print(f"   ✅ Created position trading profile: {position_config.profile_name}")
    
    print()
    
    # 2. List available profiles
    print("2. Available Profiles:")
    profiles = config_manager.list_profiles()
    for profile in profiles:
        print(f"   - {profile}")
    print()
    
    # 3. Load and validate configuration
    print("3. Loading and Validating Configuration...")
    config = config_manager.load_config("swing_trading")
    validation_issues = config_manager.validate_config(config)
    
    if validation_issues:
        print("   ⚠️ Configuration validation issues:")
        for section, issues in validation_issues.items():
            print(f"     {section}: {', '.join(issues)}")
    else:
        print("   ✅ Configuration validation passed")
    print()
    
    # 4. Customize configuration
    print("4. Customizing Configuration...")
    
    # Update RSI settings
    config.indicators.rsi_period = 21
    config.indicators.rsi_overbought = 75
    config.indicators.rsi_oversold = 25
    
    # Update performance settings
    config.performance.max_workers = 6
    config.performance.enable_caching = True
    
    # Save customized config
    config_manager.save_config(config, "custom_swing")
    print("   ✅ Saved customized configuration as 'custom_swing'")
    print()
    
    # 5. Create analyzer with different configurations
    print("5. Creating Analyzers with Different Configurations...")
    
    # Default analyzer
    analyzer_default = EnhancedMultiTimeframeAnalyzer()
    print(f"   ✅ Default analyzer: {analyzer_default.config.profile_name}")
    print(f"      - Timeframes: {list(analyzer_default.timeframes.keys())}")
    print(f"      - Parallel: {analyzer_default.parallel_processing}, Workers: {analyzer_default.max_workers}")
    
    # Custom configuration analyzer
    analyzer_custom = EnhancedMultiTimeframeAnalyzer(config_profile="custom_swing")
    print(f"   ✅ Custom analyzer: {analyzer_custom.config.profile_name}")
    print(f"      - RSI Period: {analyzer_custom.config.indicators.rsi_period}")
    print(f"      - Workers: {analyzer_custom.max_workers}")
    
    # Scalping analyzer
    analyzer_scalping = EnhancedMultiTimeframeAnalyzer(config_profile="scalping")
    print(f"   ✅ Scalping analyzer: {analyzer_scalping.config.profile_name}")
    print(f"      - Timeframes: {list(analyzer_scalping.timeframes.keys())}")
    
    print()
    
    # 6. Show configuration details
    print("6. Configuration Details (Swing Trading):")
    print(f"   Profile: {config.profile_name}")
    print(f"   Exchange: {config.exchange.exchange_id}")
    print(f"   Indicators:")
    print(f"     - RSI: Period {config.indicators.rsi_period}, OB/OS: {config.indicators.rsi_overbought}/{config.indicators.rsi_oversold}")
    print(f"     - MACD: {config.indicators.macd_fast}/{config.indicators.macd_slow}/{config.indicators.macd_signal}")
    print(f"     - BB: Period {config.indicators.bb_period}, StdDev {config.indicators.bb_std}")
    print(f"   Performance:")
    print(f"     - Caching: {config.performance.enable_caching}, TTL: {config.performance.ohlcv_cache_ttl}s")
    print(f"     - Parallel: {config.performance.parallel_processing}, Workers: {config.performance.max_workers}")
    print(f"   Timeframes:")
    for tf_name, tf_config in config.timeframes.items():
        if tf_config.enabled:
            print(f"     - {tf_name}: Weight {tf_config.weight}, Periods {tf_config.periods}, Priority {tf_config.priority}")
    print()
    
    return config_manager, analyzer_custom

def demonstrate_runtime_config_updates():
    """Demonstrate runtime configuration updates"""
    
    print("=== Runtime Configuration Updates ===\n")
    
    config_manager = ConfigManager()
    config = config_manager.load_config("swing_trading")
    
    print("1. Original Configuration:")
    print(f"   RSI Period: {config.indicators.rsi_period}")
    print(f"   Max Workers: {config.performance.max_workers}")
    print()
    
    # Update configuration at runtime
    print("2. Updating Configuration...")
    updates = {
        'indicators': {
            'rsi_period': 25,
            'rsi_overbought': 80
        },
        'performance': {
            'max_workers': 8
        }
    }
    
    success = config_manager.update_config(updates)
    if success:
        updated_config = config_manager.get_current_config()
        print("   ✅ Configuration updated successfully")
        print(f"   New RSI Period: {updated_config.indicators.rsi_period}")
        print(f"   New Max Workers: {updated_config.performance.max_workers}")
    else:
        print("   ❌ Configuration update failed")
    
    print()

def demonstrate_custom_analyzer_creation():
    """Demonstrate creating analyzer with custom overrides"""
    
    print("=== Custom Analyzer Creation ===\n")
    
    # Create analyzer with specific overrides
    custom_analyzer = EnhancedMultiTimeframeAnalyzer.create_with_custom_config(
        exchange={"exchange_id": "binance", "timeout": 20000},
        performance={"max_workers": 8, "enable_caching": False},
        indicators={"rsi_period": 30, "macd_fast": 8, "macd_slow": 21}
    )
    
    print("Created custom analyzer with overrides:")
    print(f"   Exchange: {custom_analyzer.config.exchange.exchange_id}")
    print(f"   Timeout: {custom_analyzer.config.exchange.timeout}")
    print(f"   Max Workers: {custom_analyzer.max_workers}")
    print(f"   Caching: {custom_analyzer._cache.config.enabled}")
    print(f"   RSI Period: {custom_analyzer.config.indicators.rsi_period}")
    print(f"   MACD Fast: {custom_analyzer.config.indicators.macd_fast}")
    
    return custom_analyzer

if __name__ == "__main__":
    try:
        # Run demonstrations
        config_manager, analyzer = demonstrate_config_system()
        demonstrate_runtime_config_updates()
        custom_analyzer = demonstrate_custom_analyzer_creation()
        
        print("\n=== Configuration Management Demo Complete ===")
        print("\nKey Features Demonstrated:")
        print("✅ Multiple trading profile creation (scalping, swing, position)")
        print("✅ Configuration validation and error checking")
        print("✅ Runtime configuration updates")
        print("✅ Custom analyzer creation with overrides")
        print("✅ Profile-based analyzer initialization")
        print("✅ Comprehensive configuration persistence (JSON format)")
        
        print(f"\nConfiguration files created in: {config_manager.config_dir}")
        print("Available profiles:", ", ".join(config_manager.list_profiles()))
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise