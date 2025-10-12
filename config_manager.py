#!/usr/bin/env python3
"""
Advanced Configuration Management System
Provides user-customizable analysis parameters, indicator settings, and analysis profiles
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass, asdict, field
from enum import Enum
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional YAML support - will work without it
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available - YAML format disabled")

class ConfigFormat(Enum):
    """Supported configuration file formats"""
    JSON = "json"
    YAML = "yaml"

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    # RSI Settings
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # MACD Settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_threshold: float = 0.1
    
    # Moving Averages
    sma_short: int = 10
    sma_long: int = 50
    ema_period: int = 21
    
    # ATR
    atr_period: int = 14
    atr_multiplier: float = 2.0
    
    # ADX
    adx_period: int = 14
    adx_threshold: float = 25.0
    
    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    
    # Volume Profile
    vp_bins: int = 50
    vp_value_area_percentage: float = 70.0

@dataclass
class TimeframeConfig:
    """Configuration for individual timeframes"""
    enabled: bool = True
    weight: float = 0.25
    periods: int = 200
    priority: int = 2
    indicators_enabled: List[str] = field(default_factory=lambda: [
        'rsi', 'macd', 'bollinger_bands', 'atr', 'adx', 'stochastic', 'volume_profile'
    ])

@dataclass
class ExchangeConfig:
    """Exchange connection configuration"""
    exchange_id: str = 'kraken'
    enable_rate_limit: bool = True
    timeout: int = 30000
    rate_limit: int = 1000
    sandbox_mode: bool = False
    api_key: str = ""
    api_secret: str = ""

@dataclass 
class PerformanceConfig:
    """Performance optimization settings"""
    enable_caching: bool = True
    cache_dir: str = "cache/crypto_analyzer"
    ohlcv_cache_ttl: int = 300  # 5 minutes
    indicators_cache_ttl: int = 60  # 1 minute
    max_cache_size: int = 100
    
    parallel_processing: bool = True
    max_workers: int = 4
    timeout_per_timeframe: int = 60

@dataclass
class AnalysisConfig:
    """Main analysis configuration settings"""
    # Analysis behavior
    min_data_points: int = 50
    data_quality_threshold: float = 0.8
    confluence_threshold: float = 0.6
    signal_strength_threshold: float = 0.7
    
    # Risk management
    risk_free_rate: float = 0.05
    max_position_size: float = 0.1
    stop_loss_atr_multiple: float = 2.0
    take_profit_ratio: float = 2.0
    
    # Market structure
    swing_strength: int = 5
    structure_break_threshold: float = 0.02
    trend_strength_period: int = 20

@dataclass
class OutputConfig:
    """Output and reporting configuration"""
    output_dir: str = "output/ultimate_analysis"
    save_raw_data: bool = True
    generate_charts: bool = False
    chart_format: str = "png"
    
    # Report formatting
    use_ascii_formatting: bool = True
    include_performance_metrics: bool = True
    detailed_logging: bool = True
    log_level: str = "INFO"

@dataclass
class CryptoAnalyzerConfig:
    """Complete configuration for the crypto analyzer"""
    version: str = "2.0.0"
    profile_name: str = "default"
    
    # Component configurations
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Timeframe configurations
    timeframes: Dict[str, TimeframeConfig] = field(default_factory=lambda: {
        '1w': TimeframeConfig(
            enabled=True,
            weight=0.35,
            periods=100,
            priority=1,
            indicators_enabled=['rsi', 'macd', 'bollinger_bands', 'atr', 'adx']
        ),
        '1d': TimeframeConfig(
            enabled=True,
            weight=0.40,
            periods=200,
            priority=1,
            indicators_enabled=['rsi', 'macd', 'bollinger_bands', 'atr', 'adx', 'stochastic', 'volume_profile', 'williams_r', 'cci', 'parabolic_sar', 'ichimoku']
        ),
        '4h': TimeframeConfig(
            enabled=True,
            weight=0.20,
            periods=300,
            priority=2,
            indicators_enabled=['rsi', 'macd', 'bollinger_bands', 'atr', 'stochastic', 'adx', 'williams_r', 'cci', 'parabolic_sar']
        ),
        '1h': TimeframeConfig(
            enabled=True,
            weight=0.05,
            periods=168,
            priority=3,
            indicators_enabled=['rsi', 'macd', 'bollinger_bands', 'atr', 'stochastic', 'adx', 'williams_r', 'cci', 'parabolic_sar']
        )
    })

class ConfigManager:
    """
    Advanced configuration management system with profiles, validation, and hot-reloading
    """
    
    def __init__(self, config_dir: str = "config", default_profile: str = "default"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.default_profile = default_profile
        self.current_config: Optional[CryptoAnalyzerConfig] = None
        self.config_file_path: Optional[Path] = None
        
    def load_config(self, profile_name: Optional[str] = None, format_type: ConfigFormat = ConfigFormat.JSON) -> CryptoAnalyzerConfig:
        """
        Load configuration from file or create default if not exists
        """
        try:
            profile = profile_name or self.default_profile
            
            # Determine file path based on format
            if format_type == ConfigFormat.YAML:
                config_file = self.config_dir / f"{profile}.yaml"
            else:
                config_file = self.config_dir / f"{profile}.json"
            
            self.config_file_path = config_file
            
            if config_file.exists():
                logger.info(f"Loading configuration from {config_file}")
                
                with open(config_file, 'r', encoding='utf-8') as f:
                    if format_type == ConfigFormat.YAML:
                        if not YAML_AVAILABLE:
                            raise ValueError("YAML format requested but PyYAML not installed")
                        config_dict = yaml.safe_load(f)
                    else:
                        config_dict = json.load(f)
                
                # Convert dict to config object with validation
                config = self._dict_to_config(config_dict)
                self.current_config = config
                
                logger.info(f"Successfully loaded {profile} configuration")
                return config
            else:
                logger.info(f"Configuration file not found, creating default: {config_file}")
                return self.create_default_config(profile, format_type)
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Falling back to default configuration")
            return self.create_default_config(profile_name or self.default_profile, format_type)
    
    def save_config(self, config: CryptoAnalyzerConfig, profile_name: Optional[str] = None, 
                   format_type: ConfigFormat = ConfigFormat.JSON) -> bool:
        """
        Save configuration to file
        """
        try:
            profile = profile_name or config.profile_name
            config.profile_name = profile  # Update profile name in config
            
            # Determine file path based on format
            if format_type == ConfigFormat.YAML:
                config_file = self.config_dir / f"{profile}.yaml"
            else:
                config_file = self.config_dir / f"{profile}.json"
            
            # Convert config to dict
            config_dict = asdict(config)
            
            # Save to file
            with open(config_file, 'w', encoding='utf-8') as f:
                if format_type == ConfigFormat.YAML:
                    if not YAML_AVAILABLE:
                        raise ValueError("YAML format requested but PyYAML not installed")
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.config_file_path = config_file
            self.current_config = config
            
            logger.info(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def create_default_config(self, profile_name: str = "default", 
                            format_type: ConfigFormat = ConfigFormat.JSON) -> CryptoAnalyzerConfig:
        """
        Create and save default configuration
        """
        config = CryptoAnalyzerConfig(profile_name=profile_name)
        
        # Save the default configuration
        if self.save_config(config, profile_name, format_type):
            logger.info(f"Default configuration created: {profile_name}")
        
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> CryptoAnalyzerConfig:
        """
        Convert dictionary to CryptoAnalyzerConfig with validation
        """
        try:
            # Handle nested configurations
            if 'indicators' in config_dict:
                config_dict['indicators'] = IndicatorConfig(**config_dict['indicators'])
            
            if 'exchange' in config_dict:
                config_dict['exchange'] = ExchangeConfig(**config_dict['exchange'])
            
            if 'performance' in config_dict:
                config_dict['performance'] = PerformanceConfig(**config_dict['performance'])
                
            if 'analysis' in config_dict:
                config_dict['analysis'] = AnalysisConfig(**config_dict['analysis'])
                
            if 'output' in config_dict:
                config_dict['output'] = OutputConfig(**config_dict['output'])
            
            # Handle timeframes
            if 'timeframes' in config_dict:
                timeframes = {}
                for tf_name, tf_config in config_dict['timeframes'].items():
                    if isinstance(tf_config, dict):
                        timeframes[tf_name] = TimeframeConfig(**tf_config)
                    else:
                        timeframes[tf_name] = tf_config
                config_dict['timeframes'] = timeframes
            
            return CryptoAnalyzerConfig(**config_dict)
            
        except Exception as e:
            logger.error(f"Error converting dict to config: {e}")
            logger.info("Using default configuration")
            return CryptoAnalyzerConfig()
    
    def list_profiles(self) -> List[str]:
        """
        List all available configuration profiles
        """
        profiles = []
        for file_path in self.config_dir.glob("*.json"):
            profiles.append(file_path.stem)
        for file_path in self.config_dir.glob("*.yaml"):
            if file_path.stem not in profiles:  # Avoid duplicates
                profiles.append(file_path.stem)
        return sorted(profiles)
    
    def validate_config(self, config: CryptoAnalyzerConfig) -> Dict[str, List[str]]:
        """
        Validate configuration and return any issues
        """
        issues = {}
        
        # Validate indicators
        indicator_issues = []
        if config.indicators.rsi_period < 1 or config.indicators.rsi_period > 100:
            indicator_issues.append("RSI period must be between 1 and 100")
        if not (0 < config.indicators.rsi_overbought <= 100):
            indicator_issues.append("RSI overbought level must be between 0 and 100")
        if not (0 <= config.indicators.rsi_oversold < 100):
            indicator_issues.append("RSI oversold level must be between 0 and 100")
        if config.indicators.rsi_oversold >= config.indicators.rsi_overbought:
            indicator_issues.append("RSI oversold must be less than overbought")
        
        if indicator_issues:
            issues['indicators'] = indicator_issues
        
        # Validate timeframes
        timeframe_issues = []
        enabled_timeframes = [tf for tf, cfg in config.timeframes.items() if cfg.enabled]
        if not enabled_timeframes:
            timeframe_issues.append("At least one timeframe must be enabled")
        
        total_weight = sum(cfg.weight for cfg in config.timeframes.values() if cfg.enabled)
        if abs(total_weight - 1.0) > 0.01:
            timeframe_issues.append(f"Enabled timeframe weights should sum to 1.0 (currently: {total_weight:.2f})")
        
        if timeframe_issues:
            issues['timeframes'] = timeframe_issues
        
        # Validate performance settings
        performance_issues = []
        if config.performance.max_workers < 1 or config.performance.max_workers > 16:
            performance_issues.append("Max workers must be between 1 and 16")
        if config.performance.ohlcv_cache_ttl < 10:
            performance_issues.append("OHLCV cache TTL should be at least 10 seconds")
        
        if performance_issues:
            issues['performance'] = performance_issues
        
        return issues
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update current configuration with new values
        """
        if not self.current_config:
            logger.error("No current configuration loaded")
            return False
        
        try:
            # Apply updates to current config
            for key, value in updates.items():
                if hasattr(self.current_config, key):
                    current_obj = getattr(self.current_config, key)
                    
                    if hasattr(current_obj, '__dict__'):  # It's a dataclass
                        if isinstance(value, dict):
                            # Update individual fields in the dataclass
                            for field_name, field_value in value.items():
                                if hasattr(current_obj, field_name):
                                    setattr(current_obj, field_name, field_value)
                    elif isinstance(current_obj, dict):
                        # Handle dict updates
                        if isinstance(value, dict):
                            current_obj.update(value)
                    else:
                        # Direct assignment
                        setattr(self.current_config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            # Validate updated config
            validation_issues = self.validate_config(self.current_config)
            if validation_issues:
                logger.warning(f"Configuration validation issues: {validation_issues}")
            
            # Save updated config
            return self.save_config(self.current_config)
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def create_trading_profile(self, profile_name: str, trading_style: str = "balanced") -> CryptoAnalyzerConfig:
        """
        Create specialized trading profiles
        """
        config = CryptoAnalyzerConfig(profile_name=profile_name)
        
        if trading_style == "scalping":
            # Fast, short-term focused
            config.timeframes = {
                '5m': TimeframeConfig(True, 0.5, 288, 1, ['rsi', 'macd', 'bollinger_bands']),
                '15m': TimeframeConfig(True, 0.3, 96, 1, ['rsi', 'macd', 'bollinger_bands', 'stochastic']),
                '1h': TimeframeConfig(True, 0.2, 72, 2, ['rsi', 'macd', 'atr'])
            }
            config.performance.ohlcv_cache_ttl = 30  # 30 seconds
            config.analysis.signal_strength_threshold = 0.8
            
        elif trading_style == "swing":
            # Medium-term focused
            config.timeframes = {
                '4h': TimeframeConfig(True, 0.4, 300, 1, ['rsi', 'macd', 'bollinger_bands', 'atr', 'adx']),
                '1d': TimeframeConfig(True, 0.4, 200, 1, ['rsi', 'macd', 'bollinger_bands', 'atr', 'volume_profile']),
                '1w': TimeframeConfig(True, 0.2, 100, 2, ['rsi', 'macd', 'atr'])
            }
            config.analysis.swing_strength = 3
            config.analysis.signal_strength_threshold = 0.6
            
        elif trading_style == "position":
            # Long-term focused
            config.timeframes = {
                '1d': TimeframeConfig(True, 0.5, 365, 1, ['rsi', 'macd', 'bollinger_bands', 'atr', 'volume_profile']),
                '1w': TimeframeConfig(True, 0.4, 200, 1, ['rsi', 'macd', 'atr', 'adx']),
                '1M': TimeframeConfig(True, 0.1, 60, 2, ['rsi', 'macd'])
            }
            config.analysis.trend_strength_period = 50
            config.analysis.signal_strength_threshold = 0.5
        
        # Save the profile
        self.save_config(config, profile_name)
        return config
    
    def get_current_config(self) -> Optional[CryptoAnalyzerConfig]:
        """Get the currently loaded configuration"""
        return self.current_config

# Example usage and configuration templates
def create_example_configs():
    """Create example configuration files for different trading styles"""
    manager = ConfigManager()
    
    # Create different trading style profiles
    manager.create_trading_profile("scalping", "scalping")
    manager.create_trading_profile("swing_trading", "swing")
    manager.create_trading_profile("position_trading", "position")
    
    # Create default balanced profile
    manager.create_default_config("balanced")
    
    logger.info("Example configuration profiles created")

if __name__ == "__main__":
    create_example_configs()