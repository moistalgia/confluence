#!/usr/bin/env python3
"""
Enhanced Logging System for Crypto Analysis Pipeline
Provides comprehensive logging, verification, and data integrity checks
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import inspect

class CryptoAnalysisLogger:
    """
    Comprehensive logging system for crypto analysis pipeline
    Tracks data flow, verification points, and potential issues
    """
    
    def __init__(self, log_level=logging.INFO):
        self.setup_logging(log_level)
        self.verification_points = []
        self.data_checkpoints = {}
        self.analysis_session = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create detailed logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Session log file
        self.session_log_file = self.logs_dir / f"analysis_session_{self.analysis_session}.log"
        
        self.logger.info("="*80)
        self.logger.info("CRYPTO ANALYSIS PIPELINE - SESSION START")
        self.logger.info(f"Session ID: {self.analysis_session}")
        self.logger.info("="*80)
    
    def setup_logging(self, log_level):
        """Setup comprehensive logging configuration"""
        
        # Create logger
        self.logger = logging.getLogger('CryptoAnalysis')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (detailed)
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            logs_dir / f"crypto_analysis_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
    
    def log_function_entry(self, func_name: str, params: Dict = None):
        """Log function entry with parameters"""
        caller = inspect.stack()[1]
        module = caller.filename.split('\\')[-1] if '\\' in caller.filename else caller.filename.split('/')[-1]
        
        self.logger.info(f"🔄 ENTERING: {module}::{func_name}")
        if params:
            # Filter sensitive data
            safe_params = {k: v for k, v in params.items() 
                          if not any(sensitive in str(k).lower() 
                                   for sensitive in ['key', 'token', 'password', 'secret'])}
            self.logger.debug(f"   Parameters: {safe_params}")
    
    def log_function_exit(self, func_name: str, result_summary: str = None, success: bool = True):
        """Log function exit with results"""
        caller = inspect.stack()[1]
        module = caller.filename.split('\\')[-1] if '\\' in caller.filename else caller.filename.split('/')[-1]
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.logger.info(f"{status}: {module}::{func_name}")
        if result_summary:
            self.logger.info(f"   Result: {result_summary}")
    
    def log_data_checkpoint(self, checkpoint_name: str, data: Any, verify_func=None):
        """Log data checkpoint with optional verification"""
        self.logger.info(f"📊 DATA CHECKPOINT: {checkpoint_name}")
        
        # Basic data summary
        if isinstance(data, dict):
            summary = f"Dict with {len(data)} keys: {list(data.keys())[:5]}"
            if len(data) > 5:
                summary += "..."
        elif isinstance(data, list):
            summary = f"List with {len(data)} items"
        elif hasattr(data, '__len__'):
            summary = f"{type(data).__name__} with {len(data)} items"
        else:
            summary = f"{type(data).__name__}: {str(data)[:100]}"
        
        self.logger.info(f"   Data: {summary}")
        
        # Store checkpoint
        self.data_checkpoints[checkpoint_name] = {
            'timestamp': datetime.now().isoformat(),
            'data_type': type(data).__name__,
            'data_summary': summary,
            'verification_passed': None
        }
        
        # Optional verification
        if verify_func:
            try:
                verification_result = verify_func(data)
                self.data_checkpoints[checkpoint_name]['verification_passed'] = verification_result
                status = "✅ VERIFIED" if verification_result else "⚠️ VERIFICATION FAILED"
                self.logger.info(f"   {status}")
            except Exception as e:
                self.data_checkpoints[checkpoint_name]['verification_passed'] = False
                self.logger.error(f"   ❌ VERIFICATION ERROR: {e}")
    
    def log_ta_calculation(self, indicator_name: str, input_data_shape: tuple, 
                          output_data_shape: tuple = None, calculation_params: Dict = None):
        """Log technical analysis calculation details"""
        self.logger.info(f"📈 TA CALCULATION: {indicator_name}")
        self.logger.debug(f"   Input shape: {input_data_shape}")
        if output_data_shape:
            self.logger.debug(f"   Output shape: {output_data_shape}")
        if calculation_params:
            self.logger.debug(f"   Parameters: {calculation_params}")
    
    def log_exchange_request(self, exchange_name: str, symbol: str, timeframe: str, 
                           limit: int, success: bool, data_points: int = None):
        """Log exchange API request details"""
        status = "✅" if success else "❌"
        self.logger.info(f"{status} EXCHANGE REQUEST: {exchange_name}")
        self.logger.debug(f"   Symbol: {symbol}, Timeframe: {timeframe}, Limit: {limit}")
        if data_points is not None:
            self.logger.debug(f"   Data points received: {data_points}")
    
    def log_file_operation(self, operation: str, file_path: str, success: bool, 
                          file_size: int = None, error: str = None):
        """Log file operations (save/load)"""
        status = "✅" if success else "❌"
        self.logger.info(f"{status} FILE {operation.upper()}: {file_path}")
        if file_size:
            self.logger.debug(f"   File size: {file_size:,} bytes")
        if error:
            self.logger.error(f"   Error: {error}")
    
    def log_llm_interaction(self, model: str, input_tokens: int, output_tokens: int, 
                           cost: float, cached_tokens: int = 0, success: bool = True):
        """Log LLM API interactions"""
        status = "✅" if success else "❌"
        self.logger.info(f"{status} LLM INTERACTION: {model}")
        self.logger.info(f"   Input: {input_tokens:,} tokens, Output: {output_tokens:,} tokens")
        if cached_tokens > 0:
            self.logger.info(f"   Cached: {cached_tokens:,} tokens (cost savings!)")
        self.logger.info(f"   Cost: ${cost:.6f}")
    
    def log_verification_point(self, description: str, passed: bool, details: str = None):
        """Log verification point results"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.logger.info(f"{status} VERIFICATION: {description}")
        if details:
            self.logger.debug(f"   Details: {details}")
        
        self.verification_points.append({
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'passed': passed,
            'details': details
        })
    
    def log_data_integrity_check(self, data_source: str, expected_fields: List[str], 
                               actual_data: Dict, critical_fields: List[str] = None):
        """Comprehensive data integrity verification"""
        self.logger.info(f"🔍 DATA INTEGRITY CHECK: {data_source}")
        
        missing_fields = [field for field in expected_fields if field not in actual_data]
        present_fields = [field for field in expected_fields if field in actual_data]
        
        if critical_fields:
            missing_critical = [field for field in critical_fields if field not in actual_data]
        else:
            missing_critical = []
        
        # Log results
        self.logger.info(f"   Expected fields: {len(expected_fields)}")
        self.logger.info(f"   Present fields: {len(present_fields)}")
        self.logger.info(f"   Missing fields: {len(missing_fields)}")
        
        if missing_fields:
            self.logger.warning(f"   Missing: {missing_fields}")
        
        if missing_critical:
            self.logger.error(f"   CRITICAL MISSING: {missing_critical}")
            return False
        
        # Check for null/empty values in critical data
        null_critical = []
        for field in (critical_fields or expected_fields):
            if field in actual_data:
                value = actual_data[field]
                if value is None or (isinstance(value, (list, dict, str)) and len(value) == 0):
                    null_critical.append(field)
        
        if null_critical:
            self.logger.error(f"   NULL/EMPTY CRITICAL FIELDS: {null_critical}")
            return False
        
        self.logger.info("   ✅ DATA INTEGRITY VERIFIED")
        return True
    
    def generate_session_report(self) -> Dict:
        """Generate comprehensive session analysis report"""
        report = {
            'session_id': self.analysis_session,
            'timestamp': datetime.now().isoformat(),
            'data_checkpoints': self.data_checkpoints,
            'verification_points': self.verification_points,
            'summary': {
                'total_checkpoints': len(self.data_checkpoints),
                'total_verifications': len(self.verification_points),
                'passed_verifications': sum(1 for vp in self.verification_points if vp['passed']),
                'failed_verifications': sum(1 for vp in self.verification_points if not vp['passed'])
            }
        }
        
        # Save session report
        report_file = self.logs_dir / f"session_report_{self.analysis_session}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info("="*80)
        self.logger.info("SESSION ANALYSIS REPORT")
        self.logger.info(f"Total Data Checkpoints: {report['summary']['total_checkpoints']}")
        self.logger.info(f"Total Verifications: {report['summary']['total_verifications']}")
        self.logger.info(f"Passed Verifications: {report['summary']['passed_verifications']}")
        self.logger.info(f"Failed Verifications: {report['summary']['failed_verifications']}")
        self.logger.info(f"Session Report: {report_file}")
        self.logger.info("="*80)
        
        return report

# Global logger instance
crypto_logger = CryptoAnalysisLogger()

def log_function(func):
    """Decorator to automatically log function entry/exit"""
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        params = {**{f'arg_{i}': arg for i, arg in enumerate(args)}, **kwargs}
        
        crypto_logger.log_function_entry(func_name, params)
        
        try:
            result = func(*args, **kwargs)
            crypto_logger.log_function_exit(func_name, success=True)
            return result
        except Exception as e:
            crypto_logger.log_function_exit(func_name, success=False)
            crypto_logger.logger.error(f"Exception in {func_name}: {e}")
            raise
    
    return wrapper

def main():
    """Test the logging system"""
    
    logger = CryptoAnalysisLogger(logging.DEBUG)
    
    # Test various logging features
    logger.log_function_entry("test_function", {"symbol": "BTC/USDT", "timeframe": "1h"})
    
    logger.log_data_checkpoint("test_data", {"prices": [1, 2, 3], "volumes": [100, 200, 300]})
    
    logger.log_ta_calculation("RSI", (100, 4), (100,), {"period": 14})
    
    logger.log_exchange_request("Kraken", "BTC/USDT", "1h", 100, True, 100)
    
    logger.log_verification_point("Data completeness", True, "All required fields present")
    
    logger.log_llm_interaction("claude-3-5-sonnet", 1000, 500, 0.025, 800)
    
    # Generate session report
    report = logger.generate_session_report()
    
    print("✅ Logging system test complete!")

if __name__ == "__main__":
    main()