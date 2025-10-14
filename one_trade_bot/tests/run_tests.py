"""
Test Runner Script
🎯 Run all test suites for the One Trade Bot

Executes comprehensive test coverage across all bot components
"""

import unittest
import sys
import os
from io import StringIO
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test_suite():
    """Run complete test suite with detailed reporting"""
    
    print("🎯 ONE TRADE BOT - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()
    
    # Test modules to run
    test_modules = [
        'tests.test_regime_filter',
        'tests.test_setup_scanner', 
        'tests.test_confluence_checker',
        'tests.test_workflow_engine',
        'tests.test_backtester'
    ]
    
    # Track overall results
    total_tests = 0
    total_failures = 0
    total_errors = 0
    suite_results = {}
    
    start_time = time.time()
    
    for module_name in test_modules:
        print(f"🔍 Running {module_name.replace('tests.', '').replace('test_', '').upper()} Tests...")
        
        # Create test suite for this module
        try:
            module = __import__(module_name, fromlist=[''])
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            
            # Run tests with custom result handler
            stream = StringIO()
            runner = unittest.TextTestRunner(
                stream=stream, 
                verbosity=2,
                failfast=False
            )
            
            result = runner.run(suite)
            
            # Capture results
            tests_run = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            
            total_tests += tests_run
            total_failures += failures
            total_errors += errors
            
            # Store detailed results
            suite_results[module_name] = {
                'tests_run': tests_run,
                'failures': failures,
                'errors': errors,
                'success_rate': ((tests_run - failures - errors) / tests_run * 100) if tests_run > 0 else 0,
                'details': stream.getvalue()
            }
            
            # Print summary for this module
            if failures == 0 and errors == 0:
                print(f"   ✅ {tests_run} tests passed ({suite_results[module_name]['success_rate']:.1f}%)")
            else:
                print(f"   ❌ {tests_run} tests: {tests_run - failures - errors} passed, {failures} failed, {errors} errors")
            
            print()
            
        except Exception as e:
            print(f"   💥 ERROR loading test module: {e}")
            print()
            continue
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print overall summary
    print("=" * 60)
    print("📊 OVERALL TEST RESULTS")
    print("=" * 60)
    
    success_tests = total_tests - total_failures - total_errors
    overall_success_rate = (success_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests:     {total_tests}")
    print(f"Passed:          {success_tests}")
    print(f"Failed:          {total_failures}")
    print(f"Errors:          {total_errors}")
    print(f"Success Rate:    {overall_success_rate:.1f}%")
    print(f"Duration:        {duration:.2f} seconds")
    print()
    
    # Print individual module results
    print("📋 DETAILED MODULE RESULTS")
    print("-" * 60)
    
    for module_name, results in suite_results.items():
        module_display = module_name.replace('tests.test_', '').replace('_', ' ').title()
        status = "✅ PASS" if results['failures'] == 0 and results['errors'] == 0 else "❌ FAIL"
        
        print(f"{module_display:20} | {status} | {results['success_rate']:5.1f}% | {results['tests_run']} tests")
    
    print()
    
    # Print failure details if any
    if total_failures > 0 or total_errors > 0:
        print("🔍 FAILURE DETAILS")
        print("-" * 60)
        
        for module_name, results in suite_results.items():
            if results['failures'] > 0 or results['errors'] > 0:
                print(f"\n{module_name}:")
                print(results['details'])
    
    # Final assessment
    if total_failures == 0 and total_errors == 0:
        print("🎉 ALL TESTS PASSED! Bot is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please review and fix issues before deployment.")
        return False

def run_specific_test(test_name):
    """Run a specific test module"""
    
    if not test_name.startswith('test_'):
        test_name = f'test_{test_name}'
    
    module_name = f'tests.{test_name}'
    
    print(f"🎯 Running specific test: {test_name}")
    print("=" * 40)
    
    try:
        module = __import__(module_name, fromlist=[''])
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.failures == 0 and result.errors == 0:
            print(f"\n✅ All {result.testsRun} tests passed!")
            return True
        else:
            print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import test module '{module_name}': {e}")
        return False

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run full test suite
        success = run_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)