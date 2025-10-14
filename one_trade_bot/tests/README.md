# One Trade Bot Test Suite

This directory contains comprehensive tests for all components of the One Good Trade Per Day bot.

## 🎯 Test Coverage

### Unit Tests
- **`test_regime_filter.py`** - Tests regime detection and market condition analysis
- **`test_setup_scanner.py`** - Tests pullback pattern detection and setup validation
- **`test_confluence_checker.py`** - Tests multi-timeframe confluence analysis
- **`test_workflow_engine.py`** - Tests complete workflow orchestration
- **`test_backtester.py`** - Tests historical backtesting functionality

### Integration Tests
- **`test_integration.py`** - End-to-end workflow integration testing

### Test Configuration
- **`conftest.py`** - Shared test fixtures and mock data generators
- **`run_tests.py`** - Test runner script with detailed reporting

## 🚀 Running Tests

### Run All Tests
```bash
cd one_trade_bot/tests
python run_tests.py
```

### Run Specific Test Suite
```bash
python run_tests.py regime_filter
python run_tests.py setup_scanner
python run_tests.py confluence_checker
python run_tests.py workflow_engine
python run_tests.py backtester
python run_tests.py integration
```

### Run Individual Test Methods
```bash
python -m unittest test_regime_filter.TestRegimeFilter.test_regime_filter_initialization
python -m unittest test_setup_scanner.TestSetupScanner.test_detect_pullback_pattern_present
```

## 📊 Test Structure

Each test module follows a consistent structure:

### 1. Setup Phase
- Mock data generation for realistic market conditions
- Configuration of test dependencies
- Initialization of components under test

### 2. Test Categories
- **Initialization Tests** - Verify proper component setup
- **Data Processing Tests** - Test indicator calculations and data validation
- **Logic Tests** - Test filtering logic and decision making
- **Integration Tests** - Test component interaction
- **Error Handling Tests** - Test graceful error management
- **Edge Case Tests** - Test boundary conditions

### 3. Mock Data Types
- **Trending Markets** - Strong uptrends/downtrends for setup detection
- **Sideways Markets** - Range-bound conditions for regime filtering
- **Volatile Markets** - High volatility for risk management testing
- **Low Volume Markets** - Testing volume-based filters

## 🔍 Test Scenarios

### Regime Filter Tests
- ✅ Trending market detection
- ✅ Volatility regime analysis
- ✅ Volume condition validation
- ✅ Market structure assessment
- ✅ Minimum checks requirement enforcement

### Setup Scanner Tests
- ✅ Pullback pattern detection
- ✅ Trend strength validation
- ✅ Setup quality scoring
- ✅ Minimum trend bars requirement
- ✅ Pullback depth validation

### Confluence Checker Tests
- ✅ Multi-timeframe analysis
- ✅ SMA alignment checking
- ✅ Support/resistance identification
- ✅ Momentum indicator analysis
- ✅ Minimum confluence count enforcement

### Workflow Engine Tests
- ✅ Complete 5-filter pipeline execution
- ✅ Filter result aggregation
- ✅ Trade direction determination
- ✅ Exception handling
- ✅ Timing and performance tracking

### Backtester Tests
- ✅ Historical simulation
- ✅ Performance metrics calculation
- ✅ Slippage and commission modeling
- ✅ Filter effectiveness analysis
- ✅ Risk management validation

### Integration Tests
- ✅ End-to-end workflow execution
- ✅ Component initialization chain
- ✅ Error propagation handling
- ✅ Configuration validation
- ✅ Performance tracking integration

## 📈 Test Metrics

The test suite tracks:
- **Code Coverage** - Percentage of code executed during tests
- **Success Rate** - Percentage of tests passing per module
- **Performance** - Test execution time and bottlenecks
- **Mock Data Quality** - Realism of generated test data

## 🛠 Mock Data Generation

### Market Scenarios
```python
# Bullish trending market with pullback
bullish_data = mock_ohlcv_data(days=30, symbol='BTC/USDT', trend='up')

# High volatility choppy market
volatile_data = mock_ohlcv_data(days=30, symbol='ETH/USDT', trend='sideways', volatility='high')

# Clean downtrend
bearish_data = mock_ohlcv_data(days=30, symbol='SOL/USDT', trend='down')
```

### Trade Scenarios
- **Winning Trades** - Profit target hits with 2:1 risk:reward
- **Losing Trades** - Stop loss triggers with 1% account risk
- **Breakeven Trades** - Close to entry price exits
- **Extended Trades** - Multi-day position management

## 🎯 Quality Assurance

### Test Quality Standards
- **Isolated Testing** - Each test is independent
- **Deterministic Results** - Tests produce consistent results
- **Realistic Data** - Mock data resembles real market conditions
- **Comprehensive Coverage** - All code paths tested
- **Fast Execution** - Complete suite runs in under 60 seconds

### Continuous Integration
- Tests run automatically on code changes
- Performance regression detection
- Coverage reporting and trending
- Automated test result notifications

## 📋 Test Checklist

Before deploying the bot:
- [ ] All unit tests pass (100% success rate)
- [ ] Integration tests pass
- [ ] Backtesting validation completed
- [ ] Error handling verified
- [ ] Performance benchmarks met
- [ ] Configuration validation passed

## 🚨 Troubleshooting

### Common Issues
1. **Import Errors** - Ensure project root is in Python path
2. **Mock Data Issues** - Verify realistic data generation
3. **Timeout Errors** - Check test execution time limits
4. **Configuration Issues** - Validate test configuration setup

### Debug Mode
```bash
python run_tests.py --debug
python -m unittest test_regime_filter -v
```

## 📝 Adding New Tests

When adding new functionality:

1. **Create Test Module** - Follow naming convention `test_[component].py`
2. **Add Mock Data** - Create realistic test scenarios in `conftest.py`
3. **Test All Paths** - Cover success, failure, and edge cases
4. **Update Test Runner** - Add new module to `run_tests.py`
5. **Document Tests** - Add test descriptions and scenarios

The test suite ensures the One Good Trade Per Day bot maintains high quality and reliability standards before live deployment.