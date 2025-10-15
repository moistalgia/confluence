# One Good Trade Per Day Bot - Clean Architecture 🎯

**Disciplined crypto trading with ONE rule: Wait patiently for high-probability setups**

## 🏗️ **Clean Architecture (Post-Cleanup)**

### **Single Complete System**
```
core/disciplined_trading_engine.py  ⭐ THE COMPLETE SYSTEM
├── Paper trading simulation (realistic order execution)
├── SQLite database (trade journaling & performance tracking)  
├── Daily discipline (8AM scan → ONE target → patient entry)
├── Live market data integration (CCXT)
└── Professional analytics & reporting
```

### **Supporting Infrastructure**  
```
core/paper_trading.py       # Base classes & order simulation
core/data_provider.py       # Market data interface
core/workflow_engine.py     # 5-filter scanning pipeline
filters/                    # Signal generation logic
```

## 🚀 **Usage**

### **Quick Test (Simulated Data)**
```bash
# Run comprehensive system test
python main_simple.py --test

# Live paper trading with simulated data (BTC ~$63.5k)
python main_simple.py --live --duration 4
```

### **Live Data (Real Market Prices)** 
```bash
# Enable live market data (BTC ~$110k)
python main_simple.py --live --live-data --duration 24
```

## ⚙️ **Configuration**

Edit `config.yaml` for all settings:
```yaml
# Enable live market data
use_live_market_data: true  # false = simulated, true = real prices

# Trading discipline  
trading:
  max_risk_per_trade: 0.01  # 1% risk per trade
  max_positions: 1          # ONE trade rule enforced

# Timing discipline
schedule:
  scan_time: "08:00"        # Daily scan time

execution:
  entry_timeout_hours: 6    # Entry window timeout
  monitor_interval: 1       # Check frequency (minutes)
```

## 🎯 **Core Trading Discipline**

**UNCHANGED since original system:**

1. **Daily Scan (8AM)**: Run 5-filter pipeline to pick THE ONE candidate
2. **Patient Entry**: Wait up to 6 hours for optimal entry in target zone  
3. **ONE Trade Rule**: Maximum one position per 24 hours
4. **Risk Management**: 1% risk per trade, 2:1+ reward ratio required
5. **No Chasing**: If entry zone missed, wait for tomorrow

## 📊 **What Was Removed**

### **Obsolete Files (Cleaned Up)**
- ❌ `enhanced_paper_trading.py` → Integrated into disciplined engine
- ❌ `hybrid_execution_engine.py` → Integrated into disciplined engine  
- ❌ `paper_trading_workflow.py` → Replaced by direct engine usage
- ❌ `paper_trading_cli.py` → Replaced by `main_simple.py`
- ❌ `enhanced_paper_trading_cli.py` → Replaced by `main_simple.py`
- ❌ Multiple analysis docs → Moved to `archive/` folder

### **Result: Single Engine**
No more confusion about which engine to use. `DisciplinedTradingEngine` is the complete system with:
- ✅ All enhanced paper trading features
- ✅ All hybrid timing discipline  
- ✅ Database integration & analytics
- ✅ Live market data capability

## 🔧 **Development**

### **File Structure**
```
one_trade_bot/
├── main_simple.py              # 🎯 Simple entry point
├── test_disciplined_engine.py  # 🧪 Comprehensive system test
├── config.yaml                 # ⚙️ All configuration
├── core/
│   └── disciplined_trading_engine.py  # ⭐ Complete system
├── filters/                    # Signal generation
├── tests/                      # Test infrastructure  
├── backtest/                   # Historical validation
└── archive/                    # Historical analysis docs
```

### **Key Benefits**
1. **No Confusion**: One engine, one entry point, one test file
2. **Complete Integration**: Database + execution + timing discipline
3. **Ready for Live**: Real market data integration available  
4. **Maintainable**: Single codebase instead of multiple half-systems
5. **Proven Working**: Comprehensive testing validates all functionality

## 📈 **Next Steps**

1. **Test with Live Data**: `python main_simple.py --live --live-data --duration 24`
2. **Multi-Day Validation**: Run extended paper trading sessions
3. **Performance Analysis**: Use database to analyze setup quality over time
4. **Live Trading**: Connect to real exchange when ready

---

**Remember**: The system maintains the core discipline that made it profitable - patience and selectivity over frequency and complexity.