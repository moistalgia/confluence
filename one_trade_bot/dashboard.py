#!/usr/bin/env python3
"""
🚀 Crypto Trading Bot - Interactive Dashboard
============================================

Complete interface for running the enhanced paper trading system:
- Run daily scans with multi-pair analysis
- View transparency reports and scan breakdowns  
- Monitor trades and track performance
- Interactive menu system for easy operation

Usage:
    python dashboard.py
"""

import asyncio
import os
import sys
import yaml
from datetime import datetime, timedelta
import sqlite3
import json

# Add the bot directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.disciplined_trading_engine import DisciplinedTradingEngine
from core.transparency_dashboard import ScanningTransparencyDashboard, print_transparency_report

class TradingDashboard:
    """Interactive dashboard for the enhanced trading system"""
    
    def __init__(self):
        self.config = self.load_config()
        self.engine = None
        self.db_path = 'paper_trading.db'
        self.transparency = ScanningTransparencyDashboard(self.db_path)
        
    def load_config(self):
        """Load system configuration"""
        try:
            with open('config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            sys.exit(1)
    
    async def initialize_engine(self):
        """Initialize the trading engine"""
        if not self.engine:
            self.engine = DisciplinedTradingEngine(
                config=self.config.get('paper_trading', {}),
                data_provider=None,
                db_path=self.db_path
            )
            print("✅ Trading engine initialized")
    
    def print_header(self):
        """Print dashboard header"""
        print("\n" + "="*70)
        print("🚀 CRYPTO TRADING BOT - ENHANCED PAPER TRADING DASHBOARD")
        print("="*70)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show system status
        if self.engine:
            balance = self.engine.current_balance
            positions = len(self.engine.positions)
            trades = len(self.engine.completed_trades)
            print(f"💰 Balance: ${balance:,.2f} | 📊 Positions: {positions} | 📈 Trades: {trades}")
        print()
    
    def show_menu(self):
        """Display main menu options"""
        print("📋 MAIN MENU:")
        print("1. 🔍 Run Daily Scan (Multi-Pair Analysis)")
        print("2. 📊 View Transparency Report (Latest Scan)")
        print("3. 📈 View Trading Performance & Statistics")
        print("4. 🎯 View Current Target & Status")
        print("5. 📋 View All Scan History")
        print("6. 💻 Run Continuous Trading (Monitor Mode)")
        print("7. 🧪 Run System Test")
        print("8. ❌ Exit")
        print()
    
    async def run_daily_scan(self):
        """Run the daily scan with multi-pair analysis"""
        print("🌅 RUNNING DAILY SCAN...")
        print("-" * 50)
        
        await self.initialize_engine()
        
        # Run the scan
        await self.engine._run_daily_scan()
        
        # Show results
        if self.engine.daily_target:
            target = self.engine.daily_target
            print(f"\n🎯 TARGET SELECTED:")
            print(f"   Symbol: {target.symbol}")
            print(f"   Entry Zone: ${target.entry_low:.4f} - ${target.entry_high:.4f}")
            print(f"   Stop Loss: ${target.stop_loss:.4f}")
            print(f"   Take Profit: ${target.take_profit:.4f}")
            print(f"   Risk:Reward: {target.risk_reward:.2f}:1")
            print(f"   Confluence Score: {target.confluence_score}/100")
            print(f"   Expires: {target.expires_at.strftime('%H:%M')} ({(target.expires_at - datetime.now()).seconds//3600}h left)")
        else:
            print("\n📅 TODAY IS A REST DAY")
            print("   No trade candidates met all filter criteria")
            print("   Discipline: Quality over quantity")
    
    def view_transparency_report(self):
        """Show detailed transparency report"""
        print("📊 TRANSPARENCY REPORT - LATEST SCAN")
        print("-" * 50)
        
        try:
            print_transparency_report(self.db_path)
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            print("💡 Try running a daily scan first")
    
    def view_trading_performance(self):
        """Show trading performance and statistics"""
        print("📈 TRADING PERFORMANCE & STATISTICS")
        print("-" * 50)
        
        if not self.engine:
            print("❌ Engine not initialized. Run a scan first.")
            return
        
        try:
            stats = self.engine.get_discipline_stats()
            
            # Basic performance
            print(f"💰 Account Balance: ${self.engine.current_balance:,.2f}")
            print(f"📊 Total Trades: {len(self.engine.completed_trades)}")
            print(f"📍 Active Positions: {len(self.engine.positions)}")
            print()
            
            # Show recent trades
            if self.engine.completed_trades:
                print("📋 RECENT TRADES:")
                for i, trade in enumerate(self.engine.completed_trades[-5:], 1):
                    pnl = trade.exit_price - trade.entry_price if trade.direction.value == 'long' else trade.entry_price - trade.exit_price
                    pnl_pct = (pnl / trade.entry_price) * 100
                    print(f"   {i}. {trade.symbol} {trade.direction.value.upper()}: "
                          f"${trade.entry_price:.4f} → ${trade.exit_price:.4f} "
                          f"({pnl_pct:+.2f}%)")
            
            # Discipline stats
            if 'discipline_stats' in stats:
                discipline = stats['discipline_stats']
                print(f"\n🎯 DISCIPLINE METRICS:")
                print(f"   Trading Days: {discipline.get('total_trading_days', 0)}")
                print(f"   Rest Days: {discipline.get('rest_days', 0)} ({discipline.get('rest_day_pct', 0):.1f}%)")
                print(f"   Targets Selected: {discipline.get('targets_selected', 0)}")
                print(f"   Execution Rate: {discipline.get('execution_rate', 0):.1f}%")
                print(f"   Avg Confluence Score: {discipline.get('avg_confluence_score', 0):.1f}/100")
            
        except Exception as e:
            print(f"❌ Error getting performance stats: {e}")
    
    def view_current_status(self):
        """Show current target and system status"""
        print("🎯 CURRENT TARGET & SYSTEM STATUS")
        print("-" * 50)
        
        if not self.engine:
            print("❌ Engine not initialized. Run a scan first.")
            return
        
        status = self.engine.get_status()
        
        print(f"🔄 Execution State: {status['execution_state'].replace('_', ' ').title()}")
        print(f"📅 Trade Entered Today: {'Yes' if status['trade_entered_today'] else 'No'}")
        print(f"💰 Current Balance: ${status['current_balance']:,.2f}")
        print()
        
        if 'daily_target' in status and status['daily_target']:
            target = status['daily_target']
            print("🎯 ACTIVE TARGET:")
            print(f"   Symbol: {target['symbol']}")
            print(f"   Entry Zone: {target['entry_zone']}")
            print(f"   Score: {target['confluence_score']}/100")
            print(f"   Setup: {target['setup_type']}")
            print(f"   Time Remaining: {target['time_remaining_hours']:.1f} hours")
        else:
            print("❌ No active target")
    
    def view_scan_history(self):
        """Show history of all scans"""
        print("📋 SCAN HISTORY")
        print("-" * 50)
        
        try:
            recent_scans = self.transparency.get_recent_scans(10)
            
            if not recent_scans:
                print("❌ No scan history found")
                print("💡 Run a daily scan to generate data")
                return
            
            print(f"{'#':<3} {'Date/Time':<17} {'Pairs':<6} {'Valid':<6} {'Best Symbol':<12} {'Score'}")
            print("-" * 65)
            
            for i, scan in enumerate(recent_scans, 1):
                timestamp = scan['timestamp'][:16].replace('T', ' ')
                best_symbol = scan['best_symbol'] or 'None'
                best_score = scan['best_score'] or 0
                
                print(f"{i:<3} {timestamp:<17} {scan['pairs_analyzed']:<6} "
                      f"{scan['valid_setups']:<6} {best_symbol:<12} {best_score}/100")
                      
        except Exception as e:
            print(f"❌ Error getting scan history: {e}")
    
    async def run_continuous_trading(self):
        """Run continuous trading with monitoring"""
        print("💻 STARTING CONTINUOUS TRADING MODE...")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        await self.initialize_engine()
        
        try:
            # Run the disciplined cycle
            await self.engine.run_disciplined_cycle()
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        except Exception as e:
            print(f"❌ Trading error: {e}")
    
    async def run_system_test(self):
        """Run comprehensive system test"""
        print("🧪 RUNNING SYSTEM TEST...")
        print("-" * 50)
        
        try:
            # Import and run the complete system test
            exec(open('test_complete_system.py').read())
        except Exception as e:
            print(f"❌ Test error: {e}")
    
    async def main_loop(self):
        """Main interactive loop"""
        while True:
            self.print_header()
            self.show_menu()
            
            try:
                choice = input("👉 Select option (1-8): ").strip()
                print()
                
                if choice == '1':
                    await self.run_daily_scan()
                elif choice == '2':
                    self.view_transparency_report()
                elif choice == '3':
                    self.view_trading_performance()
                elif choice == '4':
                    self.view_current_status()
                elif choice == '5':
                    self.view_scan_history()
                elif choice == '6':
                    await self.run_continuous_trading()
                elif choice == '7':
                    await self.run_system_test()
                elif choice == '8':
                    print("👋 Goodbye! Happy trading!")
                    break
                else:
                    print("❌ Invalid option. Please select 1-8.")
                
                if choice != '8':
                    input("\n📱 Press Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Happy trading!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("\n📱 Press Enter to continue...")

async def main():
    """Main entry point"""
    dashboard = TradingDashboard()
    await dashboard.main_loop()

if __name__ == "__main__":
    asyncio.run(main())