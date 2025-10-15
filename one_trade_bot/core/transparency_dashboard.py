"""
Scanning Transparency Dashboard
==============================

Provides detailed transparency into multi-pair scanning decisions:
- Logs all pair analysis results to database
- Shows filter-by-filter breakdown for each pair
- Explains ranking rationale and final selection
- Generates human-readable decision reports

This ensures users can see exactly why each trading decision was made.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import pandas as pd

from core.multi_pair_kraken_scanner import PairAnalysis

logger = logging.getLogger(__name__)

class ScanningTransparencyDashboard:
    """
    Comprehensive transparency system for scanning decisions
    
    Features:
    - Database storage of all scan results
    - Detailed filter breakdown logging
    - Ranking rationale explanation
    - Decision audit trail
    - Human-readable reports
    """
    
    def __init__(self, db_path: str = 'paper_trading.db'):
        self.db_path = db_path
        self._setup_transparency_tables()
    
    def _setup_transparency_tables(self):
        """Create database tables for transparency tracking"""
        conn = sqlite3.connect(self.db_path)
        
        # Main scan results table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_timestamp TEXT NOT NULL,
                scan_duration_seconds REAL,
                pairs_analyzed INTEGER,
                liquid_pairs INTEGER,
                valid_setups INTEGER,
                best_setup_symbol TEXT,
                best_setup_score INTEGER,
                selection_criteria TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Detailed pair analysis table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS pair_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                symbol TEXT NOT NULL,
                price REAL,
                volume_24h REAL,
                spread_pct REAL,
                is_liquid BOOLEAN,
                confluence_score INTEGER,
                setup_valid BOOLEAN,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                risk_reward REAL,
                setup_type TEXT,
                rejection_reason TEXT,
                filter_results TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (scan_id) REFERENCES scan_results (id)
            )
        ''')
        
        # Filter breakdown table for detailed analysis
        conn.execute('''
            CREATE TABLE IF NOT EXISTS filter_breakdown (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_analysis_id INTEGER,
                filter_name TEXT NOT NULL,
                filter_score INTEGER,
                filter_passed BOOLEAN,
                filter_details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pair_analysis_id) REFERENCES pair_analysis (id)
            )
        ''')
        
        # Decision rationale table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS decision_rationale (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                decision_type TEXT NOT NULL,
                winner_symbol TEXT,
                winner_score INTEGER,
                runner_up_symbol TEXT,
                runner_up_score TEXT,
                score_difference INTEGER,
                rationale_text TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (scan_id) REFERENCES scan_results (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Transparency dashboard database tables ready")
    
    def log_scan_results(self, scan_results: Dict[str, Any]) -> int:
        """
        Log complete scan results to database for transparency
        
        Returns scan_id for linking detailed results
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Insert main scan record
            cursor = conn.execute('''
                INSERT INTO scan_results (
                    scan_timestamp, scan_duration_seconds, pairs_analyzed,
                    liquid_pairs, valid_setups, best_setup_symbol,
                    best_setup_score, selection_criteria
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scan_results['scan_timestamp'].isoformat(),
                scan_results.get('scan_duration_seconds', 0),
                scan_results['pairs_analyzed'],
                scan_results['liquid_pairs'],
                len([r for r in scan_results['rankings'] if r.setup_valid]),
                scan_results['best_setup']['symbol'] if scan_results['best_setup'] else None,
                scan_results['best_setup']['confluence_score'] if scan_results['best_setup'] else None,
                json.dumps(scan_results.get('transparency_report', {}).get('selection_criteria', {}))
            ))
            
            scan_id = cursor.lastrowid
            
            # Log all pair analysis results
            for pair_result in scan_results['rankings']:
                pair_analysis_id = self._log_pair_analysis(conn, scan_id, pair_result)
                self._log_filter_breakdown(conn, pair_analysis_id, pair_result)
            
            # Log decision rationale
            self._log_decision_rationale(conn, scan_id, scan_results.get('transparency_report', {}))
            
            conn.commit()
            logger.info(f"📊 Scan results logged to transparency database (scan_id: {scan_id})")
            return scan_id
            
        except Exception as e:
            logger.error(f"Error logging scan results: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()
    
    def _log_pair_analysis(self, conn: sqlite3.Connection, scan_id: int, pair_result: PairAnalysis) -> int:
        """Log individual pair analysis results"""
        cursor = conn.execute('''
            INSERT INTO pair_analysis (
                scan_id, symbol, price, volume_24h, spread_pct, is_liquid,
                confluence_score, setup_valid, entry_price, stop_loss,
                take_profit, risk_reward, setup_type, rejection_reason, filter_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scan_id, pair_result.symbol, pair_result.price, pair_result.volume_24h,
            pair_result.spread_pct, pair_result.is_liquid, pair_result.confluence_score,
            pair_result.setup_valid, pair_result.entry_price, pair_result.stop_loss,
            pair_result.take_profit, pair_result.risk_reward, pair_result.setup_type,
            pair_result.rejection_reason, json.dumps(pair_result.filter_results)
        ))
        
        return cursor.lastrowid
    
    def _log_filter_breakdown(self, conn: sqlite3.Connection, pair_analysis_id: int, pair_result: PairAnalysis):
        """Log detailed filter breakdown for each pair"""
        for filter_name, filter_data in pair_result.filter_results.items():
            filter_score = filter_data.get('score', 0)
            filter_passed = filter_score > 15  # Assuming >15 is a pass
            
            conn.execute('''
                INSERT INTO filter_breakdown (
                    pair_analysis_id, filter_name, filter_score, filter_passed, filter_details
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                pair_analysis_id, filter_name, filter_score, filter_passed,
                json.dumps(filter_data)
            ))
    
    def _log_decision_rationale(self, conn: sqlite3.Connection, scan_id: int, transparency_report: Dict[str, Any]):
        """Log the reasoning behind the final selection"""
        rationale = transparency_report.get('ranking_rationale', {})
        
        if rationale:
            runner_up = rationale.get('runner_up', {})
            
            conn.execute('''
                INSERT INTO decision_rationale (
                    scan_id, decision_type, winner_symbol, winner_score,
                    runner_up_symbol, runner_up_score, score_difference, rationale_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scan_id, 'DAILY_SELECTION', rationale.get('winner'),
                rationale.get('winning_score'), runner_up.get('symbol'),
                runner_up.get('score'), runner_up.get('score_difference'),
                rationale.get('why_selected', '')
            ))
    
    def generate_transparency_report(self, scan_id: Optional[int] = None) -> str:
        """
        Generate human-readable transparency report
        
        If scan_id is None, generates report for most recent scan
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get scan information
            if scan_id is None:
                scan_data = conn.execute('''
                    SELECT * FROM scan_results ORDER BY scan_timestamp DESC LIMIT 1
                ''').fetchone()
            else:
                scan_data = conn.execute('''
                    SELECT * FROM scan_results WHERE id = ?
                ''', (scan_id,)).fetchone()
            
            if not scan_data:
                return "❌ No scan data found"
            
            scan_id = scan_data[0]
            
            # Build comprehensive report
            report = []
            report.append("📊 SCANNING TRANSPARENCY REPORT")
            report.append("=" * 50)
            report.append(f"Scan Time: {scan_data[1]}")
            report.append(f"Duration: {scan_data[2]:.1f}s")
            report.append(f"Pairs Analyzed: {scan_data[3]}")
            report.append(f"Liquid Pairs: {scan_data[4]}")
            report.append(f"Valid Setups: {scan_data[5]}")
            report.append("")
            
            # Get decision rationale
            rationale = conn.execute('''
                SELECT * FROM decision_rationale WHERE scan_id = ? ORDER BY created_at DESC LIMIT 1
            ''', (scan_id,)).fetchone()
            
            if rationale:
                report.append("🎯 FINAL SELECTION")
                report.append("-" * 30)
                report.append(f"Winner: {rationale[2]} (Score: {rationale[3]}/100)")
                if rationale[4]:  # Has runner-up
                    report.append(f"Runner-up: {rationale[4]} (Score: {rationale[5]})")
                    report.append(f"Score Difference: {rationale[6]} points")
                report.append(f"Rationale: {rationale[7]}")
                report.append("")
            
            # Get detailed pair breakdown
            pair_data = conn.execute('''
                SELECT symbol, confluence_score, setup_valid, rejection_reason, is_liquid, volume_24h
                FROM pair_analysis WHERE scan_id = ?
                ORDER BY confluence_score DESC
            ''', (scan_id,)).fetchall()
            
            report.append("📈 PAIR ANALYSIS BREAKDOWN")
            report.append("-" * 30)
            
            for pair in pair_data:
                symbol, score, valid, reason, liquid, volume = pair
                status = "✅ VALID" if valid else "❌ REJECTED"
                liquidity = "💧 LIQUID" if liquid else "🚫 ILLIQUID"
                
                report.append(f"{symbol:12} | Score: {score:3}/100 | {status} | {liquidity} | Vol: ${volume:,.0f}")
                
                # Get detailed filter breakdown for this pair
                pair_id = conn.execute('''
                    SELECT id FROM pair_analysis 
                    WHERE scan_id = ? AND symbol = ?
                ''', (scan_id, symbol)).fetchone()
                
                if pair_id:
                    filter_details = conn.execute('''
                        SELECT filter_name, filter_score, filter_passed
                        FROM filter_breakdown 
                        WHERE pair_analysis_id = ?
                        ORDER BY filter_score DESC
                    ''', (pair_id[0],)).fetchall()
                    
                    if filter_details:
                        filter_line = "             └─ Filters: "
                        filter_components = []
                        for fname, fscore, fpassed in filter_details:
                            status_icon = "✅" if fpassed else "❌"
                            filter_components.append(f"{fname}:{fscore:2.0f}{status_icon}")
                        report.append(filter_line + " | ".join(filter_components))
                
                if reason:
                    report.append(f"             └─ Rejection: {reason}")
            
            report.append("")
            
            # Get filter performance summary
            filter_stats = conn.execute('''
                SELECT filter_name, AVG(filter_score) as avg_score, 
                       SUM(CASE WHEN filter_passed THEN 1 ELSE 0 END) as passed_count,
                       COUNT(*) as total_count
                FROM filter_breakdown fb
                JOIN pair_analysis pa ON fb.pair_analysis_id = pa.id
                WHERE pa.scan_id = ?
                GROUP BY filter_name
                ORDER BY avg_score DESC
            ''', (scan_id,)).fetchall()
            
            report.append("🔍 FILTER PERFORMANCE")
            report.append("-" * 30)
            for filter_name, avg_score, passed, total in filter_stats:
                pass_rate = (passed / total * 100) if total > 0 else 0
                report.append(f"{filter_name:20} | Avg Score: {avg_score:5.1f} | Pass Rate: {pass_rate:5.1f}% ({passed}/{total})")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating transparency report: {e}")
            return f"❌ Error generating report: {e}"
        finally:
            conn.close()
    
    def get_recent_scans(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recent scans with summary information"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            scans = conn.execute('''
                SELECT id, scan_timestamp, pairs_analyzed, valid_setups,
                       best_setup_symbol, best_setup_score
                FROM scan_results
                ORDER BY scan_timestamp DESC
                LIMIT ?
            ''', (limit,)).fetchall()
            
            result = []
            for scan in scans:
                result.append({
                    'scan_id': scan[0],
                    'timestamp': scan[1],
                    'pairs_analyzed': scan[2],
                    'valid_setups': scan[3],
                    'best_symbol': scan[4],
                    'best_score': scan[5]
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting recent scans: {e}")
            return []
        finally:
            conn.close()
    
    def export_scan_data(self, scan_id: int, format: str = 'csv') -> str:
        """Export scan data for external analysis"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get pair analysis data
            df = pd.read_sql('''
                SELECT pa.*, sr.scan_timestamp
                FROM pair_analysis pa
                JOIN scan_results sr ON pa.scan_id = sr.id
                WHERE pa.scan_id = ?
                ORDER BY pa.confluence_score DESC
            ''', conn, params=(scan_id,))
            
            if format.lower() == 'csv':
                filename = f'scan_export_{scan_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                df.to_csv(filename, index=False)
                return filename
            else:
                return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error exporting scan data: {e}")
            return ""
        finally:
            conn.close()

def print_transparency_report(db_path: str = 'paper_trading.db'):
    """Utility function to print the latest transparency report"""
    dashboard = ScanningTransparencyDashboard(db_path)
    report = dashboard.generate_transparency_report()
    print(report)