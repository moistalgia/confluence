#!/usr/bin/env python3
"""
Simple test to check if database has scan data
"""

import sqlite3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_database(db_path):
    print(f"🔍 Checking database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database file doesn't exist: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Check scan_results table
        cursor = conn.execute("SELECT COUNT(*) FROM scan_results")
        scan_count = cursor.fetchone()[0]
        print(f"📊 Scan results: {scan_count} records")
        
        if scan_count > 0:
            # Get latest scan
            cursor = conn.execute("""
                SELECT id, scan_timestamp, pairs_analyzed, best_setup_symbol, best_setup_score 
                FROM scan_results 
                ORDER BY scan_timestamp DESC 
                LIMIT 1
            """)
            latest = cursor.fetchone()
            print(f"   Latest scan: ID={latest[0]}, Time={latest[1]}, Pairs={latest[2]}, Best={latest[3]} ({latest[4]}/100)")
            
            # Check pair analysis
            cursor = conn.execute("SELECT COUNT(*) FROM pair_analysis WHERE scan_id = ?", (latest[0],))
            pair_count = cursor.fetchone()[0]
            print(f"   Pair analyses: {pair_count} records")
            
            if pair_count > 0:
                # Show top 3 pairs
                cursor = conn.execute("""
                    SELECT symbol, confluence_score, setup_valid 
                    FROM pair_analysis 
                    WHERE scan_id = ? 
                    ORDER BY confluence_score DESC 
                    LIMIT 3
                """, (latest[0],))
                top_pairs = cursor.fetchall()
                print("   Top pairs:")
                for pair in top_pairs:
                    status = "✅" if pair[2] else "❌"
                    print(f"     {pair[0]}: {pair[1]}/100 {status}")
        else:
            print("❌ No scan results found in database")
            
    except Exception as e:
        print(f"❌ Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Check test databases
    databases = [
        'enhanced_system_test.db',
        'transparency_test.db', 
        'paper_trading.db',
        'debug_transparency.db'
    ]
    
    for db in databases:
        if os.path.exists(db):
            check_database(db)
            print()