import sqlite3
import os

db_path = 'paper_trading.db'
print(f'🔍 Checking database: {db_path}')
print(f'📁 File exists: {os.path.exists(db_path)}')

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    
    # Check if transparency tables exist
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f'📊 Tables in database: {[t[0] for t in tables]}')
    
    # Check scan_results if it exists
    if 'scan_results' in [t[0] for t in tables]:
        count = conn.execute('SELECT COUNT(*) FROM scan_results').fetchone()[0]
        print(f'📈 Scan results count: {count}')
        
        if count > 0:
            latest = conn.execute('SELECT * FROM scan_results ORDER BY scan_timestamp DESC LIMIT 1').fetchone()
            print(f'🕐 Latest scan: {latest}')
        else:
            print('❌ No scan results in table')
    else:
        print('❌ scan_results table does not exist')
    
    # Check pair_analysis if it exists  
    if 'pair_analysis' in [t[0] for t in tables]:
        count = conn.execute('SELECT COUNT(*) FROM pair_analysis').fetchone()[0]
        print(f'📈 Pair analysis count: {count}')
        
        if count > 0:
            latest = conn.execute('SELECT pair, confluence_score FROM pair_analysis ORDER BY timestamp DESC LIMIT 5').fetchall()
            print(f'🔍 Latest pair analyses: {latest}')
    else:
        print('❌ pair_analysis table does not exist')
    
    conn.close()
else:
    print('❌ Database file does not exist')