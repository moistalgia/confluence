#!/usr/bin/env python3
"""
Quick script to fix the indentation issue in real_kraken_paper_trading.py
"""

import re

# Read the file
with open('real_kraken_paper_trading.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the problematic section
old_pattern = r'''            # STRICT DATA INTEGRITY POLICY - NO FALLBACKS ALLOWED
            try:
            
            # .* FETCH REAL VOLUME DATA
            real_volume_data = await self\._fetch_current_volume_data\(pair\)'''

new_code = '''            # STRICT DATA INTEGRITY POLICY - NO FALLBACKS ALLOWED
            try:
                # 📊 FETCH REAL VOLUME DATA
                real_volume_data = await self._fetch_current_volume_data(pair)
                logger.info(f"✅ Real volume data acquired for {pair}")
            except ValueError as e:
                logger.error(f"🚨 SKIPPING ANALYSIS for {pair} - No real volume data available")
                logger.error(f"🚨 Error: {str(e)}")
                logger.warning(f"⚠️ Analysis requires authentic market data - refusing to proceed with estimates")
                return None  # Skip this pair entirely'''

# Replace using regex
content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE)

# Write back
with open('real_kraken_paper_trading.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed volume data strict policy")