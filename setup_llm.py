#!/usr/bin/env python3
"""
Setup script for LLM Trading Analyst
Helps configure API keys and test the system
"""

import os
import sys
from pathlib import Path

def create_config_file():
    """Create config file from template"""
    
    config_path = Path("config.env")
    template_path = Path("config.env.template")
    
    if config_path.exists():
        print(f"✅ Config file already exists: {config_path}")
        return
    
    if not template_path.exists():
        print(f"❌ Template file missing: {template_path}")
        return
    
    # Copy template to config
    with open(template_path, 'r') as template:
        content = template.read()
    
    with open(config_path, 'w') as config:
        config.write(content)
    
    print(f"✅ Created config file: {config_path}")
    print(f"📝 Please edit {config_path} and add your API keys")

def test_api_keys():
    """Test API key configuration"""
    
    config_path = Path("config.env")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print(f"Run create_config_file() first")
        return False
    
    # Load config
    openai_key = None
    anthropic_key = None
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('OPENAI_API_KEY=') and not line.endswith('your_openai_api_key_here'):
                openai_key = line.split('=', 1)[1].strip()
            elif line.startswith('ANTHROPIC_API_KEY=') and not line.endswith('your_anthropic_api_key_here'):
                anthropic_key = line.split('=', 1)[1].strip()
    
    print("🔑 API Key Status:")
    print(f"  OpenAI: {'✅' if openai_key else '❌ Not configured'}")
    print(f"  Anthropic: {'✅' if anthropic_key else '❌ Not configured'}")
    
    # Test connections
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            # Simple test call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            print(f"  OpenAI Connection: ✅ Working")
        except Exception as e:
            print(f"  OpenAI Connection: ❌ Error - {e}")
    
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            # Simple test call
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=5,
                messages=[{"role": "user", "content": "Test"}]
            )
            print(f"  Anthropic Connection: ✅ Working")
        except Exception as e:
            print(f"  Anthropic Connection: ❌ Error - {e}")
    
    return bool(openai_key or anthropic_key)

def show_usage_examples():
    """Show usage examples"""
    
    print("\n🚀 Usage Examples:")
    print()
    print("1. Process single prompt file:")
    print("   from llm_trading_analyst import LLMTradingAnalyst")
    print("   analyst = LLMTradingAnalyst()")
    print("   result = analyst.process_analysis_file('output/ultimate_analysis/enhanced_prompt_BTC_USDT_xxx.txt')")
    print()
    print("2. Batch process all prompts with Claude Sonnet (90% cost reduction):")
    print("   analyst.batch_process_prompts(model='claude-sonnet')")
    print()
    print("3. Use GPT-4 instead:")
    print("   analyst.batch_process_prompts(model='gpt-4')")
    print()
    print("4. Process specific directory:")
    print("   analyst.batch_process_prompts('output/ultimate_analysis', model='claude-sonnet')")
    print()
    print("💡 Claude models support prompt caching for 90% cost reduction!")
    print("💡 Responses are automatically saved to output/llm_responses/")

def main():
    """Setup wizard"""
    
    print("🤖 LLM Trading Analyst Setup")
    print("=" * 50)
    
    # Step 1: Create config file
    print("\n📝 Step 1: Configuration")
    create_config_file()
    
    # Step 2: Test API keys
    print("\n🔑 Step 2: API Key Validation")
    if test_api_keys():
        print("✅ At least one API key is configured")
    else:
        print("❌ No API keys configured")
        print("Edit config.env and add your API keys:")
        print("  - OpenAI: https://platform.openai.com/api-keys") 
        print("  - Anthropic: https://console.anthropic.com/")
        return
    
    # Step 3: Check for prompt files
    print("\n📄 Step 3: Prompt Files")
    prompt_dir = Path("output/ultimate_analysis")
    if prompt_dir.exists():
        prompt_files = list(prompt_dir.glob("enhanced_prompt_*.txt"))
        print(f"✅ Found {len(prompt_files)} prompt files ready for processing")
    else:
        print("❌ No prompt files found")
        print("Run ultimate_crypto_analyzer.py first to generate enhanced prompts")
        return
    
    # Step 4: Show usage
    show_usage_examples()
    
    print("\n🎉 Setup complete! Ready to analyze with AI.")

if __name__ == "__main__":
    main()