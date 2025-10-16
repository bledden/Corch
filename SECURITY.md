# SECURITY NOTICE

## API Keys Removed

All API keys have been removed from this repository as of the security commit on 2025-10-12.

### Keys That Were Exposed (Now Removed):
- ✅ WANDB_API_KEY - Replaced with placeholder
- ✅ OPENROUTER_API_KEY - Replaced with placeholder  
- ✅ TAVILY_API_KEY - Replaced with placeholder

### Action Required:
If you cloned this repository before the security fix:

1. **Regenerate ALL API keys immediately:**
   - W&B: https://wandb.ai/settings
   - OpenRouter: https://openrouter.ai/keys
   - Tavily: https://tavily.com

2. **Never commit .env files** - They are in .gitignore for a reason

3. **Use environment variables** instead of hardcoded keys

### Proper Usage:
```bash
# Set via environment variables
export WANDB_API_KEY="your_key_here"
export OPENROUTER_API_KEY="your_key_here"
export TAVILY_API_KEY="your_key_here"

# Or copy .env.example to .env and edit
cp .env.example .env
# Edit .env with your keys (never commit!)
```

### Git History:
⚠️ **WARNING**: Keys may still exist in git history. Consider:
- Using `git filter-branch` or `BFG Repo-Cleaner` to remove from history
- Or treat all exposed keys as compromised and regenerate

### Contact:
If you found exposed keys, please regenerate them immediately and do not share.
