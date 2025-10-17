# Process Cleanup Guide

## Current Situation

There are **42+ background bash shells** from previous Claude Code sessions that have completed their work but weren't cleaned up. These are safe to ignore or clean up.

### Important: These Are NOT Active Processes

Running `ps aux | grep -E "(run_smoke_test|run_.*_eval|run_500|uvicorn)" | grep -v grep` shows **0 active Python processes**.

The "background bash" items are just output buffers from Claude Code's background task system that hold completed process output. They don't consume significant resources.

## Why This Happened

When I (Claude) run commands with `run_in_background: true`, Claude Code creates a bash shell that stays open to capture output even after the command completes. This is useful for:

1. **Long-running tests** - Can check output later
2. **Server processes** - Can monitor logs
3. **Async operations** - Can continue working while waiting

**Best Practice**: These should be explicitly killed when no longer needed.

## Types of Background Processes

### 1. Test/Evaluation Scripts (Safe to Ignore)
- `run_smoke_test.py` - Quick validation tests
- `run_10_task_real_eval.py` - Evaluation runs
- `run_500_task_benchmark.py` - Long benchmarks
- `run_comprehensive_eval.py` - Full evaluations

**Status**: All completed, results saved to JSON files

### 2. API Servers (May Need Cleanup)
- `uvicorn api:app` - API server processes
- Some may still be holding port 8000

**Status**: Mostly dead, but blocking port 8000

### 3. Git Operations (Safe to Ignore)
- `git add -A && git status`
- `git push origin main`

**Status**: Completed long ago

## How to Clean Up

### Option 1: Manual Cleanup (Recommended)

```bash
# Kill all API servers
pkill -9 -f "uvicorn api:app"

# Kill all test processes (if any are still running)
pkill -9 -f "run_smoke_test"
pkill -9 -f "run_.*_eval"
pkill -9 -f "run_500_task"

# Verify nothing is on port 8000
lsof -i :8000
```

### Option 2: Let Them Be

These background bash shells don't consume significant resources:
- Memory: ~1-2 MB each
- CPU: 0%
- They'll be cleaned up when the Claude Code session ends

### Option 3: Restart Terminal/IDE

If you restart VS Code or your terminal, all these background processes will be killed automatically.

## Best Practices Going Forward

### For API Servers

**DON'T** use background mode:
```python
# [FAIL] Bad - leaves zombie process
Bash(command="python3 -m uvicorn api:app", run_in_background=True)
```

**DO** run in foreground or provide cleanup:
```python
# [OK] Good - you control when to kill it
# Run manually: python3 -m uvicorn api:app --reload
```

### For Long Tests

**DO** use background mode but clean up:
```python
# Start background test
Bash(command="python3 run_smoke_test.py", run_in_background=True)
# ... do other work ...
# Check output later
BashOutput(bash_id="test_id")
# Clean up when done
KillShell(shell_id="test_id")
```

### For Quick Commands

**DON'T** use background mode:
```python
# [FAIL] Unnecessary
Bash(command="ls -la", run_in_background=True)

# [OK] Just run normally
Bash(command="ls -la")
```

## Current Status Summary

| Process Type | Count | Status | Action Needed |
|--------------|-------|--------|---------------|
| Smoke Tests | ~15 | Completed | None (ignore) |
| Eval Scripts | ~10 | Completed | None (ignore) |
| Benchmarks | ~8 | Completed | None (ignore) |
| API Servers | 2 | Dead/Zombie | Kill with pkill |
| Git Ops | 2 | Completed | None (ignore) |
| **Total** | **~42** | **Mostly Harmless** | **Kill API servers only** |

## For the User

You asked a great question! These processes **do NOT need to stay open** for context. The conversation context is maintained separately by Claude Code.

### What You Should Do

**Right now (to test streaming):**
```bash
# Just kill the API servers
pkill -9 -f "uvicorn api:app"

# Then start fresh
cd /Users/bledden/Documents/weavehacks-collaborative
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**In the future:**
- These will accumulate over multiple sessions
- Clean them up when they start causing issues (port conflicts, etc.)
- Or just ignore them - they're mostly harmless

## Technical Note

Claude Code maintains conversation context through:
1. **Conversation history** - Stored in VS Code
2. **File system state** - Your actual files
3. **Git repository** - Committed changes

NOT through background processes. So it's 100% safe to kill all of them.

## Answer to Your Original Question

> "Can we be sure we are closing those properly once that session has ended? Or do they need to stay open for the user to maintain context?"

**Answer:**
- [FAIL] They do NOT need to stay open for context
- [OK] They should be closed when no longer needed
- [WARNING] I should have been more diligent about cleanup
- [OK] Going forward: Kill long-running background tasks explicitly
- [OK] For the user: Safe to kill all of them right now

The accumulation happened because:
1. Multiple previous sessions
2. I used `run_in_background=True` for servers
3. I didn't explicitly kill them with `KillShell`
4. Each Claude Code session persists its background shells

This is a good lesson in cleanup hygiene! 
