# Test Monitoring Dashboard

Real-time CLI dashboard for monitoring Facilitair test progress with a beautiful interface showing live updates of task completion, agent activity, and results.

## Quick Start

### Option 1: Run Tests with Built-in Monitoring (Recommended)

```bash
# Terminal 1: Run tests and write logs
./run_with_monitoring.sh BALANCED fe6e5e &
./run_with_monitoring.sh OPEN 8c42af &
./run_with_monitoring.sh CLOSED d9ac2f &

# Terminal 2: Watch the dashboard
python3 monitor.py fe6e5e 8c42af d9ac2f
```

### Option 2: Monitor Existing Tests

If you already have tests running in the background:

```bash
# The monitor will display progress from log files
python3 monitor.py fe6e5e 8c42af d9ac2f
```

## Dashboard Features

The live dashboard displays:

| Feature | Description |
|---------|-------------|
| **Test ID** | Short identifier for each running test |
| **Strategy** | Model selection strategy (BALANCED/OPEN/CLOSED) |
| **Progress** | Visual progress bar showing current task / total tasks |
| **Agent** | Currently active agent (architect, coder, reviewer, etc.) |
| **Sequential** | Pass rate for sequential collaboration method |
| **Baseline** | Pass rate for baseline single-agent method |
| **Status** | Running 🏃, Completed ✅, or Error ❌ |

## Example Dashboard Output

```
╭─────────────────── 🚀 Facilitair Test Monitor Dashboard ───────────────────╮
│                                                                              │
│  Test    Strategy   Progress                     Agent     Sequential   ...  │
│  ──────────────────────────────────────────────────────────────────────────  │
│  fe6e5e  BALANCED   ████████████░░░░░░░░ 6/10    coder     Pending...  ...  │
│  8c42af  OPEN       ███████████████░░░░░ 8/10    reviewer  4/8 (50%)   ...  │
│  d9ac2f  CLOSED     ████████████████████ 10/10   done      ✅ 7/10 (70%)...  │
│                                                                              │
│  Legend: 🏃 Running  ✅ Completed  ❌ Error                                  │
│  Press Ctrl+C to exit                                                        │
╰─────────────────────────────────────── 17:45:23 | Monitoring 3 test(s) ─────╯
```

## How It Works

1. **Test Execution**: Tests write their output to `/tmp/facilitair_logs/test_<bash_id>.log`
2. **Monitor Parsing**: The dashboard reads these log files every 2 seconds
3. **Live Display**: Uses `rich` library to create a beautiful live-updating TUI
4. **Auto-Detection**: Automatically detects when tests complete

## Files

- **[monitor.py](monitor.py)**: Main dashboard application with live TUI
- **[run_with_monitoring.sh](run_with_monitoring.sh)**: Wrapper script to run tests with logging
- **[setup_monitoring.py](setup_monitoring.py)**: Helper to bridge bash outputs to log files
- **[monitor_tests.py](monitor_tests.py)**: Alternative monitoring implementation

## Advanced Usage

### Custom Refresh Rate

```bash
# Edit monitor.py and change refresh_interval
python3 monitor.py fe6e5e 8c42af d9ac2f
```

### Monitor Single Test

```bash
python3 monitor.py fe6e5e
```

### Check Log Files Manually

```bash
# View raw log output
tail -f /tmp/facilitair_logs/test_fe6e5e.log
```

## Troubleshooting

### Dashboard shows "Loading..." for all tests

- **Cause**: Log files haven't been created yet
- **Solution**: Make sure tests are running with logging enabled using `run_with_monitoring.sh`

### No progress updates

- **Cause**: Tests may have failed or completed
- **Solution**: Check the log files manually or verify tests are still running with `ps aux | grep run_smoke_test`

### Dashboard freezes

- **Press Ctrl+C** to exit and restart
- Check that log directory `/tmp/facilitair_logs/` exists and is writable

## Integration with Facilitair

This monitoring system integrates seamlessly with:

- `run_smoke_test.py` - 10-task smoke tests
- `run_10_task_real_eval.py` - Real evaluation tests
- `run_comprehensive_eval.py` - Comprehensive benchmarks
- `run_500_task_benchmark.py` - Large-scale benchmarks

## Future Enhancements

Potential improvements:

- [ ] WebSocket-based real-time updates
- [ ] Web UI dashboard (FastAPI + React)
- [ ] Historical test result comparison
- [ ] Alert notifications for test failures
- [ ] Export results to CSV/JSON
- [ ] Integration with W&B for metric tracking

## Example Workflow

```bash
# 1. Start 3 tests in parallel (BALANCED, OPEN, CLOSED strategies)
cd /Users/bledden/Documents/weavehacks-collaborative

# Run in background with logging
./run_with_monitoring.sh BALANCED fe6e5e &
./run_with_monitoring.sh OPEN 8c42af &
./run_with_monitoring.sh CLOSED d9ac2f &

# 2. Open dashboard in separate terminal
python3 monitor.py fe6e5e 8c42af d9ac2f

# 3. Watch real-time progress!
# Dashboard updates every 2 seconds
# Press Ctrl+C to exit when done

# 4. View final results
# Dashboard shows summary when all tests complete
```

## Dependencies

All dependencies are already included in the project:

- `rich` - Beautiful terminal formatting (already installed)
- `python3` - Python 3.9+ required

No additional installation needed!
