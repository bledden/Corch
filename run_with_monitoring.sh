#!/bin/bash
# Wrapper script to run tests with logging for monitor dashboard
#
# Usage:
#    ./run_with_monitoring.sh <strategy> <bash_id>
#
# Example:
#    ./run_with_monitoring.sh BALANCED fe6e5e
#
# This will:
# 1. Run the smoke test with the specified strategy
# 2. Write output to /tmp/facilitair_logs/test_<bash_id>.log
# 3. Allow monitor.py to display real-time progress

STRATEGY=$1
BASH_ID=$2

if [ -z "$STRATEGY" ] || [ -z "$BASH_ID" ]; then
    echo "Usage: $0 <STRATEGY> <BASH_ID>"
    echo "Example: $0 BALANCED fe6e5e"
    exit 1
fi

# Create log directory
mkdir -p /tmp/facilitair_logs

# Log file path
LOG_FILE="/tmp/facilitair_logs/test_${BASH_ID}.log"

echo "Starting test with strategy: $STRATEGY"
echo "Bash ID: $BASH_ID"
echo "Log file: $LOG_FILE"
echo "Monitor with: python3 monitor.py $BASH_ID"
echo ""

# Run the test and tee output to both stdout and log file
STRATEGY=$STRATEGY python3 run_smoke_test.py 2>&1 | tee "$LOG_FILE"
