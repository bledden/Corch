#!/bin/bash

# Quick Test Script for Streaming Implementation
# This tests the streaming infrastructure end-to-end

echo "=============================================="
echo "Facilitair Streaming Quick Test"
echo "=============================================="
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Step 1: Kill any existing servers
echo "Step 1: Cleaning up old servers..."
pkill -9 -f "uvicorn api:app" 2>/dev/null
sleep 2

# Step 2: Start the server
echo "Step 2: Starting API server with streaming..."
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload > /tmp/facilitair_streaming_test.log 2>&1 &
SERVER_PID=$!
echo "   Server PID: $SERVER_PID"
sleep 8

# Step 3: Check if server started
echo "Step 3: Verifying server is running..."
if ! lsof -i :8000 > /dev/null 2>&1; then
    echo "   [FAIL] Server failed to start!"
    echo "   Check logs: tail -f /tmp/facilitair_streaming_test.log"
    exit 1
fi
echo "   [OK] Server is running on port 8000"

# Step 4: Check for streaming endpoints
echo "Step 4: Checking for streaming endpoints..."
sleep 2
if curl -s http://localhost:8000/openapi.json | grep -q "/api/stream/task"; then
    echo "   [OK] Streaming endpoints found!"
    echo "   - POST /api/stream/task"
    echo "   - GET /api/stream/events/{stream_id}"
else
    echo "   [FAIL] Streaming endpoints NOT found!"
    echo "   The server may not have the new code."
    echo "   Check logs: tail -f /tmp/facilitair_streaming_test.log"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Step 5: Test CLI client
echo ""
echo "Step 5: Testing CLI streaming client..."
echo "   Task: 'Write a hello world function'"
echo ""

export FAC_API_KEY="test-key-12345"

# Run with timeout to prevent hanging
timeout 60 python3 cli/streaming_client.py "Write a hello world function" 2>&1 | tee /tmp/cli_output.log

CLI_EXIT=$?

echo ""
echo "=============================================="
echo "Test Results"
echo "=============================================="

if [ $CLI_EXIT -eq 0 ]; then
    echo "[OK] CLI client completed successfully"
elif [ $CLI_EXIT -eq 124 ]; then
    echo "[TIMEOUT] CLI client timed out (60s limit)"
else
    echo "[FAIL] CLI client failed with exit code: $CLI_EXIT"
fi

echo ""
echo "Logs:"
echo "   Server: /tmp/facilitair_streaming_test.log"
echo "   CLI Output: /tmp/cli_output.log"
echo ""
echo "Server PID: $SERVER_PID"
echo "   To stop: kill $SERVER_PID"
echo "   Or run: pkill -f 'uvicorn api:app'"
echo ""
echo "=============================================="
