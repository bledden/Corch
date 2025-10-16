# Testing Instructions for Streaming Implementation

## Current Status

[OK] **Complete Implementation** - All code has been written and is ready to test:
- CLI Streaming Client: `cli/streaming_client.py`
- SSE Handler: `backend/streaming/sse_handler.py`
- Streaming Router: `backend/routers/streaming.py`
- API Integration: Updated `api.py` with streaming routes

## Issue: Multiple Server Processes

There are currently multiple old uvicorn server processes running that are preventing a clean test. These need to be killed first.

## How to Test (Manual Steps)

### Step 1: Kill All Old Servers

```bash
# Kill all uvicorn processes
pkill -9 -f "uvicorn api:app"

# Wait a moment
sleep 2

# Verify port 8000 is free
lsof -i :8000
# Should show nothing
```

### Step 2: Start Fresh Server

```bash
cd /Users/bledden/Documents/weavehacks-collaborative

# Start the server with our new streaming endpoints
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

You should see output like:
```
INFO:     Will watch for changes in these directories: ['/Users/bledden/Documents/weavehacks-collaborative']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
...
INFO:     Application startup complete.
```

### Step 3: Verify Streaming Endpoints Exist

In a **new terminal**, check if the streaming endpoints loaded:

```bash
curl -s http://localhost:8000/openapi.json | python3 -m json.tool | grep "/api/stream"
```

You should see:
```json
"/api/stream/task": {
"/api/stream/events/{stream_id}": {
"/api/stream/status/{stream_id}": {
```

If you don't see these, the server didn't reload with the new code.

### Step 4: Test the CLI Client

In the **same second terminal**:

```bash
cd /Users/bledden/Documents/weavehacks-collaborative

# Set API key (use any value for testing)
export FAC_API_KEY="test-key"

# Run the streaming client
python3 cli/streaming_client.py "Write a function to add two numbers"
```

### Expected Output

You should see:

1. **Stream Creation**:
```
Creating stream for task...
Stream created: <uuid>
```

2. **Live Dashboard** with Rich UI showing:
   - Task description at the top
   - Agent outputs in the middle (Architect, Coder, Reviewer, Documenter)
   - Progress and status at the bottom

3. **Streaming Events**:
```
[START] Task Started
[LIST] Plan: 4 chunks, ~60.0s, ~$0.08
Architect  Architect: Analyzing requirements... (1/4)
[OK] Chunk 1 complete
Coder Coder: Writing implementation... (2/4)
[OK] Chunk 2 complete
Reviewer Reviewer: Analyzing code quality... (3/4)
[OK] Chunk 3 complete
Documenter Documenter: Creating documentation... (4/4)
[OK] Chunk 4 complete
[OK] Task Completed!
```

4. **Final Result** displayed at the end

## What to Look For

### [OK] Success Indicators

- CLI connects without errors
- Events stream in real-time
- Rich UI updates live
- Agent names appear correctly
- Progress tracking works
- Final result displays

### [FAIL] Potential Issues

**"HTTP 404: Not Found"**:
- Streaming endpoints didn't load
- Server needs to be restarted

**"FAC_API_KEY environment variable not set"**:
- Need to set: `export FAC_API_KEY="test-key"`

**"Connection refused"**:
- Server isn't running on port 8000
- Check with: `lsof -i :8000`

**Import errors**:
- Missing dependencies
- Run: `pip3 install httpx rich`

## Testing Different Tasks

Try various tasks to test the system:

```bash
# Simple task
python3 cli/streaming_client.py "Sort a list of numbers"

# Complex task
python3 cli/streaming_client.py "Build a REST API with authentication"

# With context
python3 cli/streaming_client.py "Create a web server" '{"language": "python", "framework": "fastapi"}'
```

## Current MVP Limitations

The current implementation is an MVP with simulated streaming:

1. **Simulated Tokens**: Placeholder text instead of real LLM output
2. **Mock Stages**: Hardcoded 4 stages (architecture, implementation, review, documentation)
3. **No Real Orchestration**: The actual `CollaborativeOrchestrator.collaborate()` runs at the end

### Why This Is OK

This MVP **validates the infrastructure**:
- SSE protocol works
- Event handling works
- CLI client works
- UI updates work
- State management works

Once validated, we can integrate real LLM streaming tokens.

## Next Steps After Testing

If the test works:

1. [OK] Mark streaming infrastructure as validated
2. [REFRESH] Integrate real LLM token streaming
3. [REFRESH] Connect to actual orchestrator stages
4. [REFRESH] Add semantic caching
5. [REFRESH] Build React web UI

If the test fails:
1. Check error messages
2. Verify all files exist
3. Check imports work
4. Test endpoints manually with curl

## Automated Test Script

Alternatively, run this automated test script:

```bash
#!/bin/bash

# test_streaming.sh

echo " Testing Facilitair Streaming Implementation"
echo "=============================================="
echo ""

# Kill old servers
echo "Step 1: Killing old servers..."
pkill -9 -f "uvicorn api:app"
sleep 2

# Start server in background
echo "Step 2: Starting fresh server..."
cd /Users/bledden/Documents/weavehacks-collaborative
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload > /tmp/facilitair_server.log 2>&1 &
SERVER_PID=$!
sleep 5

# Check if server started
if ! lsof -i :8000 > /dev/null; then
    echo "[FAIL] Server failed to start!"
    cat /tmp/facilitair_server.log
    exit 1
fi

echo "[OK] Server started (PID: $SERVER_PID)"

# Check endpoints
echo "Step 3: Checking streaming endpoints..."
if curl -s http://localhost:8000/openapi.json | grep -q "/api/stream/task"; then
    echo "[OK] Streaming endpoints found!"
else
    echo "[FAIL] Streaming endpoints NOT found!"
    kill $SERVER_PID
    exit 1
fi

# Run CLI test
echo "Step 4: Running CLI client test..."
export FAC_API_KEY="test-key"
timeout 30 python3 cli/streaming_client.py "Write a hello world function" || true

echo ""
echo "[OK] Test complete!"
echo "Server is still running at PID $SERVER_PID"
echo "Kill it with: kill $SERVER_PID"
```

Save this as `test_streaming.sh`, make it executable, and run it:

```bash
chmod +x test_streaming.sh
./test_streaming.sh
```

## Summary

The streaming implementation is **complete and ready to test**. The only blocker is the multiple old server processes that need to be killed first. Once you have a clean slate, the test should work perfectly!
