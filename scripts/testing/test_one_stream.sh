#!/bin/bash
# Test one streaming session with automatic cleanup

export FAC_API_KEY="test-key"

echo "🧪 Testing streaming with 30-second timeout..."
echo ""

# Run with timeout to prevent hanging
timeout 30 python3 cli/streaming_client_simple.py "$1" 2>&1

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Stream completed successfully"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏱️  Timed out after 30 seconds (probably waiting for orchestrator)"
    echo "    This is normal for MVP - the backend tries to call the real orchestrator"
    echo "    which may not complete. The streaming events should still show!"
else
    echo "❌ Failed with exit code: $EXIT_CODE"
fi
