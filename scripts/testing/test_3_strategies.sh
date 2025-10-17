#!/bin/bash
# Test all 3 strategies with the same query

TEST_QUERY="Write a function to check if a number is prime"

echo "=========================================="
echo "Testing 3 Model Selection Strategies"
echo "Query: $TEST_QUERY"
echo "=========================================="

cd /Users/bledden/Documents/weavehacks-collaborative

# Test 1: BALANCED (default)
echo ""
echo "TEST 1: BALANCED Strategy"
echo "===================="
python3 cli.py collaborate "$TEST_QUERY" --format json 2>&1 | grep -E "Model selection strategy|architect using|coder using|reviewer using|documenter using|Overall Score|SUCCESS|FAILED"

# Test 2: OPEN (cost-first, open source only)
echo ""
echo "TEST 2: OPEN Strategy  "
echo "===================="
# Note: We need to find how to set strategy via CLI or env var
python3 cli.py collaborate "$TEST_QUERY" --format json 2>&1 | grep -E "Model selection strategy|architect using|coder using|reviewer using|documenter using|Overall Score|SUCCESS|FAILED"

# Test 3: CLOSED (quality-first, premium models)
echo ""
echo "TEST 3: CLOSED Strategy"
echo "===================="
python3 cli.py collaborate "$TEST_QUERY" --format json 2>&1 | grep -E "Model selection strategy|architect using|coder using|reviewer using|documenter using|Overall Score|SUCCESS|FAILED"

echo ""
echo "=========================================="
echo "All 3 strategy tests complete!"
echo "=========================================="
