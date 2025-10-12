#!/bin/bash
echo "🧪 Running Smoke Tests..."
echo ""

# Test 1: CLI Health
echo "Test 1: CLI Health Check"
python3 cli.py health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ CLI health check passed"
else
    echo "❌ CLI health check failed"
    exit 1
fi

# Test 2: Import core modules
echo "Test 2: Import Core Modules"
python3 -c "from collaborative_orchestrator import CollaborativeOrchestrator" && echo "✅ collaborative_orchestrator imports" || exit 1
python3 -c "import sequential_orchestrator" && echo "✅ sequential_orchestrator imports" || exit 1
python3 -c "from agents.llm_client import MultiAgentLLMOrchestrator" && echo "✅ agents.llm_client imports" || exit 1

# Test 3: API module loads
echo "Test 3: API Module"
python3 -c "import api" && echo "✅ API module loads" || exit 1

# Test 4: CLI module loads  
echo "Test 4: CLI Module"
python3 -c "from cli import FacilitairCLI" && echo "✅ CLI module loads" || exit 1

# Test 5: Evaluation script loads
echo "Test 5: Evaluation Scripts"
python3 -c "from run_sequential_vs_baseline_eval import HallucinationDetector" && echo "✅ Evaluation script loads" || exit 1

echo ""
echo "✅ All smoke tests passed!"
