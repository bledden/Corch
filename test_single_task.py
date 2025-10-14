"""
Single task test to validate multi-model selection
"""
import asyncio
from collaborative_orchestrator import CollaborativeOrchestrator
import yaml

# No weave.init here - collaborative_orchestrator.py already initializes Weave

async def test_single_task():
    print("Loading config...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Initializing orchestrator...")
    orchestrator = CollaborativeOrchestrator(config)

    print("\n=== TESTING SINGLE TASK ===")
    print("Task: Write a function to add two numbers\n")

    try:
        result = await orchestrator.collaborate("Write a function to add two numbers")

        print("\n=== RESULTS ===")
        print(f"Success: {result.success}")
        print(f"Final output length: {len(result.final_output)}")
        print(f"Output preview: {result.final_output[:200]}...")

        # Check which models were used
        if hasattr(result, 'metadata') and 'models_used' in result.metadata:
            print(f"\n=== MODELS USED ===")
            for agent, model in result.metadata['models_used'].items():
                print(f"  {agent}: {model}")

        return True

    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_single_task())
    exit(0 if success else 1)
