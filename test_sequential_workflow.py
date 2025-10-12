"""
Test Sequential Collaboration Workflow
Verifies that the new Facilitair_v2-style sequential architecture works correctly.
"""

import asyncio
import sys
from collaborative_orchestrator import CollaborativeOrchestrator, Strategy

async def main():
    print("=" * 80)
    print("Testing Sequential Collaboration Workflow (Facilitair_v2 Architecture)")
    print("=" * 80)
    print()

    # Initialize orchestrator with sequential mode ENABLED
    print("🔧 Initializing orchestrator with sequential collaboration...")
    orchestrator = CollaborativeOrchestrator(
        use_sequential=True,
        use_sponsors=False,  # Disable sponsors for quick test
        user_strategy=Strategy.BALANCED
    )
    print()

    # Simple test task
    task = "Create a Python function to calculate fibonacci numbers with memoization"

    print(f"📋 Task: {task}")
    print()

    print("🚀 Executing sequential workflow...")
    print()

    try:
        result = await orchestrator.collaborate(task)

        print("✅ Workflow completed!")
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()

        print(f"🔹 Workflow Type: {result.consensus_method}")
        print(f"🔹 Agents Used: {', '.join(result.agents_used)}")
        print(f"🔹 Stages Completed: {result.consensus_rounds}")
        print(f"🔹 Iterations (refinements): {result.conflicts_resolved}")
        print()

        print("📊 Metrics:")
        print(f"  - Quality: {result.metrics['quality']:.2f}")
        print(f"  - Efficiency: {result.metrics['efficiency']:.2f}")
        print(f"  - Harmony: {result.metrics['harmony']:.2f}")
        print(f"  - Overall: {result.metrics['overall']:.2f}")
        print()

        print("=" * 80)
        print("STAGE OUTPUTS")
        print("=" * 80)

        for agent_id, output in result.individual_outputs.items():
            print()
            print(f"🤖 {agent_id.upper()}")
            print("-" * 80)
            print(output[:300] + "..." if len(output) > 300 else output)
            print()

        print("=" * 80)
        print("FINAL OUTPUT")
        print("=" * 80)
        print()
        print(result.final_output[:500] + "..." if len(result.final_output) > 500 else result.final_output)
        print()

        print("=" * 80)
        print("✅ TEST PASSED: Sequential workflow working correctly!")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
