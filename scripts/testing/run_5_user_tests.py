"""
Run 5 user-simulation tests to validate the system works as a real user would use it
"""
import subprocess
import time

# 5 realistic user tasks
TASKS = [
    ("Write a function to calculate fibonacci numbers", "python"),
    ("Create a REST API endpoint for user authentication", "python"),
    ("Implement a binary search tree with insert and search methods", "python"),
    ("Write a function to validate email addresses using regex", "python"),
    ("Create a simple calculator class with add, subtract, multiply, divide", "python")
]

print("=" * 60)
print("Running 5 User-Simulation Tests")
print("=" * 60)

for i, (task, language) in enumerate(TASKS, 1):
    print(f"\n\n{'='*60}")
    print(f"TEST {i}/5: {task}")
    print('='*60)
    
    start = time.time()
    
    cmd = ["python3", "cli.py", "collaborate", task, "--language", language]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    duration = time.time() - start
    
    if result.returncode == 0:
        print(f"✅ SUCCESS (took {duration:.1f}s)")
        # Show last 30 lines of output
        lines = result.stdout.split('\n')
        print('\n'.join(lines[-30:]))
    else:
        print(f"❌ FAILED (took {duration:.1f}s)")
        print("STDERR:", result.stderr[-500:])
    
    print(f"\n{'='*60}\n")
    time.sleep(2)  # Brief pause between tests

print("\n" + "="*60)
print("All 5 tests complete!")
print("="*60)
