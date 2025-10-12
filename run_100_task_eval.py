#!/usr/bin/env python3
"""
100-Task Comprehensive Evaluation
Tests sequential collaboration vs single-model baseline at scale
"""

import sys
sys.path.insert(0, '/Users/bledden/Documents/weavehacks-collaborative')

from run_sequential_vs_baseline_eval import *

# Override to 100 tasks
EVALUATION_TASKS = []

# Generate 100 diverse tasks
task_templates = [
    ("coding_easy", 0.2, [
        "Write a function to {action} in Python",
        "Create a {data_structure} implementation",
        "Implement {algorithm} algorithm",
    ]),
    ("coding_medium", 0.5, [
        "Build a {feature} with error handling",
        "Design a {pattern} implementation",
        "Create a {system} with {constraint}",
    ]),
    ("coding_hard", 0.9, [
        "Optimize {algorithm} for {constraint}",
        "Implement {complex_structure} with {operations}",
        "Design scalable {system} architecture",
    ]),
    ("debugging", 0.6, [
        "Debug {issue} in {context}",
        "Fix {problem} causing {symptom}",
        "Resolve {error_type} in {scenario}",
    ])
]

actions = ["sort a list", "reverse a string", "calculate factorial", "find duplicates", "merge arrays"]
data_structures = ["linked list", "binary tree", "hash table", "queue", "stack"]
algorithms = ["binary search", "quicksort", "breadth-first search", "dynamic programming"]
features = ["user authentication", "rate limiting", "caching layer", "API endpoint"]
patterns = ["singleton pattern", "factory pattern", "observer pattern", "decorator"]
systems = ["task scheduler", "cache system", "message queue", "load balancer"]
constraints = ["O(log n) time", "constant space", "thread-safe", "distributed"]
complex_structures = ["LRU cache", "B-tree", "trie", "skip list"]
operations = ["O(1) operations", "concurrent access", "persistence", "replication"]
issues = ["memory leak", "race condition", "deadlock", "stack overflow"]
contexts = ["multi-threaded code", "async operations", "recursive functions"]
problems = ["null pointer", "off-by-one error", "infinite loop"]
symptoms = ["crashes", "hangs", "data corruption"]
error_types = ["segmentation fault", "type error", "index error"]
scenarios = ["production environment", "high load", "edge cases"]

# Generate tasks
task_id = 1
for category, complexity, templates in task_templates:
    for _ in range(25):  # 25 tasks per category = 100 total
        import random
        template = random.choice(templates)

        # Fill template
        task_desc = template.format(
            action=random.choice(actions),
            data_structure=random.choice(data_structures),
            algorithm=random.choice(algorithms),
            feature=random.choice(features),
            pattern=random.choice(patterns),
            system=random.choice(systems),
            constraint=random.choice(constraints),
            complex_structure=random.choice(complex_structures),
            operations=random.choice(operations),
            issue=random.choice(issues),
            context=random.choice(contexts),
            problem=random.choice(problems),
            symptom=random.choice(symptoms),
            error_type=random.choice(error_types),
            scenario=random.choice(scenarios)
        )

        EVALUATION_TASKS.append({
            "id": task_id,
            "category": category,
            "description": task_desc,
            "complexity": complexity
        })
        task_id += 1

if __name__ == "__main__":
    print(f"Starting 100-task evaluation...")
    print(f"Tasks: {len(EVALUATION_TASKS)}")

    asyncio.run(run_comparison_evaluation())
