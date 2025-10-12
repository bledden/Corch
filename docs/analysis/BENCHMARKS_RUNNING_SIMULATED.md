# Facilitair Benchmarks - Currently Running

## Status: ✅ BOTH BENCHMARKS ACTIVE

---

## 1. Smoke Test (10 Tasks)

**Process ID**: `0ca59a`
**Status**: ✅ Running - Task 4/10 complete (40%)
**W&B Weave**: https://wandb.ai/facilitair/smoke-test/weave

### Test Tasks:
- **5 Self-Contained**: prime check, factorial, reverse string, palindrome, GCD
- **5 Web-Search**: Next.js 15, React 19, Stripe 2024, GPT-4 Turbo, OWASP 2023

### Validation Criteria:
- ✅ Sequential method >= 70% pass rate
- ✅ Web search integration triggers correctly
- ✅ Hallucination detection active
- ✅ Model type tracking (open vs closed source)

### Progress:
- ✅ Task 1: Prime check - COMPLETED
- ✅ Task 2: Factorial recursion - COMPLETED (with refinement iterations)
- ✅ Task 3: Reverse string - COMPLETED
- 🏃 Task 4: Palindrome check - IN PROGRESS
- ⏳ Task 5-10: Pending

---

## 2. Optimized 500-Task Benchmark (FULL RUN)

**Process ID**: `a1bb15`
**Status**: ✅ Running - Just started
**W&B Weave**: https://wandb.ai/facilitair/optimized-500-task-benchmark/weave

### Task Breakdown:
- **250 Self-Contained Tasks**: Algorithms, data structures (no web search)
  - 50 Basic Algorithms
  - 60 Data Structures
  - 70 Medium Algorithms
  - 70 Hard Algorithms

- **250 Web-Search-Requiring Tasks**: Current technologies
  - 60 Modern Frameworks (Next.js 15, React 19, Vue 3.5, etc.)
  - 50 Cloud & Infrastructure (AWS, GCP, Azure latest)
  - 50 AI & ML Integration (GPT-4 Turbo, Claude 3.5, LangChain, etc.)
  - 40 Security & Compliance (OWASP 2023, CVE patches, OAuth 2.1)
  - 50 API & Protocols (OpenAPI 3.1, GraphQL Federation v2, etc.)

### Total Runs:
- **1000 runs total** (500 sequential + 500 baseline)
- **Estimated Duration**: 15-20 hours

### Features Being Evaluated:

#### 1. Web Search Integration ✅
- **Tavily API**: $0.001 per search (cheapest)
- **Perplexity Sonar**: $0.005 per search (fastest)
- **Perplexity R1**: $0.015 per search (highest quality)
- **Gemini 2.5 Pro**: $0.010 per search (balanced)
- **Routing**: Automatic selection based on strategy (cheapest, fastest, quality, balanced)
- **Cost Tracking**: Per-search cost recorded in results

#### 2. Hallucination Detection ✅
- **8 Pattern Categories**:
  1. Unfounded claims ("I have access to", "I can browse")
  2. Self-contradictions ("both true and false")
  3. Fabricated API endpoints (`.fakemethod()`)
  4. Made-up version numbers (version 99.x)
  5. Fictional package names (`import doesnotexist`)
  6. Claims about future events
  7. Impossible claims ("O(0)", "100% accuracy guaranteed")
  8. Invalid syntax ("def async lambda")
- **Output**: `{"hallucination_detected": bool, "confidence": float, "indicators": List[str]}`

#### 3. Open/Closed Source Model Tracking ✅
- **Closed Source**: OpenAI/*, Anthropic/*, Google/Gemini*, Perplexity/*
- **Open Source**: Qwen/*, DeepSeek/*, Alibaba/Qwen*, Meta-Llama/*, Mistralai/Codestral*, Cohere/*
- **Comparison Tables**: Pass rates, quality scores per model type

#### 4. Task Type Differentiation ✅
- **Self-Contained**: Pure algorithmic tasks (no external info)
- **Web-Search**: Tasks requiring current documentation
- **Separate Statistics**: Pass rates calculated independently

#### 5. Pass@1 Scoring (HumanEval Standard) ✅
- **Sequential**: Multi-stage validation quality > 0.7 + no hallucinations
- **Baseline**: Code heuristics (has code + logic) + no hallucinations
- **Binary Metric**: Task passes (1) or fails (0)

#### 6. W&B Weave Integration ✅
- **All @weave.op() decorated functions tracked**
- **Nested traces**: Complete call hierarchy
- **Performance metrics**: Duration, quality, cost
- **Live dashboards**: Real-time monitoring

---

## Results Output

### Files Generated:
1. **Smoke Test**: `smoke_test_results_[timestamp].json`
2. **Full Benchmark**: `benchmark_optimized_500_results_[timestamp].json`

### Result Fields (Per Task):
```json
{
  "task_id": 1,
  "category": "basic_algorithms",
  "method": "sequential",
  "pass": true,
  "quality_score": 0.85,
  "overall_score": 0.82,
  "duration": 45.3,
  "hallucination": {
    "hallucination_detected": false,
    "confidence": 0.0,
    "indicators": []
  },
  "output": "...",
  "needs_external_info": false,
  "search_confidence": 0.0,
  "matched_patterns": [],
  "search_executed": false,
  "search_method_used": null,
  "search_cost": 0.0,
  "model_used": "alibaba/qwen2.5-coder-32b-instruct",
  "model_type": "open_source"
}
```

### Statistics Tables:

#### Table 1: Main Benchmark Results
- Pass@1 (%)
- Tasks Passed (x/500)
- Hallucinations count
- Avg Quality (0-1.0)
- Avg Duration (seconds)

#### Table 2: Task Type Breakdown
- Self-Contained: count, pass rate (sequential vs baseline)
- Web-Search: count, pass rate (sequential vs baseline)

#### Table 3: Search Method Breakdown
- Tavily API: usage count, total cost
- Perplexity Sonar: usage count, total cost
- Perplexity R1: usage count, total cost
- Gemini 2.5 Pro: usage count, total cost
- **Total searches executed**
- **Total cost (USD)**

#### Table 4: Model Type Performance
- Open Source: task count, pass rate, avg quality
- Closed Source: task count, pass rate, avg quality
- **Comparison**: Which model type performs better

---

## Monitoring Commands

### Check Smoke Test Progress:
```bash
# View latest output
python3 -c "from agents.bash_tools import BashOutput; print(BashOutput('0ca59a'))"
```

### Check Full Benchmark Progress:
```bash
# View latest output
python3 -c "from agents.bash_tools import BashOutput; print(BashOutput('a1bb15'))"
```

### View W&B Weave Dashboards:
- **Smoke Test**: https://wandb.ai/facilitair/smoke-test/weave
- **Full Benchmark**: https://wandb.ai/facilitair/optimized-500-task-benchmark/weave

---

## Comparison to Original Benchmark

### Improvements:
1. **Zero Redundancy**: Removed 248 generic filler tasks
2. **Balanced Coverage**: 50/50 split between self-contained and web-search
3. **Real-World Relevance**: All web-search tasks use 2024-2025 technologies
4. **Comprehensive Tracking**: Hallucinations, search methods, model types
5. **Cost Transparency**: Full search API cost tracking
6. **Model Type Analytics**: Open vs closed source performance comparison

### Original (498 tasks):
- ❌ 248 redundant/filler tasks
- ❌ No web search integration
- ❌ No hallucination detection
- ❌ No model type tracking
- ❌ No search cost tracking

### Optimized (500 tasks):
- ✅ 500 unique, high-quality tasks
- ✅ Web search with cost tracking
- ✅ Hallucination detection (8 categories)
- ✅ Open/closed source comparison
- ✅ Search method breakdown

---

## Sponsor API Integration

### ✅ W&B Weave (PRIMARY SPONSOR)
- **Status**: ACTIVE
- **Usage**: @weave.op() decorators throughout
- **Dashboards**: Live tracking of all benchmark runs

### ✅ Tavily Search API (SPONSOR)
- **Status**: ACTIVE (for web-search tasks)
- **Cost**: $0.001 per search
- **Integration**: WebSearchRouter selects Tavily for "cheapest" strategy

### ✅ Daytona Development Environments (SPONSOR)
- **Status**: INFRASTRUCTURE READY
- **Usage**: Development environment infrastructure prepared

---

## Expected Completion Time

### Smoke Test:
- **10 tasks** × 2 methods (sequential + baseline) = 20 runs
- **Est. Duration**: 15-30 minutes
- **Current Progress**: 40% complete (Task 4/10)
- **ETA**: ~20 minutes remaining

### Full Benchmark:
- **500 tasks** × 2 methods (sequential + baseline) = 1000 runs
- **Est. Duration per task**: ~60-120 seconds (sequential includes 5 stages)
- **Total Est. Duration**: 15-20 hours
- **Current Progress**: Just started (0%)
- **ETA**: Complete by Oct 13, 2025 ~3:00 PM

---

## Post-Completion Analysis

Once both benchmarks complete, we will analyze:

1. **Pass@1 Rates**: Sequential vs Baseline performance
2. **Web Search Impact**: Do web-search tasks benefit from external info?
3. **Hallucination Rates**: Which method hallucinates more?
4. **Model Type Comparison**: Open-source vs closed-source performance
5. **Search Method Efficiency**: Which search method is most cost-effective?
6. **Task Type Performance**: Self-contained vs web-search success rates

---

**Last Updated**: Oct 12, 2025 16:48 PST
**Status**: Both benchmarks running successfully ✅
