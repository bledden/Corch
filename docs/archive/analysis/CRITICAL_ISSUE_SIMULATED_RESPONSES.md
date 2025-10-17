# CRITICAL ISSUE: Benchmarks Using Simulated Responses

## Problem Discovery

The benchmark system is running with **simulated/mock responses** instead of real API calls to OpenRouter.

### Evidence:

1. **No OpenRouter token usage reported** - User noticed no tokens being consumed
2. **LLM client fallback** - Code at `agents/llm_client.py:113-114` falls back to `_simulate_response()` when no real LLM client is available
3. **Output shows "using gpt-4"** but these are mock responses:
   ```
   [LLM] anthropic/claude-3.5-sonnet using gpt-4: # Design Document...
   [LLM] coder using gpt-4: Sure, I can help...
   ```

### Root Cause:

The `LLMClient` class (`agents/llm_client.py`) is structured to use **direct API clients**:
- Lines 106-107: `if "gpt" in model.lower() and self.openai_client:`
- Lines 108-109: `elif "claude" in model.lower() and self.anthropic_client:`
- Lines 110-111: `elif "gemini" in model.lower() and self.google_client:`
- **Lines 112-114**: `else: response = await self._simulate_response(...)`

BUT:
- `self.openai_client` is only initialized if `OPENAI_API_KEY` is set (it's not, shows as placeholder)
- `self.anthropic_client` is only initialized if `ANTHROPIC_API_KEY` is set (it's not, shows as placeholder)
- `self.google_client` is only initialized if `GOOGLE_API_KEY` is set (it's not, shows as placeholder)
- **OPENROUTER_API_KEY is valid** but **NOT BEING USED**

### What's Actually Happening:

```python
# Current flow:
1. Benchmark calls: execute_agent_task("coder", task)
2. LLM client checks: if "gpt" in model and self.openai_client  # False!
3. LLM client checks: elif "claude" in model and self.anthropic_client  # False!
4. LLM client checks: elif "gemini" in model and self.google_client  # False!
5. LLM client falls back: _simulate_response(agent_id, task, model)  # [WARNING] MOCK!
```

### Simulated Response Example:

```python
async def _simulate_response(self, agent_id: str, task: str, model: str) -> LLMResponse:
    """Simulate response when no LLM is available"""
    await asyncio.sleep(0.5)  # Fake latency

    responses = {
        "architect": f"For '{task[:50]}...', I recommend a modular architecture...",
        "coder": f"```python\ndef solution():\n    # Implementation for {task[:30]}\n    pass\n```",
        "reviewer": '{"issues_found": false, "suggestions": []}',
        "documenter": f"# Documentation\n\nThis solution addresses {task[:50]}..."
    }

    content = responses.get(agent_id, f"Response for {task[:100]}")

    return LLMResponse(
        content=content,
        model=model,
        tokens_used=len(content.split()) * 1.3,  # FAKE token count
        latency=0.5  # FAKE latency
    )
```

---

## What REAL Results Should Look Like

### Real API Call Result:
```json
{
  "task_id": 1,
  "category": "basic_algorithms",
  "method": "sequential",
  "pass": true,
  "quality_score": 0.85,
  "duration": 127.3,
  "output": "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n\n# Test cases\nassert is_prime(2) == True\nassert is_prime(17) == True\nassert is_prime(4) == False\nassert is_prime(1) == False\n```",
  "hallucination": {
    "hallucination_detected": false,
    "confidence": 0.0,
    "indicators": []
  },
  "model_used": "alibaba/qwen2.5-coder-32b-instruct",
  "model_type": "open_source",
  "tokens_used": 342,  // REAL token count from OpenRouter
  "api_cost": 0.000068,  // REAL cost: 342 tokens * $0.20/1M tokens
  "search_executed": false,
  "needs_external_info": false
}
```

### Current Simulated Result:
```json
{
  "task_id": 1,
  "category": "basic_algorithms",
  "method": "sequential",
  "pass": true,
  "quality_score": 0.75,
  "duration": 4.2,  // FAKE - way too fast (5 stages in 4 seconds?)
  "output": "```python\ndef solution():\n    # Implementation for Write a function to check if a number is prime\n    pass\n```",  // FAKE - generic mock response
  "hallucination": {
    "hallucination_detected": false,
    "confidence": 0.0,
    "indicators": []
  },
  "model_used": "alibaba/qwen2.5-coder-32b-instruct",
  "model_type": "open_source",
  "tokens_used": 26,  // FAKE - estimated from mock response length
  "api_cost": 0.0,  // FAKE - no real API call made
  "search_executed": false,
  "needs_external_info": false
}
```

**Key Differences:**
- **Duration**: Real would be ~60-180s for sequential (5 stages), not 4s
- **Output**: Real would have actual working code, not `pass` placeholder
- **Tokens**: Real would show actual OpenRouter token usage (hundreds per task)
- **Cost**: Real would show per-task API costs ($0.0001-$0.01 per task)

---

## Solutions

### Option 1: Integrate LiteLLM with OpenRouter (RECOMMENDED)

LiteLLM provides a unified interface for OpenRouter:

```python
# agents/llm_client.py
import litellm

class LLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Set OpenRouter API key for litellm
        litellm.api_key = os.getenv("OPENROUTER_API_KEY")
        litellm.api_base = "https://openrouter.ai/api/v1"

    async def execute_llm(
        self,
        agent_id: str,
        task: str,
        model: str,  # e.g., "openai/gpt-4", "anthropic/claude-sonnet-4.5"
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Execute task with specified LLM via OpenRouter"""

        prompt = self._build_prompt(agent_id, task)

        try:
            # LiteLLM automatically routes to OpenRouter
            response = await litellm.acompletion(
                model=model,  # "openai/gpt-4", "anthropic/claude-sonnet-4.5", etc.
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                tokens_used=response.usage.total_tokens,  // REAL tokens
                latency=response._response_ms / 1000.0,  // REAL latency
            )

        except Exception as e:
            # Real error handling, no fallback to mock
            raise Exception(f"OpenRouter API error: {str(e)}")
```

### Option 2: Direct OpenRouter API Integration

```python
import openai

class LLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.openrouter_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    async def execute_llm(self, agent_id: str, task: str, model: str, **kwargs) -> LLMResponse:
        prompt = self._build_prompt(agent_id, task)

        response = await self.openrouter_client.chat.completions.create(
            model=model,  # "openai/gpt-4", "anthropic/claude-sonnet-4.5"
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2000),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            tokens_used=response.usage.total_tokens,
            latency=0.0  # Calculate from timestamps
        )
```

---

## Impact on Current Benchmarks

### Smoke Test (0ca59a) - USING MOCKS [WARNING]
- All 10 tasks are getting simulated responses
- Pass/fail metrics are based on mock data
- No real API calls being made
- No token usage tracking
- Results are NOT VALID for evaluation

### Full 500-Task Benchmark (a1bb15) - USING MOCKS [WARNING]
- All 500 tasks will use simulated responses
- Estimated 15-20 hours is TOO FAST (should be 50-100 hours with real API calls)
- Pass@1 scores will be artificially inflated or deflated
- No real cost data
- Results are NOT VALID for comparing models

---

## Action Items

### Immediate (Stop Current Benchmarks):
1. [OK] **Kill both running benchmarks** - They're generating invalid mock data
2. [OK] **Fix LLM client to use OpenRouter** - Integrate LiteLLM or direct API
3. [OK] **Verify real API calls** - Check OpenRouter dashboard for token usage
4. [OK] **Re-run smoke test** - Validate with REAL API calls (should take 15-30 min)
5. [OK] **Launch full benchmark** - Only after smoke test passes with real data

### Verification Steps:
```bash
# Before re-running, verify:
1. Check OpenRouter dashboard: https://openrouter.ai/activity
2. Run single test call:
   python3 -c "
   import asyncio
   from agents.llm_client import LLMClient
   client = LLMClient({})
   result = asyncio.run(client.execute_llm('test', 'Say hello', 'openai/gpt-4'))
   print(f'Tokens: {result.tokens_used}')  # Should be > 0
   "
3. Check OpenRouter dashboard again - should show 1 API call
4. If no API call appears = still using mocks!
```

---

## Estimated Time with Real API Calls

### Smoke Test (10 tasks):
- **Sequential**: 10 tasks × 5 stages × ~15-30s per stage = 750-1500s (12-25 min)
- **Baseline**: 10 tasks × 1 call × ~10s = 100s (2 min)
- **Total**: ~15-30 minutes

### Full Benchmark (500 tasks):
- **Sequential**: 500 tasks × 5 stages × ~20s per stage = 50,000s (14 hours)
- **Baseline**: 500 tasks × 1 call × ~10s = 5,000s (1.4 hours)
- **Total**: ~15-20 hours (was correct estimate!)

### Expected Costs (with Real API Calls):
- **Cheapest models** (Qwen 2.5 Coder): $0.20/1M input, $0.60/1M output
  - Average task: ~1000 tokens input, ~500 tokens output = $0.0005 per call
  - Full benchmark: 1000 calls × $0.0005 = **~$0.50 total**

- **Expensive models** (GPT-4, Claude Opus): $3-$15/1M tokens
  - Average task: ~$0.005-$0.02 per call
  - Full benchmark: **~$5-$20 total**

---

## Summary

**Current State**: [FAIL] Benchmarks running with MOCK data (not valid)
**Issue**: LLM client not configured to use OpenRouter
**Solution**: Integrate LiteLLM or direct OpenRouter API
**Action**: Kill current benchmarks, fix API integration, restart

The benchmark infrastructure (hallucination detection, web search, model tracking) is excellent, but we need REAL API calls for valid results!
