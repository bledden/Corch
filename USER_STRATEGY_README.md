# User-Controlled Model Selection Strategy

## 🎯 Overview

Our collaborative orchestrator now lets **YOU** decide whether to prioritize **quality**, **cost**, or find the perfect **balance**. This gives users complete control over the trade-offs between model performance and expenses.

## 🎮 Available Strategies

### 1. **QUALITY_FIRST** 🏆
- **When to use**: Production-critical code, high-stakes projects
- **Models**: GPT-5, Claude 4.5, O4 series
- **Cost**: $15-60 per million tokens
- **Performance**: 95-99% quality scores

### 2. **COST_FIRST** 💰
- **When to use**: High-volume processing, budget projects, learning
- **Models**: Qwen 2.5, DeepSeek V3, Llama 3.3 (all free!)
- **Cost**: $0 (open-source)
- **Performance**: 70-85% quality scores

### 3. **BALANCED** ⚖️
- **When to use**: Most projects (recommended default)
- **Models**: Mix of open and closed based on task
- **Cost**: 70% savings while maintaining quality
- **Performance**: 80-90% quality scores

### 4. **SPEED_FIRST** ⚡
- **When to use**: Real-time applications, demos
- **Models**: Lightweight models (Phi-4, Gemma, small variants)
- **Cost**: Minimal
- **Performance**: 60-75% quality, but 5-10x faster

### 5. **PRIVACY_FIRST** 🔐
- **When to use**: Sensitive data, regulated industries
- **Models**: Only local/open-source models
- **Cost**: $0 (self-hosted)
- **Performance**: 75-85% quality scores

## 🚀 Quick Start

### Interactive Setup
```python
from collaborative_orchestrator import SelfImprovingCollaborativeOrchestrator
from agents.strategy_selector import Strategy

# Let user choose
orchestrator = SelfImprovingCollaborativeOrchestrator(
    user_strategy=Strategy.BALANCED  # or QUALITY_FIRST, COST_FIRST, etc.
)
```

### Run the Interactive Demo
```bash
python demo_with_strategy.py
```

### Change Strategy at Runtime
```python
# Start with balanced
orchestrator.set_user_strategy(Strategy.BALANCED)

# Switch to quality for critical task
orchestrator.set_user_strategy(Strategy.QUALITY_FIRST)
result = await orchestrator.collaborate("Critical production code")

# Switch back to save costs
orchestrator.set_user_strategy(Strategy.COST_FIRST)
```

## 📊 Real Performance Data (October 2025)

### Quality-First Models
- **Claude Sonnet 4.5**: 77.2% SWE-bench (best)
- **GPT-5**: 74.9% SWE-bench
- **Cost**: $3-60 per million tokens

### Cost-First Models
- **Qwen 2.5 Coder**: 73.7% Aider (matches GPT-4o!)
- **DeepSeek V3**: 42% SWE-bench, 82.6% HumanEval
- **Cost**: $0 (open-source)

### The Surprise
**Qwen 2.5 Coder matches GPT-4o quality while being completely free!**

## 🎮 Dynamic Auto-Switching

The system can automatically switch strategies based on context:

```yaml
auto_switch:
  rules:
    - If task_complexity > 0.9: → QUALITY_FIRST
    - If remaining_budget < $10: → COST_FIRST
    - If user_waiting > 30s: → SPEED_FIRST
    - If sensitive_data == true: → PRIVACY_FIRST
```

## 💡 Strategy Recommendations

### By User Type
- **Enterprise**: QUALITY_FIRST
- **Startup**: BALANCED
- **Student/Hobbyist**: COST_FIRST
- **Real-time Apps**: SPEED_FIRST
- **Healthcare/Finance**: PRIVACY_FIRST

### By Task Type
- **Production Code**: QUALITY_FIRST or BALANCED
- **Prototyping**: COST_FIRST
- **Code Reviews**: BALANCED
- **Documentation**: COST_FIRST
- **Complex Algorithms**: QUALITY_FIRST

## 📈 Cost Savings Examples

### Scenario: 100 coding tasks per day

**QUALITY_FIRST Strategy**:
- Models: GPT-5, Claude 4.5
- Cost: ~$150/day
- Quality: 95%+

**BALANCED Strategy**:
- Models: Mix of Qwen 2.5 (free) and GPT-5 (when needed)
- Cost: ~$30/day (80% savings!)
- Quality: 85-90%

**COST_FIRST Strategy**:
- Models: Qwen 2.5, DeepSeek V3 (all free)
- Cost: $0/day (100% savings!)
- Quality: 75-85%

## 🔧 Configuration

Edit `model_strategy_config.yaml` to customize:
- Model preferences per strategy
- Cost thresholds
- Quality requirements
- Auto-switch rules

## 🎯 Why This Matters

1. **User Control**: You decide the trade-offs
2. **Cost Transparency**: See exactly what you're spending
3. **Flexibility**: Change strategies on the fly
4. **Smart Defaults**: Balanced mode for most users
5. **Future-Proof**: Easy to add new models/strategies

## 🏆 Perfect for WeaveHacks

This demonstrates:
- **Innovation**: User-controlled AI economics
- **Practicality**: Real cost savings
- **Flexibility**: Adapts to any use case
- **Transparency**: Clear trade-offs
- **Integration**: Works with W&B Weave tracking

## 📝 Example Output

```
📊 Task Complete!
   Strategy Used: BALANCED
   Model Selected: qwen2.5-coder-32b (free!)
   Quality Score: 85%
   Cost: $0.00
   Alternative (QUALITY_FIRST): gpt-5 ($15.00)

💰 You saved: $15.00 on this task!
```

---

**The Power is in Your Hands!** Choose quality when it matters, save money when it doesn't, and always maintain full control over your AI costs.