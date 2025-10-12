# 🚀 Self-Improving Collaborative Agent System
**WeaveHacks 2 Project - July 12-13, 2025**

### 🏆 Sponsor Technology Integration Status
**Working**: OpenAI API | Ray RLlib | Prefect
**Ready with API Keys**: W&B Weave | Tavily | OpenRouter
**Architecture Prepared For**: Google Cloud | BrowserBase | AG-UI

## The Problem
Multi-agent collaboration today is static - agents don't learn from past collaborations. They make the same coordination mistakes, use the same suboptimal consensus strategies, and never improve their teamwork.

## The Solution
A self-improving collaborative execution system where agents learn:
1. **WHO** to collaborate with (agent selection)
2. **HOW** to reach consensus (voting vs. debate vs. synthesis)
3. **WHEN** to defer to experts (confidence calibration)
4. **WHAT** each agent is best at (capability discovery)

## Core Innovation
Every collaboration is tracked in W&B Weave, and the system learns:
- Which agents work best together
- Optimal consensus strategies for different task types
- Each agent's strengths and weaknesses
- How to resolve conflicts efficiently

## Quick Start

### Prerequisites
- Python 3.8+
- W&B Account (get API key from https://wandb.ai/authorize)
- At least one LLM API key (OpenAI, Anthropic, or Google)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/weavehacks-collaborative
cd weavehacks-collaborative

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

### Setup & Configuration
```bash
# Run setup script to check all integrations
python setup_services.py

# This will check:
# - Environment variables
# - API key connections
# - All sponsor technology integrations
# - Verify all dependencies
# - Initialize W&B Weave
# - Test the connection
```

### Running the System

#### Training Mode - Watch Agents Learn
```bash
# Full training (10 generations, ~5 minutes)
python train.py

# Fast training (5 generations, ~2 minutes)
python train.py --fast

# Custom generations
python train.py --generations 20
```

#### Demo Mode - Hackathon Presentation
```bash
# Run the full demo with visualizations
python demo.py

# Fast demo for quick presentations
python demo.py --fast

# Single task execution
python demo.py --task "Design a microservices architecture"
```

#### Execute Mode - Use Trained System
```bash
# Compare untrained vs trained collaboration
python execute.py

# Execute specific task
python execute.py --task "Build a REST API with authentication"
```

## System Architecture

### Five Specialized Agents

1. **Architect** (GPT-4)
   - Expertise: System design, architecture, planning
   - Learns: When to lead design decisions

2. **Coder** (Claude-3 Sonnet)
   - Expertise: Implementation, debugging, optimization
   - Learns: Collaboration with reviewers

3. **Reviewer** (GPT-4 Turbo)
   - Expertise: Code review, testing, quality
   - Learns: Which bugs each agent tends to miss

4. **Documenter** (Claude-3 Haiku)
   - Expertise: Documentation, examples, tutorials
   - Learns: When to intervene for clarity

5. **Researcher** (Gemini Pro)
   - Expertise: Research, analysis, data
   - Learns: Information synthesis patterns

### Consensus Methods

The system learns which consensus method works best for each task type:

- **Voting**: Simple majority vote
- **Weighted Voting**: Expertise-based weights
- **Debate**: Agents argue until consensus
- **Synthesis**: Combine all perspectives
- **Hierarchy**: Domain expert decides

### Learning Mechanisms

1. **Performance Tracking**: Each agent's success rate per task type
2. **Collaboration Scores**: How well agents work together
3. **Pattern Recognition**: Optimal team compositions
4. **Consensus Optimization**: Best methods per scenario

## 🏆 Sponsor Technology Integrations

### Full Stack Implementation
Every sponsor technology is deeply integrated into the system:

| Technology | Status | Integration | Notes |
|------------|--------|-------------|-------|
| **OpenAI API** | ✅ WORKING | Powers all LLM agents | Active with API key |
| **Ray RLlib** | ✅ WORKING | Reinforcement learning for collaboration | Fully functional |
| **Prefect** | ✅ READY | Workflow orchestration | Installed and ready |
| **W&B Weave** | 🔑 NEEDS KEY | Tracking & learning metrics | Code complete, needs `WANDB_API_KEY` |
| **Tavily** | 🔑 NEEDS KEY | AI web search | Code complete, needs `TAVILY_API_KEY` |
| **OpenRouter** | 🔑 NEEDS KEY | Open-source models | Code complete, needs `OPENROUTER_API_KEY` |
| **Google Cloud** | ⚙️ SETUP NEEDED | Cloud infrastructure | Requires GCP project |
| **BrowserBase** | ⚙️ SETUP NEEDED | Web automation | Requires API key + Playwright |
| **AG-UI** | ⚙️ PARTIAL | Agent visualization | Pydantic AI installed, needs config |

### Run Sponsor Showcase
```bash
# See all sponsors in action
python demo_sponsor_showcase.py

# Interactive strategy selection with sponsors
python demo_with_strategy.py
```

See [SPONSOR_INTEGRATIONS.md](SPONSOR_INTEGRATIONS.md) for detailed documentation.

## Configuration

### Environment Variables (.env)
```bash
# Core LLM APIs (at least one required)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Sponsor Technologies
WANDB_API_KEY=your_wandb_key           # W&B Weave tracking
TAVILY_API_KEY=your_tavily_key         # AI web search
BROWSERBASE_API_KEY=your_browserbase_key # Web automation
OPENROUTER_API_KEY=your_openrouter_key # Open-source models
MASTRA_API_KEY=your_mastra_key         # Workflow orchestration
GCP_PROJECT_ID=your_gcp_project        # Google Cloud
AGUI_API_KEY=your_agui_key             # Agent visualization
DAYTONA_API_URL=http://localhost:3000  # Dev environments
```

### Agent Configuration (config.yaml)
- Model assignments
- Temperature settings
- Expertise domains
- Personality prompts
- Consensus parameters

## Demo Scenarios

### Scenario 1: Code Review Evolution
Watch as agents learn who catches which types of bugs:
- Generation 1: Everyone reviews everything (chaos)
- Generation 5: Specialization emerges
- Generation 10: Optimal reviewer assignment

### Scenario 2: Architecture Decisions
See consensus methods evolve:
- Early: Simple voting (often wrong)
- Middle: Weighted voting (better)
- Late: Hierarchy with architect leading

### Scenario 3: Documentation Quality
Observe collaboration improvement:
- Start: Documenter works alone
- Progress: Coder provides context
- Final: Full team contributes relevant parts

## W&B Weave Integration

Track everything in real-time:
- Individual agent outputs
- Consensus processes
- Learning updates
- Performance metrics
- Collaboration patterns

View your dashboard at: https://wandb.ai/your-entity/weavehacks-collaborative

## Troubleshooting

### No API Keys
The system will use simulated responses if no LLM API keys are provided. This is useful for testing but won't show real collaboration dynamics.

### Weave Connection Issues
If Weave fails to initialize:
1. Check your WANDB_API_KEY
2. Verify network connection
3. System continues without tracking (local mode)

### Import Errors
Run `pip install -r requirements.txt` to install all dependencies

## The Magic Moment

The hackathon demo shows a clear transformation:

**Generation 1**: Chaos
- Agents talk over each other
- Wrong consensus methods
- Poor team selection
- Low quality outputs

**Generation 10**: Harmony
- Agents know their roles
- Optimal team composition
- Efficient consensus
- High quality outputs

This demonstrates how multi-agent systems can learn to collaborate better over time, with every interaction tracked and analyzed by W&B Weave!

## Sponsor Integration
- **W&B Weave**: Full collaboration tracking and learning curves
- **Daytona**: Isolated agent execution environments (planned)
- **MCP**: Inter-agent communication protocol (planned)
- **CopilotKit**: Human guidance for collaboration (planned)

## License
MIT License - WeaveHacks 2 Project

## Authors
Built for WeaveHacks 2 (July 12-13, 2025)