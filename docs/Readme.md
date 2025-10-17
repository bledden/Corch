# Facilitair Documentation

Welcome to the Facilitair documentation! This guide will help you understand, use, and extend the Facilitair collaborative AI orchestration system.

## Quick Start

- **New Users**: Start with [guides/Interfaces_Readme.md](guides/Interfaces_Readme.md) for CLI and API usage
- **Understanding the System**: Read [architecture/Architecture_Explanation.md](architecture/Architecture_Explanation.md)
- **Model Selection**: See [architecture/Model_Recommendations.md](architecture/Model_Recommendations.md)
- **Testing**: Follow [guides/TESTING_INSTRUCTIONS.md](guides/TESTING_INSTRUCTIONS.md)

## Documentation Structure

### Architecture
Core system design and model selection:
- [Architecture_Explanation.md](architecture/Architecture_Explanation.md) - Complete system architecture with diagrams
- [Sequential_Collaboration_Design.md](architecture/Sequential_Collaboration_Design.md) - Design rationale for sequential workflow
- [Sequential_Collaboration_Complete.md](architecture/Sequential_Collaboration_Complete.md) - Implementation status
- [Model_Recommendations.md](architecture/Model_Recommendations.md) - Model selection guide (533 models analyzed)

### Guides
User-facing guides and tutorials:
- [Interfaces_Readme.md](guides/Interfaces_Readme.md) - CLI and API interface documentation
- [User_Strategy_Readme.md](guides/User_Strategy_Readme.md) - Configure strategies (QUALITY_FIRST, COST_FIRST, etc.)
- [TESTING_INSTRUCTIONS.md](guides/TESTING_INSTRUCTIONS.md) - How to test the system
- [STREAMING_AND_CACHING_GUIDE.md](guides/STREAMING_AND_CACHING_GUIDE.md) - Streaming and caching features
- [MONITORING.md](guides/MONITORING.md) - Dashboard monitoring setup

### Evaluation
Quality metrics and benchmarking:
- [QUALITY_EVALUATION_GUIDE.md](evaluation/QUALITY_EVALUATION_GUIDE.md) - 6-dimension quality metrics
- [BENCHMARK_FAILURE_ANALYSIS.md](evaluation/BENCHMARK_FAILURE_ANALYSIS.md) - Lessons learned from failures
- [OCTOBER_2025_EVAL_RESULTS.md](evaluations/OCTOBER_2025_EVAL_RESULTS.md) - Latest evaluation results

### Integrations
Sponsor technology integrations:
- [Final_Sponsor_Summary.md](integrations/Final_Sponsor_Summary.md) - Status of all sponsor integrations

### Project
Project documentation and history:
- [Pitch.md](project/Pitch.md) - Original WeaveHacks pitch
- [Submission.md](project/Submission.md) - Official hackathon submission

## Archive

Historical development documents are preserved in [archive/](archive/) for reference:
- `archive/planning/` - Development plans and execution strategies
- `archive/streaming/` - Streaming implementation evolution
- `archive/cleanup/` - Code reorganization process
- `archive/analysis/` - Development analysis and research
- `archive/evaluations/` - Historical evaluation results
- `archive/research/` - Technology research and comparisons
- `archive/submissions/` - Previous submissions
- `archive/status/` - Temporary status updates

These archived documents provide valuable context about design decisions and the evolution of the system, but are not necessary for using or understanding the current implementation.

## Contributing

When adding new documentation:
1. Keep user-facing docs in the main directories
2. Move internal planning/temporary docs to `archive/`
3. Update this README with new essential documents
4. Use clear, concise language
5. Include examples where helpful

## Getting Help

- Issues: https://github.com/bledden/Corch/issues
- Main README: [../README.md](../README.md)
- Architecture Questions: See [architecture/](architecture/)
- Usage Questions: See [guides/](guides/)
