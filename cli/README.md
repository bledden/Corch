# CLI Streaming Client

Streaming CLI client for visualizing real-time collaborative orchestration in the terminal.

## Overview

This CLI client connects to Facilitair's streaming API endpoints and displays agent collaboration as it happens, creating a "live debate" visualization in your terminal.

## Features

- **Live SSE Streaming**: Connects to `/api/stream/events/{stream_id}` for real-time updates
- **Rich Terminal UI**: Beautiful terminal interface with progress indicators
- **Agent Visualization**: Maps backend events to friendly agent metaphors:
  - 🏗️ Architect - Designs system architecture
  - 💻 Coder - Implements the solution
  - 🔍 Reviewer - Analyzes code quality
  - ✨ Refiner - Improves the implementation
  - 📝 Documenter - Creates documentation

## Installation

1. Install required dependencies:

```bash
pip install httpx>=0.25.0 rich>=13.7.0
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
export FAC_API_KEY="your-api-key-here"
python cli/streaming_client.py "Build a REST API with JWT authentication"
```

### With Custom Context

You can provide additional context as JSON:

```bash
python cli/streaming_client.py "Build a REST API" '{"language": "python", "frameworks": ["fastapi"]}'
```

### Environment Variables

- `FAC_API_KEY` (required): Your Facilitair API key
- `FACILITAIR_API_BASE` (optional): API base URL (default: `http://localhost:8000`)

## How It Works

### Event Flow

1. **Create Stream**: Client calls `POST /api/stream/task` to create a stream
2. **Connect to SSE**: Client connects to `GET /api/stream/events/{stream_id}`
3. **Receive Events**: Backend sends SSE events as agents work:
   - `task_started` - Task begins
   - `task_progress` - Planning info (chunks, time, cost estimates)
   - `chunk_started` - Agent starts working
   - `token_stream` - Live tokens from agent output
   - `chunk_completed` - Agent finishes
   - `task_completed` - Final result ready

### Event Mapping

The client maps backend events to a debate metaphor:

| Backend Event | CLI Display |
|--------------|-------------|
| `chunk_started` (architecture stage) | 🏗️ Architect: Analyzing requirements... |
| `chunk_started` (implementation stage) | 💻 Coder: Writing implementation... |
| `chunk_started` (review stage) | 🔍 Reviewer: Analyzing code quality... |
| `token_stream` | Appends tokens to current agent's output |
| `chunk_completed` | ✅ Chunk X complete |
| `task_completed` | ✅ Task Completed! |

## Implementation Status

### ✅ Completed

- [x] SSE client implementation with proper event parsing
- [x] Rich terminal UI with live dashboard
- [x] Event-to-agent mapping logic
- [x] Progress tracking and status display
- [x] Error handling and graceful shutdown
- [x] Executable script with CLI args

### ⏳ Pending (Backend Requirements)

The following backend components need to be implemented before this client can be tested:

1. **Streaming Endpoints** (`backend/routers/streaming.py`):
   - `POST /api/stream/task` - Create stream and return stream_id
   - `GET /api/stream/events/{stream_id}` - SSE endpoint

2. **SSE Handler** (`backend/streaming/sse_handler.py`):
   - Event builders for all event types
   - Stream management and cleanup
   - Token buffering and chunking

3. **Integration** with `SequentialCollaborativeOrchestrator`:
   - Emit events at each stage transition
   - Stream tokens as they're generated
   - Send progress updates

## Development

### File Structure

```
cli/
├── streaming_client.py    # Main CLI client
└── README.md             # This file
```

### Testing

Once backend streaming is implemented, test with:

```bash
# Start the API server
python -m uvicorn api:app --reload

# In another terminal, run the CLI client
export FAC_API_KEY="test-key"
python cli/streaming_client.py "Write a function to sort a list"
```

## Architecture

```
┌─────────────┐                ┌─────────────┐
│             │  POST /stream  │             │
│   CLI       ├───────────────>│   Backend   │
│   Client    │  (create)      │   API       │
│             │                │             │
│             │<───────────────┤             │
│             │  stream_id     │             │
└──────┬──────┘                └──────┬──────┘
       │                              │
       │  GET /stream/events/{id}     │
       │  (SSE connection)             │
       │<─────────────────────────────┤
       │                              │
       │  event: task_started         │
       │  data: {...}                 │
       │<─────────────────────────────┤
       │                              │
       │  event: token_stream         │
       │  data: {"tokens": "..."}     │
       │<─────────────────────────────┤
       │                              │
       │  event: task_completed       │
       │  data: {"result": "..."}     │
       │<─────────────────────────────┤
       │                              │
```

## Next Steps

1. **Implement Backend Streaming**:
   - Create `backend/routers/streaming.py` with stream endpoints
   - Create `backend/streaming/sse_handler.py` for event management
   - Integrate with `SequentialCollaborativeOrchestrator`

2. **Test the Integration**:
   - Start backend server
   - Run CLI client with test task
   - Verify events are received and displayed correctly

3. **Add Features**:
   - Cache hit indicator (show when result came from cache)
   - Cost and time estimates display
   - Save output to file option
   - Non-live mode for CI/CD environments

## Related Documentation

- [REVISED_EXECUTION_PLAN.md](../REVISED_EXECUTION_PLAN.md) - Full implementation plan
- [STREAMING_CONSENSUS_IMPLEMENTATION.md](../STREAMING_CONSENSUS_IMPLEMENTATION.md) - Technical deep dive

## Troubleshooting

### "Error: FAC_API_KEY environment variable not set"

Set your API key:

```bash
export FAC_API_KEY="your-key-here"
```

### "HTTP 404: Not Found"

The streaming endpoints don't exist yet. They need to be implemented in the backend first.

### Connection timeout

Check that:
1. Backend server is running (`python -m uvicorn api:app`)
2. `FACILITAIR_API_BASE` is set correctly
3. Server is accessible from your network

## License

Part of the Facilitair project.
