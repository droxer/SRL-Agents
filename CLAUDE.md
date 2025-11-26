# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

SRL-Agents is a LangGraph playground for building Self-Regulated Learning (SRL) agents. It implements SRL's core loop—Forethought, Performance, and Self-Reflection—via a graph-based workflow.

## Code Architecture

### Core Components
- **Graph Workflow**: Defined in `srl_agents/graph.py`, the main workflow is a StateGraph with the following nodes:
  1. `forethought`: Analyzes user queries and retrieves relevant memory
  2. `actor`: Generates initial responses
  3. `reflector`: Reflects on responses to improve quality
  4. `critic`: Evaluates responses and decides next steps (APPROVE/REVISE/DISCARD)
  5. `store`: Saves approved responses to memory
- **Memory System**: Implemented in `srl_agents/memory.py`, uses Chroma for vector storage
- **Nodes**: Each graph node is implemented in `srl_agents/nodes/` with a `build_*_node()` function
- **State**: Uses `AgentState` defined in `srl_agents/state.py` to pass data between nodes

### Key Files
- `srl_agents/graph.py`: Main workflow assembly
- `srl_agents/memory.py`: MemoryStore implementation
- `srl_agents/config.py`: Configuration and dependency injection
- `srl_agents/state.py`: AgentState definition

## Development Commands

### Dependency Management
```bash
uv sync  # Install dependencies using uv (recommended)
cp .env.example .env  # Set up environment variables
```

### Running
```bash
make run               # Run the demo with predefined scenarios
make lg-dev            # Run LangGraph dev UI for interactive debugging
```

### Linting & Type Checking
```bash
make lint  # Ruff static analysis
make type  # Pyright type checking
```

### Testing
```bash
make test  # Run all tests with pytest
```

### Memory Management
```bash
make memory-list MEMORY_LIMIT=25  # List stored reflections
make memory-delete ID=<memory-id>  # Delete a specific reflection
make memory-reset  # Wipe the entire memory store
```

## Extending the Project
- Add new node variants in `srl_agents/nodes/`
- Customize memory store by implementing `MemoryStore` interface
- Expand `AgentState` for personalized prompts
- Add scenarios in `examples/scenarios.py`
