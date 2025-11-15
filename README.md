# dangerously-prompt-permissions

Supervisor system for Claude Code that intercepts and reviews tool actions using an OpenRouter LLM.

## Overview

This library adds an automated permission layer to [Claude Code (Agent SDK)](https://github.com/anthropics/claude-code). A supervisor LLM reviews every tool call before execution and can automatically answer questions, creating a fully autonomous coding agent with configurable safety policies.

## Key Features

- **Automated Permission Management**: PreToolUse hooks intercept and review all tool calls
- **Auto-Answering**: Supervisor automatically answers `AskUserQuestion` prompts
- **OpenRouter Integration**: Use any LLM as supervisor (Grok, Claude, GPT-4, etc.)
- **Customizable Policies**: Define supervision behavior via system prompts
- **Zero Manual Intervention**: Fully autonomous after configuration

## Installation

```bash
# From source
git clone https://github.com/smarsx/dangerously-prompt-permissions.git
cd dangerously-prompt-permissions
pip install -e .
```

## Quick Start

Get API keys from [OpenRouter](https://openrouter.ai/) and [Anthropic](https://console.anthropic.com/), then set environment variables:

```bash
export OPENROUTER_API_KEY='your-openrouter-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
```

Then use the library:

```python
import asyncio
from dangerously_prompt_permissions import OpenRouterManager

async def main():
    # Uses default models: Grok (supervisor) + Claude Haiku (worker)
    manager = OpenRouterManager(
        manager_policy="""You are a security-focused code supervisor.

        Review all tool calls and:
        - ALLOW safe file reads and harmless operations
        - DENY destructive operations without explicit user consent
        - DENY commands that could expose secrets
        - For AskUserQuestion, choose the most secure default
        """
    )

    await manager.run(
        code_prompt="Create a Python script that prints 'Hello World'",
        root_dir="/path/to/workspace"
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## How It Works

1. **Worker makes tool call** → Edit file, run bash command, etc.
2. **PreToolUse hook intercepts** → Before execution
3. **Supervisor LLM reviews** → Analyzes intent and safety
4. **Decision applied** → ALLOW (execute), DENY (block), or auto-answer
5. **Execution continues** → Worker receives result or error

For `AskUserQuestion` calls, the supervisor automatically selects appropriate answers based on the policy, enabling fully autonomous operation.

## API Reference

### OpenRouterManager

```python
OpenRouterManager(
    manager_model: str = "x-ai/grok-code-fast-1",  # Supervisor model
    manager_policy: str = "",
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions",
    verbose: bool = False
)
```

**Default Models:**
- **Supervisor (Manager)**: `x-ai/grok-code-fast-1` - Fast, cost-effective code review
- **Worker (Agent)**: `claude-haiku-4-5` - Fast Claude model for code execution

**Environment Variables (Required):**
- `OPENROUTER_API_KEY` - Get at https://openrouter.ai/
- `ANTHROPIC_API_KEY` - Get at https://console.anthropic.com/

### run()

```python
await manager.run(
    code_prompt: str,                # Task description for worker
    root_dir: str,                   # Workspace directory
    model: str = "claude-haiku-4-5", # Worker model
    system_prompt: str = None,       # Custom worker instructions
    permission_mode: str = "default",
    setting_sources: list = ["project"]
)
```

## Security Considerations

**Important**: The supervisor LLM is not infallible. This is a research/development tool, not a production security solution.

- Supervisor can make mistakes or be misled by clever prompts
- Always review automated decisions in sensitive contexts
- Use restrictive policies by default
- Monitor logs for unexpected behavior
- Test policies thoroughly before autonomous operation

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on [Claude Code (Agent SDK)](https://github.com/anthropics/claude-code) by Anthropic
- Uses [OpenRouter](https://openrouter.ai/) for LLM access