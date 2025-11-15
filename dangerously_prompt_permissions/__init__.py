"""
dangerously-prompt-permissions
===============================

A general-purpose manager system that sits on top of Claude Code (Agent SDK)
to intercept and supervise tool actions using an OpenRouter LLM.

Default models: Grok (supervisor) + Claude Haiku (worker)

Example usage:
    ```python
    from dangerously_prompt_permissions import OpenRouterManager

    # Set environment variables first:
    # export OPENROUTER_API_KEY='your-key'
    # export ANTHROPIC_API_KEY='your-key'

    manager = OpenRouterManager(
        manager_policy="You are a security-focused supervisor..."
    )

    await manager.run(
        code_prompt="Implement a login feature",
        root_dir="/path/to/project"
    )
    ```
"""

from .manager import (
    OpenRouterManager,
    OpenRouterClient,
    Logger,
    create_custom_tools_server,
    make_askuser_hook,
    make_pretool_hook,
)

__version__ = "0.1.0"

__all__ = [
    "OpenRouterManager",
    "OpenRouterClient",
    "Logger",
    "create_custom_tools_server",
    "make_askuser_hook",
    "make_pretool_hook",
    "__version__",
]
