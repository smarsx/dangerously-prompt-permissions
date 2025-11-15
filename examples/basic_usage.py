"""
Basic usage example for dangerously-prompt-permissions.

Set environment variables before running:
    export OPENROUTER_API_KEY='your-openrouter-key'
    export ANTHROPIC_API_KEY='your-anthropic-key'

Get keys from:
- https://openrouter.ai/
- https://console.anthropic.com/
"""

import asyncio
from pathlib import Path

from dangerously_prompt_permissions import OpenRouterManager


async def main():
    manager_policy = """You are a security-focused code supervisor.

    Review all tool calls carefully and:
    - ALLOW safe file reads and harmless operations
    - ALLOW file writes to the workspace directory
    - DENY any destructive operations (rm, deletion, etc.)
    - DENY commands that could expose secrets or sensitive data
    - DENY network operations unless explicitly needed
    - For AskUserQuestion, choose the most secure and conservative option

    Be permissive with safe operations but strict with risky ones.
    When in doubt, choose security over convenience.
    """

    # Uses defaults: x-ai/grok-code-fast-1 (supervisor) + claude-haiku-4-5 (worker)
    manager = OpenRouterManager(
        manager_policy=manager_policy,
        verbose=True
    )

    workspace = Path("/tmp/claude-workspace")
    workspace.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("Starting supervised Claude Code session")
    print("="*60 + "\n")

    await manager.run(
        code_prompt="""Create a simple Python script called 'hello.py' that:
        1. Prints 'Hello, World!'
        2. Includes a docstring
        3. Has proper error handling

        After creating it, read the file to verify it was created correctly.
        """,
        root_dir=str(workspace),
        model="claude-haiku-4-5",  # Worker model (faster/cheaper)
    )

    print("\n" + "="*60)
    print("Session completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
