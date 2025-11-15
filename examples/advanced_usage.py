"""
Advanced usage example demonstrating custom policies, permission modes, and complex workflows.

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
    dev_policy = """You are a helpful development supervisor.

    Your goal is to facilitate rapid development while maintaining basic safety:
    - ALLOW most file operations within the workspace
    - ALLOW common development commands (git, npm, pip, etc.)
    - DENY operations outside the workspace directory
    - DENY commands that modify system settings
    - DENY operations on sensitive files (.env, credentials, keys)
    - For questions, prefer pragmatic choices that speed up development

    Be helpful and permissive for standard development workflows.
    """

    # Use more powerful supervisor for complex decisions
    manager = OpenRouterManager(
        manager_model="anthropic/claude-3.5-sonnet",  # Override default Grok
        manager_policy=dev_policy,
        verbose=True
    )

    workspace = Path("/tmp/claude-advanced-workspace")
    workspace.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("TASK 1: Create a multi-file Python package")
    print("="*70 + "\n")

    await manager.run(
        code_prompt="""Create a simple Python package called 'mathutils' with:
        1. A package directory structure (mathutils/__init__.py)
        2. A module mathutils/operations.py with add, subtract, multiply functions
        3. A test file tests/test_operations.py with basic tests
        4. A setup.py for packaging
        5. A README.md explaining the package

        After creating all files, list the directory structure to verify.
        """,
        root_dir=str(workspace),
        model="claude-sonnet-4",
    )

    print("\n" + "="*70)
    print("TASK 2: Review and refactor existing code")
    print("="*70 + "\n")

    await manager.run(
        code_prompt="""Review the mathutils package you just created and:
        1. Add type hints to all functions
        2. Add docstrings in Google style
        3. Add input validation with proper exceptions
        4. Create a simple example.py showing usage

        Use AskUserQuestion to ask about any design choices.
        """,
        root_dir=str(workspace),
        model="claude-sonnet-4",
    )

    print("\n" + "="*70)
    print("TASK 3: Run tests and verify functionality")
    print("="*70 + "\n")

    await manager.run(
        code_prompt="""In the mathutils project:
        1. Run the test suite using pytest
        2. If any tests fail, fix the issues
        3. Add at least 2 more test cases for edge cases
        4. Generate a coverage report

        Report the final test results and coverage percentage.
        """,
        root_dir=str(workspace),
        model="claude-haiku-4-5",
        permission_mode="acceptEdits",
    )

    print("\n" + "="*70)
    print("All tasks completed successfully!")
    print(f"Project created at: {workspace}")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
