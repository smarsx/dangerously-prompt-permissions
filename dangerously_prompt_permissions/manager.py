"""
OpenRouter Manager for Claude Code - supervises tool actions using an LLM.

Provides: Logger, OpenRouter client, PreToolUse hooks, custom tools, and orchestration.
"""

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import orjson
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    tool,
    create_sdk_mcp_server,
    PermissionMode
)


class Logger:
    """Rich console logger with consistent formatting."""

    def __init__(self, console: Console, verbose: bool = False, width: int = 100):
        self.c = console
        self.verbose = verbose
        self.width = width

    def _truncate(self, s: str, limit: int) -> str:
        """Truncate string to limit with ellipsis."""
        if s is None:
            return ""
        return s if len(s) <= limit else s[:limit] + "…"

    def _section(self, title: str):
        """Print a section header with a horizontal rule."""
        self.c.print(Rule(Text(title, style="bold"), characters="─"))

    def header(self, title: str, subtitle: Optional[str] = None):
        """Print a main header in a panel."""
        header = f"{title}\n{subtitle}" if subtitle else title
        self.c.print(Panel.fit(header, title=" ", border_style="grey50"))

    def info(self, msg: str):
        """Print an info message with a bullet."""
        self.c.print(Text("• ", style="grey70") + Text(msg))

    def success(self, msg: str):
        """Print a success message with a checkmark."""
        self.c.print(Text("✓ ", style="green") + Text(msg))

    def warn(self, msg: str):
        """Print a warning message."""
        self.c.print(Text("! ", style="yellow") + Text(msg))

    def error(self, msg: str):
        """Print an error message."""
        self.c.print(Text("✗ ", style="red") + Text(msg))

    def kv(self, key: str, val: str):
        """Print a key-value pair."""
        self.c.print(Text(f"{key}: ", style="grey70") + Text(val))

    def json(self, data: Any, label: Optional[str] = None, truncate_after: int = 4000):
        """Print JSON data in a panel."""
        try:
            text = orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
        except Exception:
            text = str(data)
        text = self._truncate(text, truncate_after)
        if label:
            self.kv(label, "")
        self.c.print(Panel(text, border_style="grey58"))

    def block(self, label: str, body: str, limit: int = 1000):
        """Print a labeled text block."""
        self._section(label)
        self.c.print(self._truncate(body, limit))


class OpenRouterClient:
    """Client for calling OpenRouter API."""

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        logger: Optional[Logger] = None
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.logger = logger or Logger(Console())

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2
    ) -> str:
        """
        Send a completion request to OpenRouter.

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature

        Returns:
            The model's response text
        """
        self._log_request(prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(self.base_url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            response = data["choices"][0]["message"]["content"].strip()
            self._log_response(response)
            return response

    def _log_request(self, prompt: str):
        """Log the request to the manager LLM."""
        display_prompt = prompt if len(prompt) <= 600 else prompt[:600] + "…"
        self.logger.block("Manager LLM • Request", display_prompt, limit=600)

    def _log_response(self, response: str):
        """Log the response from the manager LLM."""
        self.logger.block("Manager LLM • Response", response, limit=600)


@tool(
    "AskUserQuestion",
    """Ask the user multiple-choice questions. Provide an array of questions, each with:
- question: The question text
- header: Short label (max 12 chars)
- options: Array of choices with 'label' and 'description'
- multiSelect: Whether multiple options can be selected (default false)

Example input:
{
  "questions": [
    {
      "question": "Which approach do you prefer?",
      "header": "Approach",
      "multiSelect": false,
      "options": [
        {"label": "Recursive", "description": "Simple but slow"},
        {"label": "Iterative", "description": "Fast and efficient"}
      ]
    }
  ]
}""",
    {
        "questions": {
            "type": "array",
            "description": "Array of question objects",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask"},
                    "header": {"type": "string", "description": "Short label (max 12 chars)"},
                    "multiSelect": {"type": "boolean", "description": "Allow multiple selections"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["label", "description"]
                        }
                    }
                },
                "required": ["question", "header", "options"]
            }
        }
    }
)
async def ask_user_question_tool(args):
    """Custom AskUserQuestion tool - returns answers populated by PreToolUse hook."""
    answers = args.get("answers", {})

    if not answers:
        return {
            "content": [{
                "type": "text",
                "text": "Error: No answers were provided by the hook"
            }]
        }

    answers_text = "\n".join([f"{key}: {value}" for key, value in answers.items()])
    return {
        "content": [{
            "type": "text",
            "text": f"User responses:\n{answers_text}"
        }]
    }


def create_custom_tools_server():
    """Create an MCP server with custom tools like AskUserQuestion."""
    return create_sdk_mcp_server(
        name="custom-tools",
        version="1.0.0",
        tools=[ask_user_question_tool]
    )


def make_askuser_hook(openrouter_client: OpenRouterClient, manager_policy: str, logger: Logger):
    """Create PreToolUse hook that auto-answers AskUserQuestion via manager LLM."""
    async def askuser_hook(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {}) or {}

        if not isinstance(tool_input, dict):
            logger.error("AskUserQuestion called with invalid input format")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "AskUserQuestion called with invalid input format",
                }
            }

        questions = tool_input.get("questions", [])

        if isinstance(questions, str):
            try:
                questions = json.loads(questions)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse questions JSON: {e}")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "Failed to parse questions JSON",
                    }
                }

        logger._section("Manager • Answering AskUserQuestion")
        for i, q in enumerate(questions):
            question_text = q.get("question", "")
            logger.info(f"Q{i+1}: {question_text}")

        questions_text = []
        for i, q in enumerate(questions):
            header = q.get("header", f"Q{i+1}")
            question = q.get("question", "")
            multi_select = q.get("multiSelect", False)
            options = q.get("options", [])

            options_text = "\n".join([
                f"  {j+1}. {opt.get('label', '')}: {opt.get('description', '')}"
                for j, opt in enumerate(options)
            ])

            select_type = "multiple options" if multi_select else "one option"
            questions_text.append(
                f"Question {i+1} ({header}) - Select {select_type}:\n{question}\n{options_text}"
            )

        prompt = f"""You are answering questions on behalf of the user. Answer each question by selecting the most appropriate option(s).

{chr(10).join(questions_text)}

For each question, respond with ONLY the number(s) of your selected option(s). Format:
Q1: <number> (or <number>,<number> for multi-select)
Q2: <number>
etc.

Be concise and direct. Choose the most reasonable default options."""

        try:
            reply = await openrouter_client.complete(prompt, manager_policy, temperature=0.3)

            answers = {}
            for i, q in enumerate(questions):
                header = q.get("header", f"Q{i+1}")
                options = q.get("options", [])
                multi_select = q.get("multiSelect", False)

                pattern = rf"Q{i+1}:\s*(\d+(?:,\d+)*)"
                match = re.search(pattern, reply)

                if match:
                    selected_indices = [int(x.strip()) - 1 for x in match.group(1).split(",")]
                    selected_labels = []
                    for idx in selected_indices:
                        if 0 <= idx < len(options):
                            selected_labels.append(options[idx].get("label", ""))

                    if multi_select:
                        answers[header] = selected_labels
                    else:
                        answers[header] = selected_labels[0] if selected_labels else ""
                else:
                    if options:
                        answers[header] = [options[0].get("label", "")] if multi_select else options[0].get("label", "")

            logger.info("Selected answers")
            for key, value in answers.items():
                logger.kv(f"  {key}", f"{value}")

            updated_input = {**tool_input, "answers": answers}

            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": "Manager LLM answered questions automatically",
                    "updatedInput": updated_input,
                }
            }

        except Exception as e:
            logger.error(f"Error answering questions: {str(e)}")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Error answering questions: {str(e)}",
                }
            }

    return askuser_hook


def make_pretool_hook(openrouter_client: OpenRouterClient, manager_policy: str, logger: Logger):
    """Create PreToolUse hook that reviews tool calls via manager LLM."""
    async def pretool(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {}) or {}

        if tool_name and ("AskUserQuestion" in tool_name):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": "Skipping - handled by dedicated hook",
                }
            }

        logger._section(f"Manager • Review • {tool_name}")

        if tool_name.lower() == "bash":
            cmd_preview = tool_input.get('command', '')[:200]
            logger.kv("Command", f"{cmd_preview}{'…' if len(tool_input.get('command', '')) > 200 else ''}")
            prompt = f"""Review this bash command and reply ONLY ALLOW or DENY (or ASK if essential).
Command: {tool_input.get('command')}
CWD: {tool_input.get('cwd')}
If DENY, add a one-line correction on the next line."""
        elif tool_name.lower() in ["write", "edit"]:
            file_path = tool_input.get("file_path") or tool_input.get("path") or "unknown"
            logger.kv("File", file_path)
            prompt = f"""Review this file edit and reply ONLY ALLOW or DENY (or ASK if essential).
Patch JSON: {json.dumps(tool_input)}"""
        else:
            logger.json(tool_input, label="Tool args", truncate_after=200)
            prompt = f"""Review this tool call and reply ONLY ALLOW or DENY (or ASK if essential).
Tool: {tool_name}
Args: {json.dumps(tool_input, indent=2)}"""

        try:
            reply = (await openrouter_client.complete(prompt, manager_policy)).upper()
        except Exception as e:
            logger.error(f"Error asking manager: {str(e)}")
            reply = "ASK"

        decision = "allow" if reply.startswith("ALLOW") else "deny" if reply.startswith("DENY") else "deny"
        reason = "Auto: Manager said ALLOW" if decision == "allow" else f"Manager response: {reply}"

        if decision == "allow":
            logger.success("Manager Decision: ALLOW")
        else:
            logger.error("Manager Decision: DENY")

        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason[:500],
            }
        }

    return pretool


def log_worker_message(logger: Logger, content: str):
    """Log a worker message."""
    logger.block("Worker • Message", content, limit=1200)


def log_worker_thinking(logger: Logger, content: str):
    """Log worker thinking (only if verbose)."""
    if logger.verbose:
        logger.block("Worker • Thinking", content, limit=1000)


def log_worker_tool_use(logger: Logger, tool_name: str, tool_input: dict):
    """Log worker tool use with formatted output."""
    logger._section(f"Worker • Tool Use • {tool_name}")
    if tool_name.lower() == "bash":
        cmd = tool_input.get("command", "")
        logger.kv("Command", cmd)
        if tool_input.get("cwd"):
            logger.kv("CWD", str(tool_input.get("cwd")))
        if tool_input.get("description"):
            logger.kv("Note", str(tool_input.get("description")))
    elif tool_name.lower() in ["write", "edit"]:
        file_path = tool_input.get("file_path") or tool_input.get("path") or "unknown"
        logger.kv("File", file_path)
        if "content" in tool_input and isinstance(tool_input["content"], str):
            logger.block("Content", tool_input["content"], limit=400)
        if "old_string" in tool_input:
            logger.block("Old", tool_input["old_string"], limit=200)
            logger.block("New", tool_input.get("new_string", ""), limit=200)
    else:
        logger.json(tool_input, label="Args", truncate_after=800)


class OpenRouterManager:
    """Orchestrator that supervises Claude Code using an OpenRouter LLM."""

    def __init__(
        self,
        manager_model: str = "x-ai/grok-code-fast-1",
        manager_policy: str = "",
        openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions",
        verbose: bool = False
    ):
        """
        Initialize the OpenRouter manager.

        Args:
            manager_model: Model to use for supervision
            manager_policy: System prompt defining supervision policy
            openrouter_url: Base URL for OpenRouter API
            verbose: Enable verbose logging

        Environment Variables:
            OPENROUTER_API_KEY: Required. Get at https://openrouter.ai/
            ANTHROPIC_API_KEY: Required. Get at https://console.anthropic.com/
        """
        self.console = Console()
        self.logger = Logger(self.console, verbose=verbose)

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required.\n"
                "Get your key at: https://openrouter.ai/\n"
                "Set it with: export OPENROUTER_API_KEY='your-key'"
            )

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required.\n"
                "Get your key at: https://console.anthropic.com/\n"
                "Set it with: export ANTHROPIC_API_KEY='your-key'"
            )

        self.openrouter_client = OpenRouterClient(
            api_key=openrouter_api_key,
            model=manager_model,
            base_url=openrouter_url,
            logger=self.logger
        )

        self.manager_policy = manager_policy
        self.custom_tools_server = create_custom_tools_server()
        self.anthropic_api_key = anthropic_api_key

    async def run(
        self,
        code_prompt: str,
        root_dir: str,
        model: str = "claude-haiku-4-5",
        system_prompt: Optional[str] = None,
        permission_mode: PermissionMode = "default",
        setting_sources: Optional[list] = None
    ):
        """
        Run a supervised Claude Code session.

        Args:
            code_prompt: Task for the worker agent
            root_dir: Workspace directory
            model: Claude model to use
            system_prompt: Custom system prompt for the worker
            permission_mode: Permission mode ('default', 'acceptEdits', etc.)
            setting_sources: List of setting sources to load
        """
        os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

        self.logger.header(
            "Claude Code Orchestrator",
            f"Root: {root_dir}\nTask: {code_prompt[:200]}{'…' if len(code_prompt) > 200 else ''}"
        )

        if system_prompt is None:
            system_prompt = (
                "You are a coding worker supervised by an automated manager. "
                "When you have questions or need clarification, use AskUserQuestion - "
                "the manager will answer automatically. "
                "Prefer small, reversible steps and explain your plan first."
            )

        if setting_sources is None:
            setting_sources = ["project"]

        options = ClaudeAgentOptions(
            model=model,
            system_prompt=system_prompt,
            cwd=root_dir,
            mcp_servers={"custom-tools": self.custom_tools_server},
            permission_mode=permission_mode,
            setting_sources=setting_sources,
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="mcp__custom-tools__AskUserQuestion",
                        hooks=[make_askuser_hook(self.openrouter_client, self.manager_policy, self.logger)]
                    ),
                    HookMatcher(
                        matcher="AskUserQuestion",
                        hooks=[make_askuser_hook(self.openrouter_client, self.manager_policy, self.logger)]
                    ),
                    HookMatcher(
                        matcher="*",
                        hooks=[make_pretool_hook(self.openrouter_client, self.manager_policy, self.logger)]  # type: ignore
                    )
                ]
            },
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(code_prompt)

            async for msg in client.receive_response():
                try:
                    payload = json.loads(orjson.dumps(msg).decode())

                    subtype = payload.get("subtype")
                    content = payload.get("content", [])
                    has_model = "model" in payload

                    if subtype in ["init", "success"]:
                        continue

                    if isinstance(content, list) and content:
                        first_item = content[0] if content else {}

                        if "tool_use_id" in first_item and not has_model:
                            tool_use_id = first_item.get("tool_use_id")
                            result_content = first_item.get("content")
                            is_error = first_item.get("is_error", False)

                            self.logger._section("Tool Result")
                            if is_error:
                                self.logger.error(f"Error: {result_content}")
                            else:
                                preview = result_content if not isinstance(result_content, str) else result_content[:1000] + ("…" if isinstance(result_content, str) and len(result_content) > 1000 else "")
                                if isinstance(preview, (dict, list)):
                                    self.logger.json(preview, truncate_after=1000)
                                else:
                                    self.logger.info(str(preview))

                        elif "id" in first_item and "name" in first_item and "input" in first_item:
                            tool_name = first_item.get("name", "unknown")
                            tool_input = first_item.get("input", {})
                            log_worker_tool_use(self.logger, tool_name, tool_input)

                        elif "text" in first_item and has_model:
                            text_parts = []
                            thinking_parts = []

                            for item in content:
                                if isinstance(item, dict):
                                    if "text" in item:
                                        text_parts.append(item["text"])
                                    elif "thinking" in item:
                                        thinking_parts.append(item["thinking"])

                            if thinking_parts:
                                log_worker_thinking(self.logger, "\n".join(thinking_parts))

                            if text_parts:
                                log_worker_message(self.logger, "\n".join(text_parts))

                except Exception as e:
                    self.logger.warn(f"Error parsing SDK event: {e}")
                    if self.logger.verbose:
                        self.logger.block("Traceback", traceback.format_exc(), limit=2000)

        self.logger.success("Task completed")
