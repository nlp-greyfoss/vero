import ast
from typing import Any, Optional, List

from json_repair import loads

from vero.tool import tool, Tool
from vero.core.exceptions import ToolCallError
from .message import Message
from .chat_openai import ChatOpenAI


class LLMMixin:
    """
    Mixin that injects an ``llm_tool`` into the agent's tool list.

    This allows planner-style agents (e.g. ReWOO, LLMCompiler) to use the LLM
    itself as a tool — typically for extracting or normalising facts from noisy
    evidence within a larger tool-calling pipeline.
    """

    def __init__(
        self, name: str, llm: ChatOpenAI, tools: Optional[List[Tool]] = None, **kwargs
    ):
        """
        Initialize the LLM tool and prepend it to the tools list.

        Args:
            name: The name of the agent.
            llm: An instance of the LLM (ChatOpenAI).
            tools: The list of existing tools. ``llm_tool`` will be appended.
            **kwargs: Any additional arguments forwarded to the Agent base class.
        """
        tools = tools or []

        if llm:
            llm_tool = self._create_llm_tool(llm)
            tools.append(llm_tool)
            # log is available after super().__init__, so use print here.
            print(f"🔧 LLMMixin: Added LLM tool to tools list: {tools}.")

        super().__init__(name=name, llm=llm, tools=tools, **kwargs)

    def _create_llm_tool(self, llm: ChatOpenAI) -> Tool:
        """
        Create an ``llm_tool`` wrapping the provided ChatOpenAI instance.

        Args:
            llm: The LLM instance to be used by the tool.

        Returns:
            A ``Tool`` instance that accepts a prompt string and returns the
            LLM-generated response.
        """

        @tool
        def llm_tool(prompt: str) -> str:
            """
            Use the LLM to process a prompt and return the generated response.

            Args:
                prompt: The input prompt for the LLM.

            Returns:
                str: The generated response.
            """
            messages = [Message.user(prompt)]
            response = llm.generate(messages)
            return response.content or ""

        return llm_tool


class ToolInvocationMixin:
    """
    Mixin that provides generic tool-input parsing and tool-invocation helpers.

    Agents that need to invoke tools from free-form planner output (e.g. ReWOO,
    LLMCompiler, CRITIC) can inherit this mixin to share a single, well-tested
    implementation of ``_parse_tool_input`` and ``_invoke_tool`` rather than
    duplicating the logic in each agent.
    """

    def _parse_tool_input(self, tool_input: Any) -> Any:
        """
        Parse a raw tool input into a structured Python value when possible.

        Planner outputs often produce:
            - plain strings  → returned as-is
            - JSON objects   → parsed via ``json_repair``
            - Python literals (dicts/lists) → parsed via ``ast.literal_eval``

        The parser tries JSON repair first, then ``ast.literal_eval``, and falls
        back to the raw string when the input should be treated as plain text.

        Args:
            tool_input: Raw tool input, typically a string from the planner.

        Returns:
            A parsed ``dict`` or ``list`` when structured input is detected,
            otherwise the original string, or an empty dict for empty input.
        """
        if not tool_input:
            return {}

        if not isinstance(tool_input, str):
            return tool_input

        stripped = tool_input.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = loads(stripped)
                return parsed
            except Exception:
                try:
                    return ast.literal_eval(stripped)
                except Exception:
                    return tool_input

        return tool_input

    def _invoke_tool(self, tool: Tool, parsed_input: Any) -> str:
        """
        Invoke a tool with the most suitable calling convention.

        Supported conventions:
            - ``dict``   → keyword arguments
            - ``list``   → positional arguments
            - scalar/str → single raw-string argument when the tool shape allows

        For scalar/string inputs, the tool signature is inspected to ensure it
        does not require more than one argument. If multiple required parameters
        are detected, a ``ToolCallError`` is raised rather than passing a
        potentially mismatched string.

        ``parsed_input`` is guaranteed to be the original raw string when
        ``_parse_tool_input`` cannot parse it as structured data, so no
        separate ``raw_tool_input`` fallback is needed.

        Args:
            tool: The tool instance to invoke.
            parsed_input: Structured input produced by ``_parse_tool_input``;
                falls back to the original raw string for plain-text inputs.

        Returns:
            Tool execution result as a string.

        Raises:
            ToolCallError: If a multi-argument tool receives only an unstructured
                string input.
        """
        if isinstance(parsed_input, dict):
            return tool(**parsed_input)

        if isinstance(parsed_input, list):
            return tool(*parsed_input)

        non_self_params = [
            param
            for param in tool.signature.parameters.values()
            if param.name != "self"
        ]
        required_params = [
            param
            for param in non_self_params
            if param.default is param.empty
            and param.kind
            in (
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            )
        ]

        if len(non_self_params) <= 1 or len(required_params) <= 1:
            return tool(parsed_input)

        raise ToolCallError(
            f"Tool `{tool.name}` expects structured parameters, got: {parsed_input}"
        )
