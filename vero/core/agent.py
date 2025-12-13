from abc import ABC, abstractmethod
from typing import Optional, List

from vero.tool import Tool
from .message import Message
from .chat_openai import ChatOpenAI


class Agent(ABC):
    """
    Base class for an intelligent agent capable of interacting with a language model
    and optionally using external tools.

    This class is intentionally minimal and generic:
    - It does not dictate how tool calls are executed.
    - It does not parse tool-call formats.
    - It provides only shared structures and utilities.
    - Different agent subclasses may implement different tool invocation protocols
      (e.g., JSON function calling, custom TOOL_CALL strings, DSL-style calls, etc.).

    Subclasses must implement the `run` method, which defines the reasoning loop.
    """

    def __init__(
        self,
        name: str,
        llm: ChatOpenAI,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_turns: int = 3,
    ) -> None:
        """
        Initialize a generic agent.

        Args:
            name (str): Agent name identifier.
            llm (ChatOpenAI): Language model backend used for inference.
            tools (List[Tool] | None): Optional list of tools the agent can call.
            system_prompt (str | None): System-level instructions to prime the model.
            max_turns (int): Maximum reasoning turns allowed per execution.
        """
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.max_turns = max_turns
        self.system_prompt = system_prompt

        # Internal conversation history (Message objects)
        self._history: List[Message] = []

    # -------------------------------------------------------
    # Abstract API
    # -------------------------------------------------------
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        Execute the agent with user input.

        Subclasses fully control:
            - How the LLM is prompted
            - How messages are appended or rewritten
            - How tools are selected and executed
            - When the reasoning loop terminates

        Args:
            input_text (str): User message.

        Returns:
            str: Final output response.
        """
        pass

    # -------------------------------------------------------
    # Tool metadata helpers
    # -------------------------------------------------------
    @property
    def tool_descriptions(self) -> str:
        """
        Return a readable list of all tools in the agent.

        Format example:
            calculate_sum(a: int, b: int) - Add two numbers
            search_web(query: str) - Search the internet
        """
        lines = []
        for tool in self.tools:
            args_text = ", ".join(f"{name}: {typ}" for (name, typ, *_rest) in tool.arguments)
            lines.append(f"{tool.name}({args_text}) - {tool.description}")
        return "\n".join(lines)

    @property
    def tool_names(self) -> str:
        """Return a comma-separated list of tool names."""
        return ",".join([tool.name for tool in self.tools])

    @property
    def tool_by_names(self) -> dict[str, Tool]:
        """
        Return a dictionary mapping tool_name → Tool instance.
        Useful for subclasses implementing custom tool-invocation logic.
        """
        return {tool.name: tool for tool in self.tools}

    # -------------------------------------------------------
    # Conversation memory
    # -------------------------------------------------------
    def add_message(self, message: Message) -> None:
        """Append a message to the conversation history."""
        self._history.append(message)

    def clear_history(self) -> None:
        """Clear all stored conversation history."""
        self._history.clear()

    # -------------------------------------------------------
    # Representation
    # -------------------------------------------------------
    def __str__(self) -> str:
        return f"Agent<name={self.name}>"

    __repr__ = __str__
