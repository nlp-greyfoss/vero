from typing import Optional, List

from vero.tool import tool, Tool
from .message import Message
from .chat_openai import ChatOpenAI


class LLMMixin:
    """
    Mixin class to add LLM tool to an Agent.
    This class dynamically creates an LLM tool and adds it to the agent's tools list.
    """

    def __init__(
        self, name: str, llm: ChatOpenAI, tools: Optional[List[Tool]] = None, **kwargs
    ):
        """
        Initialize the LLM tool and add it to the tools list.

        Args:
            name: The name of the agent.
            llm: An instance of the LLM (ChatOpenAI).
            tools: The list of existing tools (LLM tool will be added to this list).
            **kwargs: Any additional arguments to pass to the Agent class.
        """
        tools = tools or []

        # Create LLM tool and add it to the tools list
        if llm:
            llm_tool = self._create_llm_tool(llm)
            tools.append(llm_tool)
            print(f"🔧 LLMMixin: Added LLM tool to tools list: {tools}.")

        # Initialize the base Agent class with the new tools list
        super().__init__(name=name, llm=llm, tools=tools, **kwargs)

    def _create_llm_tool(self, llm: ChatOpenAI) -> Tool:
        """
        Create an LLM tool based on the provided ChatOpenAI instance.

        Args:
            llm: The LLM instance to be used by the tool.

        Returns:
            Tool: A Tool instance encapsulating the LLM.
        """

        @tool
        def llm_tool(prompt: str) -> str:
            """
            The LLM tool that handles text generation based on the prompt.

            Args:
                prompt: The input prompt for the LLM.

            Returns:
                str: The generated response.
            """
            # Prepare the message for LLM
            messages = [Message.user(prompt)]
            response = llm.generate(messages)
            return response.content or ""

        return llm_tool
