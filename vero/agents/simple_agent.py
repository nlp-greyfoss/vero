import re
import ast
import json
import time

from typing import List, Optional, Tuple, Dict

from vero.tool import Tool
from vero.core.message import Message
from vero.core.chat_openai import ChatOpenAI
from vero.core.agent import Agent
from vero.core.exceptions import ToolNotFoundError, ToolCallError


class SimpleAgent(Agent):
    """
    A concrete Agent implementation that uses a simple TOOL_CALL protocol.

    Protocol:
        - The LLM should emit a one-line instruction when it decides a tool is required:
          TOOL_CALL:tool_name:{"param1": 1, "param2": "abc"}
        - Parameters must be a JSON object (or a Python-literal-compatible dict).
        - If no tool is needed, the LLM should reply with normal text.

    Behavior:
        1. Append the user message to the conversation history.
        2. Ask the LLM for a reply.
        3. If the LLM outputs a TOOL_CALL, parse and execute the tool.
        4. Inject the tool result into conversation history and ask the LLM again to produce the final answer.
    """

    TOOL_CALL_PATTERN = r"TOOL_CALL:(\w+):(.+)"

    DEFAULT_PROMPT_WITHOUT_TOOLS = (
        "You are a helpful and intelligent AI assistant. Answer the user concisely and accurately."
    )

    DEFAULT_PROMPT_WITH_TOOLS = """
You are an intelligent agent capable of using external tools to help solve user queries.

Below is the list of available tools:

{tool_descriptions}

When you decide that using a tool is necessary:
- Use the exact format:
  TOOL_CALL:tool_name:{{"param1": 1, "param2": "abc"}}
- The parameters must be a valid JSON object that includes all required arguments of the tool.
- If no tool is needed, simply respond with normal text.

Follow the format strictly. Do not explain the tool call. Do not wrap the tool call in code blocks.
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
        Initialize SimpleAgent.

        Args:
            name: Human-readable identifier for the agent.
            llm: ChatOpenAI instance used for model inference.
            tools: Optional list of Tool instances the agent can call.
            system_prompt: Optional system prompt override. If omitted, a prompt
                           is generated from the provided tools.
            max_turns: Reserved for future use (e.g., limit recursive tool calls).
        """
        print(f"🚀 Initializing SimpleAgent `{name}` ...")

        super().__init__(name=name, llm=llm, tools=tools, system_prompt=system_prompt, max_turns=max_turns)

        # Ensure there is a system prompt at the beginning of the conversation history.
        if self.system_prompt:
            sp = self.system_prompt
            print("📝 Using provided system prompt.")

        else:
            sp = self._build_system_prompt()
            print("🛠️ Generated system prompt from tool list.")


        # Use the base class API to append the initial system message.
        self.add_message(Message.system(sp))

    def _build_system_prompt(self) -> str:
        """Generate a reasonable system prompt based on whether tools are available."""
        if not self.tools:
            return self.DEFAULT_PROMPT_WITHOUT_TOOLS
        return self.DEFAULT_PROMPT_WITH_TOOLS.format(tool_descriptions=self.tool_descriptions)

    def _parse_tool_call(self, text: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Detect and parse a TOOL_CALL from model output.

        Returns:
            (has_call, tool_name, params_dict_or_None)

        Notes:
            - This tries JSON parsing first (preferred), then falls back to
              Python literal parsing (`ast.literal_eval`) to accept dict-like
              representations produced by some models.
            - If parsing fails, params will be None.
        """
        print("🔍 Parsing model output for TOOL_CALL ...")

        match = re.search(self.TOOL_CALL_PATTERN, text)
        if not match:
            print("❌ No TOOL_CALL detected.")
            return False, None, None

        tool_name = match.group(1)
        params_str = match.group(2).strip()

        print(f"🧩 TOOL_CALL detected → tool: `{tool_name}`, raw params: {params_str}")

        # Attempt to parse parameters into a dict.
        params = None
        try:
            params = json.loads(params_str)
            print("📦 Parameters parsed via JSON.")

        except Exception:
            try:
                params = ast.literal_eval(params_str)
                print("📦 Parameters parsed via Python literal_eval.")
            except Exception:
                print("❌ Failed to parse parameters.")
                params = None

        return True, tool_name, params

    def run(self, user_input: str) -> str:
        """
        Execute the agent pipeline for a single user input.

        Steps:
            1. Append the user's message to history.
            2. Query the LLM for a reply.
            3. If the reply requests a tool call, execute it and feed the result back into history.
            4. Return the final LLM answer (either the first reply if no tool used, or the post-tool final reply).
        """
        print(f"\n==============================")
        print(f"👤 User Input: {user_input}")
        print("==============================\n")
        # 1) append user input
        self.add_message(Message.user(user_input))

        # 2) ask LLM for reply (llm.generate accepts the history of Message objects)
        assistant_msg: Message = self.llm.generate(self._history)
        print(f"📤 LLM Assistant Message: {assistant_msg.content}\n")

        # record assistant's raw reply
        self.add_message(assistant_msg)

        # 3) parse for a tool call
        content = assistant_msg.content or ""
        has_call, tool_name, params = self._parse_tool_call(content)

        if has_call:
            print("🛠️ Tool call detected → dispatching tool handler.\n")
            return self._handle_tool_call(tool_name, params)

        # no tool requested → return the assistant reply directly
        print("💬 No tool requested → returning LLM reply.\n")
        return content

    # ------------ Internal Methods ------------ #

    def _handle_tool_call(self, tool_name: str, params: dict) -> str:
        """
        Execute a tool identified by `tool_name` with `params`, then ask the LLM
        to produce a final answer based on the tool result.

        Args:
            tool_name: Name of the tool to invoke.
            params: Parsed parameters (ideally a dict). If params is None or invalid,
                    a ToolCallError is raised.

        Returns:
            final_answer (str): The LLM's final answer after tool execution.

        Raises:
            ToolNotFoundError: If the requested tool is not registered.
            ToolCallError: If parameters are invalid or tool execution fails.
        """
        print(f"⚙️ Handling tool call for `{tool_name}` ...")

        # Validate tool existence
        if tool_name not in self.tool_by_names:
            print("❌ Tool not found!")
            raise ToolNotFoundError(f"Unknown tool: {tool_name}")

        if not isinstance(params, dict):
            print("❌ Invalid parameter format!")
            raise ToolCallError("Tool parameters must be provided as an object/dict.")

        tool = self.tool_by_names[tool_name]

        # Execute tool function and capture any runtime errors.
        print(f"🔧 Executing tool `{tool_name}` with params: {params}")
        try:
            start = time.perf_counter()
            result = tool(**params)
            print(f"📦 Tool result: {result} | ⏱️ Cost: {time.perf_counter() - start: .1f}s")
        except Exception as e:
            print(f"💥 Tool execution failed: {e}")
            raise ToolCallError(f"Tool execution failed: {e}")
        
        print("📥 Adding TOOL_RESULT message to history.")
        # Inject tool result back into the conversation.
        # NOTE: we use a user-style message ("TOOL_RESULT:...") to make it explicit
        # in the history that this is external evidence for the model to consume.
        # This keeps assistant-generated messages separate from tool outputs and
        # makes it easier to craft follow-up prompts like "Please answer the user using this tool result."
        self.add_message(Message.user(f"TOOL_RESULT:{result}"))

        # Ask LLM for final answer
        final_msg: Message = self.llm.generate(self._history)
        self.add_message(final_msg)

        return final_msg.content or ""
