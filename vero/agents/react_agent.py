import re
import json
from json_repair import loads
import ast

import time

from typing import List, Optional, Tuple, Dict

from vero.tool import Tool
from vero.core.message import Message
from vero.core.chat_openai import ChatOpenAI
from vero.core.agent import Agent
from vero.core.exceptions import ToolNotFoundError, ToolCallError


class ReActAgent(Agent):
    """
    The ReAct protocol follows an alternating reasoning and action cycle:
    - The LLM emits a "Thought" followed by an "Action" and its corresponding "Action Input".
    - The "Action" specifies which tool to use (or "Finish" to indicate the final answer).
    - The "Action Input" provides the parameters for the tool in the form of a JSON object.

    If no tool is required, the LLM provides a normal text response, and the "Action" would be "Finish".

    The agent follows this cycle:
    1. Append the user message to the conversation history.
    2. Ask the LLM for a reply.
    3. If the LLM outputs an "Action", parse and execute the corresponding tool.
    4. Inject the tool result into the conversation history and ask the LLM again to produce the final answer.
    """

    DEFAULT_SYSTEM_PROMPT = """
You are a ReAct-style agent.

You have access to the following tools:
{tool_descriptions}

Previous steps:
{scratchpad} 

(DO NOT repeat them, continue from here)

You reason step by step using the following loop:
Thought -> Action -> Action Input -> Observation

Only **one** Thought, Action, and Action Input should be produced at a time.
## Response Format (MUST be followed exactly):



Thought: <your reasoning for the next step>
Action: <one of the tools from {tool_names} OR Finish>
Action Input: <JSON object>

### Rules for Action Input:
- Action Input MUST be a valid JSON object.
- Use DOUBLE QUOTES for all keys and string values.
- DO NOT include any text outside the JSON object.
- DO NOT wrap the JSON in markdown or code blocks.
- If the Action is Finish, the Action Input MUST be:
  {{"answer": "<final answer>"}}

### Examples

Thought: I need to add two numbers.
Action: calculate_math_expression
Action Input: {{"expression": "12345 + 23456"}}
Obsercation: 35801

Thought: I have the final result.
Action: Finish
Action Input: {{"answer": "The result is 35801"}}

Now produce the NEXT step ONLY.
"""

    def __init__(
        self,
        name: str,
        llm: ChatOpenAI,
        tools: List[Tool],
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
        print(f"🚀 Initializing ReActAgent `{name}` ...")

        assert tools, "ReActAgent must have at least one tool."

        super().__init__(
            name=name,
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            max_turns=max_turns,
        )

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def _build_system_prompt(self, scratchpad: str) -> str:
        """
        Generate the system prompt with current scratchpad.
        """
        return self.system_prompt.format(
            tool_descriptions=self.tool_descriptions,
            scratchpad=scratchpad,
            tool_names=self.tool_names,
        )

    def _parse_react_step(self, text: str) -> Tuple[str, Dict]:
        """
        Parse a ReAct-style response.

        Expected format:
            Thought: ...
            Action: <tool_name | Finish>
            Action Input: <JSON>

        Returns:
            (action, action_input)

        Raises:
            ValueError if parsing fails.
        """
        print("🔍 Parsing ReAct output...")

        # 1. Extract Action
        action_match = re.search(r"^Action:\s*(.+)$", text, re.MULTILINE)
        if not action_match:
            raise ValueError("Missing Action field in ReAct output.")

        action = action_match.group(1).strip()
        print(f"🧩 Action detected: {action}")

        # 2. Extract Action Input
        input_match = re.search(
            r"^Action Input:\s*(\{.*\})\s*$",
            text,
            re.MULTILINE | re.DOTALL,
        )
        if not input_match:
            raise ValueError("Missing or invalid Action Input field.")

        raw_input = input_match.group(1).strip()
        print(f"📦 Raw Action Input: {raw_input}")

        # 3. Parse JSON strictly
        try:
            action_input = loads(raw_input)
        except Exception as e:
            raise ValueError(f"Action Input is not valid JSON: {e}")

        try:
            action_input = loads(raw_input)
            print("📦 Parameters parsed via JSON.")

        except Exception:
            try:
                action_input = ast.literal_eval(raw_input)
                print("📦 Parameters parsed via Python literal_eval.")
            except Exception:
                raise ValueError(f"Failed to parse parameters : {raw_input}")

        return action, action_input

    def run(self, user_input: str) -> str:
        """
        Execute the ReActAgent pipeline for a single user input.

        The method handles the following steps:
            1. Adds the user input to the conversation history.
            2. Iteratively interacts with the LLM to process the input, using the ReAct reasoning loop.
            3. Updates the system message to include the latest scratchpad (conversation context).
            4. Sends the conversation history to the LLM for a response.
            5. Parses the LLM's response to extract the "Action" and "Action Input".
            6. If the action is a tool call, it invokes the appropriate tool using _handle_tool_call.
            7. If the action is "Finish", it returns the final answer.
            8. If the maximum number of turns is reached, it returns the LLM's last response.

        Features:
            - Supports multiple turns of reasoning and acting (up to `max_turns`).
            - Records all steps (thoughts, actions, observations) in the scratchpad.
            - Executes tools when needed and handles tool errors.
            - Provides the final answer after the LLM produces a "Finish" action.
        """
        print(f"\n==============================")
        print(f"👤 User Input: {user_input}")
        print("==============================\n")

        # 1. Initialize empty scratchpad
        scratchpad = ""

        # 2. Append user input to _history
        self.add_message(Message.user(user_input))

        for turn_idx in range(1, self.max_turns + 1):
            print(f"🔁 Turn {turn_idx}/{self.max_turns}")

            # 3. Update system message with latest scratchpad
            system_prompt = self._build_system_prompt(scratchpad)

            if self._history and self._history[0].role == "system":
                self._history[0].content = system_prompt
            else:
                # fallback: ensure at least one system message exists
                self._history.insert(0, Message.system(system_prompt))

            # 4. Ask LLM
            assistant_msg: Message = self.llm.generate(self._history)
            print(f"📤 LLM Assistant Message:\n{assistant_msg.content}\n")

            content = assistant_msg.content or ""

            # 5. Parse Action / Action Input
            try:
                action, action_input = self._parse_react_step(content)
            except ValueError as e:
                print(f"❌ Parsing failed: {e}")
                observation = content

            thought_match = re.search(r"Thought:\s*(.+?)\s*Action:", content, re.DOTALL)
            thought_text = thought_match.group(1).strip() if thought_match else ""
            # 6.Check Finish
            if action.strip().lower() == "finish":
                final_answer = action_input.get("answer", content)
                self.add_message(Message.assistant(final_answer))
                print(f"✅ Finish detected. Returning final answer: {final_answer}")
                return final_answer

            # 7. Handle tool call
            print("🛠️ Tool call detected → dispatching tool handler.\n")
            try:
                observation = self._handle_tool_call(action, action_input)
            except (ToolNotFoundError, ToolCallError) as e:
                print(f"❌ Tool execution error: {e}")
                observation = str(e)

            # 8. Record observation into scratchpad
            scratchpad += f"""
    Thought: {thought_text}
    Action: {action}
    Action Input: {json.dumps(action_input)}
    Observation: {observation}
    """

        # 9. Max turns reached
        print("⚠️ Max turns reached. Returning last LLM response.")
        final_answer = assistant_msg.content or ""
        self.add_message(Message.assistant(final_answer))

        return final_answer

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

        tool = self.tool_by_names[tool_name]

        # Execute tool function and capture any runtime errors.
        print(f"🔧 Executing tool `{tool_name}` with params: {params}")
        try:
            start = time.perf_counter()
            result = tool(**params)
            print(
                f"📦 Tool result: {result} | ⏱️ Cost: {time.perf_counter() - start: .1f}s"
            )
            return result
        except Exception as e:
            print(f"💥 Tool execution failed: {e}")
            raise ToolCallError(f"Tool execution failed: {e}")
