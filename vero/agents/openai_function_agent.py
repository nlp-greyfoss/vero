import json
import time
from typing import Optional, List, Dict, Any

from vero.tool import Tool
from vero.core.message import Message
from vero.core.chat_openai import ChatOpenAI
from vero.core.agent import Agent
from vero.core.exceptions import ToolNotFoundError


class OpenAIFunctionAgent(Agent):
    """
    An Agent implementation that supports OpenAI-compatible Function Calling (tools).

    Features:
        - Supports multiple tool calls in a single assistant message
        - Automatically executes tools
        - Feeds tool outputs back to the model
        - Iterates until a final text answer is produced
    """

    def __init__(
        self,
        name: str,
        llm: ChatOpenAI,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_turns: int = 5,
        tool_choice: str = "auto",
        verbose: bool = True,
    ) -> None:
        """
        Initialize OpenAIFunctionAgent.

        Args:
            name: Human-readable agent identifier.
            llm: ChatOpenAI instance used for inference.
            tools: Optional list of Tool instances.
            system_prompt: Optional system prompt override.
            max_turns: Maximum number of reasoning / tool-execution loops.
            tool_choice: OpenAI tool_choice parameter ("auto", "none", or forced tool name).
            verbose: If True, internal log messages are printed to stdout.
                Defaults to True. Set to False to silence all agent-level output.
        """
        super().__init__(
            name=name,
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            max_turns=max_turns,
            verbose=verbose,
        )

        self.tool_choice = tool_choice
        self.tools_schema = self._build_tool_schemas()

        self.log(f"🚀 Initializing OpenAIFunctionAgent `{name}` ...")
        self.log(f"🛠️ Registered tools: {self.tools}")
        self.log(f"⚙️ Tool choice mode: {self.tool_choice}")

        # Initialize conversation with system prompt
        self.add_message(Message.system(self._build_system_prompt()))

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the agent.
        """
        return (
            self.system_prompt
            or "You are an intelligent agent capable of using external tools to help solve user queries."
        )

    # ------------------------------------------------------------------
    # Tool schemas
    # ------------------------------------------------------------------
    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Convert registered tools into OpenAI-compatible schemas.
        """
        if not self.tools:
            return []
        return [t.to_openai_schema() for t in self.tools]

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------
    def run(self, user_query: str) -> str:
        """
        Execute the agent reasoning loop.

        Steps:
            1. Append user message
            2. Call LLM with tool schemas
            3. If tool_calls exist:
                - Execute each tool
                - Inject tool results as Message.tool
            4. Repeat until a pure text response is produced
        """
        self.log("\n==============================")
        self.log(f"👤 User Input: {user_query}")
        self.log("==============================\n")

        self.add_message(Message.user(user_query))

        for turn_idx in range(1, self.max_turns + 1):
            self.log(f"🔁 Turn {turn_idx}/{self.max_turns}")

            assistant_msg: Message = self.llm.generate(
                messages=self._history,
                tools=self.tools_schema,
                tool_choice=self.tool_choice,
            )

            self.log(
                f"📤 LLM Assistant Message | "
                f"content={assistant_msg.content!r}, "
                f"tool_calls={bool(assistant_msg.tool_calls)}"
            )

            self.add_message(assistant_msg)

            # -------------------------------------------------
            # Case A: Final text response (no tool calls)
            # -------------------------------------------------
            if not assistant_msg.tool_calls:
                self.log("💬 No tool calls detected. Returning final answer.\n")
                return assistant_msg.content or ""

            # -------------------------------------------------
            # Case B: Tool calls detected
            # -------------------------------------------------
            for tc in assistant_msg.tool_calls:
                func = tc["function"]
                tool_name = func["name"]
                args_text = func["arguments"]
                tool_call_id = tc["id"]

                self.log(
                    f"🧩 Tool call detected → "
                    f"name={tool_name}, id={tool_call_id}, raw_args={args_text}"
                )

                # Parse arguments (OpenAI guarantees JSON string)
                try:
                    args = json.loads(args_text)
                    self.log("📦 Tool arguments parsed successfully.")
                except Exception as e:
                    self.log(f"❌ Failed to parse tool arguments: {e}")
                    args = {}

                # Lookup tool
                tool: Tool | None = self.tool_by_names.get(tool_name)
                if not tool:
                    self.log("❌ Tool not found!")
                    raise ToolNotFoundError(f"Unknown tool: {tool_name}")

                # Execute tool
                self.log(f"🔧 Executing tool `{tool_name}` with args={args}")
                try:
                    start = time.perf_counter()
                    output = tool(**args)
                    cost = time.perf_counter() - start
                    self.log(f"📦 Tool output: {output} | ⏱️ Cost: {cost:.3f}s")
                except Exception as e:
                    output = f"Tool execution failed: {e}"
                    self.log(f"💥 Tool execution failed: {e}")

                # Inject tool result back into history
                self.log("📥 Injecting tool result into conversation history.")
                self.add_message(
                    Message.tool(
                        content=str(output),
                        tool_call_id=tool_call_id,
                    )
                )

            # Continue loop, letting the model consume tool results

        raise RuntimeError("Reached max_turns without producing a final answer")
