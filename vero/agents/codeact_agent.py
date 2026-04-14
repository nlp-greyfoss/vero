import builtins
import re
from typing import List, Optional

from vero.core.agent import Agent
from vero.core.chat_openai import ChatOpenAI
from vero.core.message import Message
from vero.tool import Tool
from vero.tool.buildin.python_repl import PythonREPL


DEFAULT_SYSTEM_PROMPT = """You are a CodeAct agent. You **must** solve tasks by writing and executing Python code.

## Available Functions
The following functions are pre-loaded in your Python environment — call them directly:

{tool_signatures}

## Output Format
Each turn, you may optionally express your reasoning before acting:

Thought: <your reasoning about what to do next>
```python
<your code here>
```

Or, when you have the final answer:

Thought: <optional reasoning>
FINAL ANSWER: <your answer here>

## Rules
1. **Always act through code.** Any time you need information or want to take action, write a ```python ... ``` code block.
2. **Batch as much as possible in one code block.** If you need multiple pieces of information, call all the relevant functions in a single code block rather than one per turn. Only split into a new turn when you genuinely need to see intermediate results before deciding what to do next.
3. **Parallelize independent calls.** When multiple function calls do not depend on each other's results, run them concurrently using `concurrent.futures.ThreadPoolExecutor` (already available via `import concurrent.futures`). This is always preferred over sequential calls for independent queries.
3. Variables and results **persist across turns** — build on prior results without re-fetching.
4. Use `print()` to surface values you want to inspect; all stdout is returned as the Observation.
5. If code raises an error, read the traceback in the Observation and fix it in the next turn.
6. Do not redefine the pre-loaded functions — they are already callable in your environment.
7. Every turn must contain **either** a ```python ... ``` code block **or** a FINAL ANSWER — never neither.
8. **Verify side-effects.** After creating/writing files, try to read them back or check their properties to confirm the operation actually worked as expected.
"""

# Matches ```python ... ``` or ``` ... ```, tolerates missing closing fence
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)(?:```|$)", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\n```|\nFINAL ANSWER:|$)", re.DOTALL)


def _build_tool_signatures(tools) -> str:
    """Render tools as Python function stubs for the system prompt."""
    blocks = []
    for tool in tools:
        args = ", ".join(f"{name}: {typ}" for (name, typ, *_rest) in tool.arguments)
        summary = tool.description.splitlines()[0].rstrip(".")
        blocks.append(
            f"def {tool.name}({args}) -> str:\n" f'    """{summary}"""\n' f"    ..."
        )
    return "\n\n".join(blocks)


class CodeActAgent(Agent):
    """
    CodeAct agent: Python code as the sole action space.

    Unlike ReAct (text actions + JSON) or OpenAI function-calling agents,
    CodeActAgent instructs the LLM to write executable Python code each turn.
    All registered tools are injected into the REPL namespace as plain Python
    callables, so the LLM can call, compose, and chain them freely — and store
    intermediate results in variables that persist across turns.

    High-level flow:
        1. **Generate** — LLM produces a ``python`` code block.
        2. **Execute**  — code runs in the persistent REPL; stdout / traceback
           is captured as the Observation.
        3. **Feed back** — Observation is appended as the next user message.
        4. Repeat until the LLM emits ``FINAL ANSWER: <answer>`` or
           ``max_turns`` is exhausted.

    Reference: Wang et al., "Executable Code Actions Elicit Better LLM Agents",
    ICML 2024. https://arxiv.org/abs/2402.01030
    """

    def __init__(
        self,
        name: str,
        llm: ChatOpenAI,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_turns: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the CodeAct agent.

        Args:
            name: Human-readable agent identifier.
            llm: LLM used for code generation.
            tools: Tools exposed in the REPL namespace as Python functions.
                Each tool is callable by its ``tool.name``.
            system_prompt: Custom system prompt override.
            max_turns: Maximum code-execute iterations before returning the
                last LLM response.
            verbose: If True, log messages are printed to stdout.
        """
        super().__init__(
            name=name,
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            max_turns=max_turns,
            verbose=verbose,
        )

        self.log(f"🚀 Initializing CodeActAgent `{name}` ...")
        self.log(f"🛠️ Registered tools: {self.tools}")

        # Build a persistent REPL with tools injected as top-level callables.
        # Full builtins are available so the LLM can freely import modules,
        # use list comprehensions, define helper functions, etc.
        repl_globals: dict = {"__builtins__": builtins}
        for t in self.tools:
            repl_globals[t.name] = t.func
        self._repl = PythonREPL(_globals=repl_globals, _locals={})

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Execute the code-act loop for one user request.

        Args:
            user_input: Raw user query or task description.

        Returns:
            The answer extracted from the LLM's ``FINAL ANSWER`` marker,
            or the last LLM response when ``max_turns`` is reached.
        """
        self.log("\n==============================")
        self.log(f"👤 User Input: {user_input}")
        self.log("==============================\n")

        system_prompt = (self.system_prompt or DEFAULT_SYSTEM_PROMPT).format(
            tool_signatures=_build_tool_signatures(self.tools),
        )

        messages: List[Message] = [
            Message.system(system_prompt),
            Message.user(user_input),
        ]

        last_response = ""

        for turn_idx in range(1, self.max_turns + 1):
            self.log(f"\n🔄 Turn {turn_idx}/{self.max_turns}")

            response = self.llm.generate(messages)
            content = response.content or ""
            last_response = content

            thought = _extract_thought(content)
            if thought:
                self.log(f"💭 Thought: {thought}\n")

            # ── 1. Final answer? ──────────────────────────────────────
            final_answer = _extract_final_answer(content)
            if final_answer is not None:
                self.log(f"✅ Final Answer: {final_answer}\n")
                self._persist(user_input, final_answer)
                return final_answer

            # ── 2. Code block? ────────────────────────────────────────
            code = _extract_code(content)
            if code is None:
                # No code and no FINAL ANSWER — treat bare text as the answer.
                self.log(
                    f"⚠️ No code block found. Treating as final answer:\n{content}\n"
                )
                self._persist(user_input, content)
                return content

            # ── 3. Execute and feed observation back ──────────────────
            self.log(f"💻 Code:\n{code}\n")
            observation = self._repl.run(code)
            self.log(f"📦 Observation:\n{observation}\n")

            messages.append(Message.assistant(content))
            messages.append(Message.user(f"Observation:\n{observation}"))

        # Max turns exhausted.
        self.log("⚠️ Max turns reached. Returning last response.")
        self._persist(user_input, last_response)
        return last_response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist(self, user_input: str, answer: str) -> None:
        """Record the final user–assistant exchange in conversation history."""
        self.add_message(Message.user(user_input))
        self.add_message(Message.assistant(answer))


# ------------------------------------------------------------------
# Module-level pure helpers (no agent state needed)
# ------------------------------------------------------------------


def _extract_code(text: str) -> Optional[str]:
    """Return the first Python code block from LLM output, or ``None``."""
    match = _CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_final_answer(text: str) -> Optional[str]:
    """Return the answer following ``FINAL ANSWER:``, or ``None``."""
    match = _FINAL_ANSWER_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_thought(text: str) -> Optional[str]:
    """Return the content following ``Thought:``, or ``None``."""
    match = _THOUGHT_RE.search(text)
    return match.group(1).strip() if match else None
