from dataclasses import dataclass
from typing import List, Optional
from json_repair import loads

from vero.core.agent import Agent
from vero.core.chat_openai import ChatOpenAI
from vero.tool import Tool
from vero.core.message import Message


@dataclass
class ReflectionResult:
    """
    Structured output from the critic's reflection step.

    Attributes:
        is_sufficient: Whether the response adequately answers the user task.
        feedback: A must-fix issue description when ``is_sufficient`` is False,
            otherwise an empty string.
    """

    is_sufficient: bool
    feedback: str


DEFAULT_GENERATE_PROMPT = """You are a helpful assistant. Answer the user task as accurately as possible.
Output only the answer to the user task.

## Conversation History
{history}
"""


DEFAULT_REFLECT_PROMPT = """You are a strict critic. Evaluate whether the response sufficiently answers the user task.

## Conversation History
{history}

## Evaluation criteria for `is_sufficient`
Set `is_sufficient` to true if the response correctly answers what the user asked, even if it could be phrased better or include extra detail.
Set `is_sufficient` to false only if:
- The response contains factual errors
- The response does not actually answer the user task
- A piece of information critical to answering the task is missing

## Output Format
{{
    "thinking": "<step-by-step reasoning to verify the response against the task requirements>",
    "is_sufficient": <true or false>,
    "feedback": "<describe the must-fix issue if is_sufficient is false, otherwise empty string>"
}}
"""


DEFAULT_REVISE_PROMPT = """You are a helpful assistant.
Revise your previous response using the critic's feedback below.
Produce only the improved response with no additional commentary.

## Conversation History
{history}

## Previous Response
{response}

## Feedback
{feedback}
"""


class BasicReflectionAgent(Agent):
    """
    Basic Reflection agent.

    A lightweight generate-reflect-revise loop using a pure LLM with no tools:

    1. **Generate** — the LLM produces an initial answer to the user request.
    2. **Reflect**  — a critic LLM evaluates the answer and returns a
       ``ReflectionResult`` indicating whether the answer is sufficient and,
       if not, what must be fixed.
    3. **Revise**   — the LLM rewrites the answer guided by the critic's feedback.

    Steps 2–3 repeat up to ``max_turns`` times. The loop exits early when the
    critic marks the answer as sufficient.

    This agent is best suited for tasks that do not depend on real-time or
    external information (e.g. text refinement, reasoning, code review).
    For tasks requiring up-to-date facts, use a tool-enabled agent such as
    ReWOO or LLMCompiler instead.

    Unlike ``ReflexionAgent``, this agent has no episodic memory across separate
    ``run`` calls — it only uses the standard conversation history (``_history``)
    that accumulates over multi-turn interactions.
    """

    def __init__(
        self,
        name: str,
        llm: ChatOpenAI,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_turns: int = 2,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the Basic Reflection agent.

        Args:
            name: Human-readable agent identifier.
            llm: LLM used for generation, reflection, and revision.
            tools: Unused; accepted for interface compatibility only.
            system_prompt: Unused; accepted for interface compatibility only.
            max_turns: Maximum number of reflect-revise iterations.
                A value of 1 means one reflection pass followed by at most one revision.
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

        self.log(f"🚀 Initializing BasicReflectionAgent `{name}` ...")

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------
    def run(self, user_input: str, **kwargs) -> str:
        """
        Execute the generate-reflect-revise loop for one user request.

        The control flow is:
            1. Generate an initial answer via the LLM.
            2. Ask the critic to reflect on the answer.
            3a. Sufficient → early exit with the current answer.
            3b. Not sufficient → revise the answer and repeat from step 2.
            4. Return the best answer after ``max_turns`` iterations.

        Args:
            user_input: Raw user request.
            **kwargs: Reserved for future execution options.

        Returns:
            Final answer text after reflection and revision.
        """
        self.log("\n==============================")
        self.log(f"👤 User Input: {user_input}")
        self.log("==============================\n")

        # 1. Generate initial answer
        answer = self._generate(user_input)

        for turn_idx in range(1, self.max_turns + 1):
            self.log(f"🔁 Reflection turn {turn_idx}/{self.max_turns}")

            # 2. Reflect on the current answer
            reflection = self._reflect(user_input, answer)

            if reflection.is_sufficient:
                # Answer sufficiently addresses the user task — exit early
                self.log("✅ Answer accepted as sufficient.")
                break

            self.log(f"💬 Feedback: {reflection.feedback}")

            # 3. Revise the answer using the feedback
            answer = self._revise(user_input, answer, reflection.feedback)

        # Persist the conversational boundary: user input and final answer
        self.add_message(Message.user(user_input))
        self.add_message(Message.assistant(answer))
        self.log(f"📤 Final Answer:\n{answer}\n")
        return answer

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    def _generate(self, user_input: str) -> str:
        """
        Produce the initial answer using the LLM.

        Args:
            user_input: The original user task.

        Returns:
            Initial answer string.
        """
        self.log("⚙️ Generating initial answer...")

        messages = [
            Message.system(
                DEFAULT_GENERATE_PROMPT.format(history=self.format_history())
            ),
            Message.user(user_input),
        ]

        result = self.llm.generate(messages).content or ""
        self.log(f"📝 Initial answer:\n{result}\n")
        return result

    # ------------------------------------------------------------------
    # Reflect
    # ------------------------------------------------------------------
    def _reflect(self, user_input: str, response: str) -> ReflectionResult:
        """
        Ask the critic LLM to evaluate the current answer.

        The critic returns a JSON object with two fields:
            - ``is_sufficient``: whether the response adequately answers the task.
            - ``feedback``: a must-fix issue description when ``is_sufficient`` is false.

        The loop exits as soon as ``is_sufficient`` is true. Any parse failure
        defaults to ``ReflectionResult(True, "")`` to avoid stalling the loop.

        Args:
            user_input: The original user task.
            response: The current answer to be critiqued.

        Returns:
            A ``ReflectionResult`` with the critic's decision.
        """
        self.log("🔍 Reflecting on answer...")

        messages = [
            Message.system(
                DEFAULT_REFLECT_PROMPT.format(history=self.format_history())
            ),
            Message.user(f"## User Task\n{user_input}\n\n## Response\n{response}"),
        ]

        raw = self.llm.generate(messages).content
        self.log(f"📤 Critic raw output: {raw}")

        try:
            parsed = loads(raw)
            thinking = parsed.get("thinking", "")
            if thinking:
                self.log(f"💭 Thinking:\n{thinking}")
            result = ReflectionResult(
                is_sufficient=bool(parsed.get("is_sufficient", True)),
                feedback=parsed.get("feedback", ""),
            )
            self.log(
                f"{'✅' if result.is_sufficient else '❌'} is_sufficient={result.is_sufficient}"
            )
            return result
        except Exception:
            # Treat unparseable output as sufficient to avoid blocking the loop.
            self.log("⚠️ Failed to parse critic output. Treating as sufficient.")
            return ReflectionResult(is_sufficient=True, feedback="")

    # ------------------------------------------------------------------
    # Revise
    # ------------------------------------------------------------------
    def _revise(self, user_input: str, response: str, feedback: str) -> str:
        """
        Rewrite the current answer guided by the critic's feedback.

        Args:
            user_input: The original user task.
            response: The current answer to be revised.
            feedback: Critique produced by ``_reflect``.

        Returns:
            Revised answer string.
        """
        self.log("✏️ Revising answer based on feedback...")

        messages = [
            Message.system(
                DEFAULT_REVISE_PROMPT.format(
                    history=self.format_history(),
                    response=response,
                    feedback=feedback,
                )
            ),
            Message.user(user_input),
        ]

        result = self.llm.generate(messages).content or ""
        self.log(f"📝 Revised answer:\n{result}\n")
        return result
