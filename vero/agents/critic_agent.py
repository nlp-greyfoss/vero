import re
from dataclasses import dataclass
from typing import List, Optional
from datetime import date

from vero.core.agent import Agent
from vero.core.chat_openai import ChatOpenAI
from vero.core.message import Message
from vero.core.mixins import ToolInvocationMixin
from vero.tool import Tool
from vero.core.exceptions import ToolNotFoundError, ToolCallError


DEFAULT_GENERATE_PROMPT = """You are a helpful assistant. Answer the user query as accurately as possible.
Current date: {current_date}. Use this if needed.
Output only the answer.
"""


DEFAULT_VERIFY_PROMPT = """You are a verifier. Identify the single most critical factual claim about the answer that must be verified against the query requirements, then verify it with an external tool.

## Query
{query}

## Answer
{answer}

## Available Tools
{tool_descriptions}

Use the following format:

Claim: <the specific factual claim derived from the query requirements>
Action: <tool name>
Action Input: <tool input>
"""


DEFAULT_CRITIQUE_PROMPT = """You are a critic. Your sole job is to check whether the answer is consistent with the Verification Result below.
Current date: {current_date}.

## Query
{query}

## Answer
{answer}

## Verified Claim
{claim}

## Verification Result
{verification}

Treat the Verification Result as ground truth. Does the answer correctly and completely reflect it?
- If the answer claims ignorance or inability to answer, but the Verification Result contains the actual answer, that is a problem.
- If the answer contains factual claims that contradict the Verification Result, that is a problem.
- If yes (the answer is accurate and complete relative to the Verification Result), output exactly: No problem.
- If no, state only what conflicts with the Verification Result. Do not raise concerns unrelated to the verified claim.
"""


DEFAULT_CORRECT_PROMPT = """You are a helpful assistant. Correct the answer based on the critique and the verified fact below.
Treat the Verification Result as ground truth. Produce only the corrected answer with no additional commentary.

## Query
{query}

## Answer
{answer}

## Verified Claim
{claim}

## Verification Result
{verification}

## Critique
{critique}
"""


@dataclass
class VerifyResult:
    """
    Structured output from the verify step.

    Attributes:
        claim: The specific claim that was selected for verification.
        result: Raw tool output used as evidence for the critique step.
    """

    claim: str
    result: str


CLAIM_PATTERN = re.compile(r"^Claim:\s*(.+)$", re.MULTILINE)
ACTION_PATTERN = re.compile(r"^Action:\s*(.+)$", re.MULTILINE)
ACTION_INPUT_PATTERN = re.compile(r"^Action Input:\s*(.+)$", re.MULTILINE | re.DOTALL)


class CRITICAgent(ToolInvocationMixin, Agent):
    """
    CRITIC (Large Language Models Can Self-Correct with Tool-Interactive Critiquing) agent.

    Unlike pure self-reflection approaches (e.g. BasicReflectionAgent), CRITIC grounds
    its self-correction in external tool feedback rather than introspection alone.
    The LLM first generates an answer, then interacts with tools to verify key claims,
    generates a critique based on the tool output, and corrects the answer accordingly.
    This Verify → Critique → Correct cycle repeats until the answer passes verification
    or ``max_turns`` is reached.

    High-level flow:
        1. **Generate** — produce an initial answer from the LLM's parametric knowledge.
        2. **Verify**   — interact with external tools (e.g. search, code interpreter)
           to check the key claim in the current answer.
        3. **Critique** — ask the LLM "What's the problem with the above answer?"
           using the tool output as grounding evidence.
        4. **Correct**  — rewrite the answer based on the critique.
        5. Repeat steps 2–4 until no problem is found or ``max_turns`` is reached.

    Reference: Gou et al., "CRITIC: Large Language Models Can Self-Correct with
    Tool-Interactive Critiquing", ICLR 2024. https://arxiv.org/abs/2305.11738
    """

    def __init__(
        self,
        name: str,
        llm: ChatOpenAI,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_turns: int = 3,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the CRITIC agent.

        Args:
            name: Human-readable agent identifier.
            llm: LLM used for generation, verification, critique, and correction.
            tools: External tools available for claim verification (e.g. search,
                code interpreter). At least one tool is recommended.
            system_prompt: Unused; accepted for interface compatibility only.
            max_turns: Maximum number of verify-critique-correct iterations.
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

        self.log(f"🚀 Initializing CRITICAgent `{name}` ...")
        self.log(f"🛠️ Registered tools: {self.tools}")

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        """
        Execute the generate → verify → critique → correct loop.

        Args:
            user_input: Raw user query.
            **kwargs: Reserved for future execution options.

        Returns:
            Final corrected answer text.
        """
        self.log("\n==============================")
        self.log(f"👤 User Input: {user_input}")
        self.log("==============================\n")

        answer = self._generate(user_input)

        for turn_idx in range(1, self.max_turns + 1):
            self.log(f"\n🔄 Turn {turn_idx}/{self.max_turns}")
            vr = self._verify(user_input, answer)
            critique = self._critique(user_input, answer, vr)
            if "no problem" in critique.lower():
                self.log("✅ Answer accepted — no problems found.")
                break

            answer = self._correct(user_input, answer, vr, critique)

        self.log(f"📤 Final Answer:\n{answer}\n")
        return answer

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    def _generate(self, user_input: str) -> str:
        """
        Produce an initial answer from the LLM's parametric knowledge.

        Args:
            user_input: The original user query.

        Returns:
            Initial answer string.
        """
        messages = [
            Message.system(
                DEFAULT_GENERATE_PROMPT.format(current_date=date.today().isoformat())
            ),
            Message.user(user_input),
        ]
        answer = self.llm.generate(messages).content or ""
        self.log(f"📝 Initial answer:\n{answer}\n")
        return answer

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    def _verify(self, user_input: str, answer: str) -> VerifyResult:
        """
        Interact with external tools to verify the most critical claim about the answer.

        The LLM identifies the single most important factual claim needed to verify
        whether the answer satisfies the query requirements, then uses a tool to check it.

        Args:
            user_input: The original user query.
            answer: The current answer to verify.

        Returns:
            A ``VerifyResult`` containing the verified claim and the raw tool output.
        """
        user_prompt = DEFAULT_VERIFY_PROMPT.format(
            query=user_input, answer=answer, tool_descriptions=self.tool_descriptions
        )

        raw_output = self.llm.generate([Message.user(user_prompt)]).content or ""
        self.log(f"📤 Verify raw output:\n{raw_output}\n")

        # Parse Claim / Action / Action Input from the LLM output.
        claim_match = CLAIM_PATTERN.search(raw_output)
        action_match = ACTION_PATTERN.search(raw_output)
        input_match = ACTION_INPUT_PATTERN.search(raw_output)

        claim = claim_match.group(1).strip() if claim_match else ""
        tool_name = action_match.group(1).strip() if action_match else ""
        tool_input = input_match.group(1).strip() if input_match else ""

        self.log(f"🔎 Claim: {claim}")
        self.log(f"🧩 Action: {tool_name}")
        self.log(f"📦 Action Input: {tool_input}")

        return VerifyResult(
            claim=claim, result=self._handle_tool_call(tool_name, tool_input)
        )

    # ------------------------------------------------------------------
    # Critique
    # ------------------------------------------------------------------
    def _critique(self, user_input: str, answer: str, vr: VerifyResult) -> str:
        """
        Ask the LLM to identify problems in the answer given tool verification results.

        Args:
            user_input: The original user query.
            answer: The current answer being critiqued.
            vr: VerifyResult containing the verified claim and tool output.

        Returns:
            Critique string, or "No problem." if the answer is correct.
        """
        user_prompt = DEFAULT_CRITIQUE_PROMPT.format(
            current_date=date.today().isoformat(),
            query=user_input,
            answer=answer,
            claim=vr.claim,
            verification=vr.result,
        )

        critique = self.llm.generate([Message.user(user_prompt)]).content or ""
        self.log(f"🔍 Critique:\n{critique}\n")
        return critique

    # ------------------------------------------------------------------
    # Correct
    # ------------------------------------------------------------------
    def _correct(
        self, user_input: str, answer: str, vr: VerifyResult, critique: str
    ) -> str:
        """
        Rewrite the answer based on the critique and verified tool results.

        Args:
            user_input: The original user query.
            answer: The current answer to be corrected.
            vr: VerifyResult used as ground truth for the correction.
            critique: Problem description produced by ``_critique``.

        Returns:
            Corrected answer string.
        """
        user_prompt = DEFAULT_CORRECT_PROMPT.format(
            query=user_input,
            answer=answer,
            claim=vr.claim,
            verification=vr.result,
            critique=critique,
        )

        corrected = self.llm.generate([Message.user(user_prompt)]).content or ""
        self.log(f"✏️ Corrected answer:\n{corrected}\n")
        return corrected

    def _handle_tool_call(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool identified by ``tool_name`` with the given raw input string.

        Args:
            tool_name: Name of the tool to invoke.
            tool_input: Raw input string parsed from the LLM's verify output.

        Returns:
            Raw tool execution result as a string.

        Raises:
            ToolNotFoundError: If the requested tool is not registered.
            ToolCallError: If parameters are invalid or tool execution fails.
        """
        self.log(f"⚙️ Handling tool call for `{tool_name}` ...")

        # Validate tool existence
        if tool_name not in self.tool_by_names:
            self.log("❌ Tool not found!")
            raise ToolNotFoundError(f"Unknown tool: {tool_name}")

        tool = self.tool_by_names[tool_name]

        parsed_input = self._parse_tool_input(tool_input)

        # Execute tool function and capture any runtime errors.
        self.log(f"🔧 Executing tool `{tool_name}` with params: {tool_input}")
        try:
            result = self._invoke_tool(tool, parsed_input)
            self.log(f"📦 Tool result: {result}")
            return result
        except Exception as e:
            self.log(f"💥 Tool execution failed: {e}")
            raise ToolCallError(f"Tool execution failed: {e}")
