import time
import os
import random

from vero.core import ChatOpenAI, Agent
from vero.agents import (
    SimpleAgent,
    OpenAIFunctionAgent,
    ReActAgent,
    ReWOOAgent,
    PlanAndExecuteAgent,
)
from vero.tool.buildin import (
    calculate_math_expression,
    duckduckgo_search,
    google_search,
    bocha_search,
)
from vero.config import settings

tools = [calculate_math_expression]

if settings.TAVILY_API_KEY:
    tools.append(google_search)
elif settings.BOCHA_API_KEY:
    tools.append(bocha_search)
else:
    tools.append(duckduckgo_search)


def run_agent(agent_class: Agent, input_text: str, max_turns=5):
    llm = ChatOpenAI()

    agent: Agent = agent_class(
        "test-agent",
        llm,
        tools=tools,
        max_turns=max_turns,
    )

    return agent.run(input_text)


def run_multi_turn_agent(agent_class: Agent, max_turns=5):
    llm = ChatOpenAI()

    agent: Agent = agent_class(
        "test-agent",
        llm,
        tools=tools,
        max_turns=max_turns,
    )

    while True:
        try:
            # Ask for user input
            user_input = input("You: ")

            # Exit condition for the loop (if user types 'bye')
            if user_input.lower() == "bye":
                print("Exiting the conversation.")
                break

            # Run the agent with the current input
            answer = agent.run(user_input)
            print(f"Assistant: {answer}\n")
        except KeyboardInterrupt:
            print("\nConversation interrupted. Exiting gracefully.")
            break


def test_single_turn_agent(agent_class: Agent, max_turns=10):
    start = time.perf_counter()
    answer = run_agent(
        agent_class,
        "what is the hometown of the current Australia open winner?",
        max_turns=max_turns,
    )
    print(f"🏁 Final LLM Answer: {answer}\n")

    print(f"⏳ Elapsed: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    agent_class = PlanAndExecuteAgent
    test_single_turn_agent(agent_class)
