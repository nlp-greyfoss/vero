import time

from vero.core import ChatOpenAI, Agent
from vero.agents import SimpleAgent
from vero.tool.buildin import math_evaluate, duckduckgo_search
from vero.config import settings


def run_agent(agent_class: Agent, input_text: str, max_turns=5):
    llm = ChatOpenAI()
    
    agent: Agent = agent_class(
        "test-agent",
        llm,
        tools=[duckduckgo_search, math_evaluate],
        max_turns=max_turns,
    )

    return agent.run(input_text)


if __name__ == "__main__":
    # settings.model_name = "xxx"
    start = time.perf_counter()
    answer = run_agent(
        SimpleAgent,
        "What is NVIDIA’s revenue for Q3 2025?",
    )
    print(f"🏁 Final LLM Answer: {answer}\n")

    print(f"⏳ Elapsed: {time.perf_counter() - start:.1f} s")
