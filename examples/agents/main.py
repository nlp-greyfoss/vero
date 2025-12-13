import time

from vero.core import ChatOpenAI, Agent
from vero.agents import SimpleAgent, OpenAIFunctionAgent
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
    # settings.model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    start = time.perf_counter()
    answer = run_agent(
        OpenAIFunctionAgent,
        "What is the total box office revenue (in USD) of the top three highest-grossing \
animated films released worldwide before 2010, and which studios produced each of those three films?",
    )
    print(f"🏁 Final LLM Answer: {answer}\n")

    print(f"⏳ Elapsed: {time.perf_counter() - start:.1f} s")
