# Vero 🚀

![](assets/logo.png)

<p align="center">
  🌐 Available Languages:
  <a href="https://www.readme-i18n.com/nlp-greyfoss/vero?lang=zh">中文</a> |
  <a href="https://www.readme-i18n.com/nlp-greyfoss/vero?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/nlp-greyfoss/vero?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/nlp-greyfoss/vero?lang=fr">Français</a> |
  <a href="https://www.readme-i18n.com/nlp-greyfoss/vero?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/nlp-greyfoss/vero?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/nlp-greyfoss/vero?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/nlp-greyfoss/vero?lang=ru">Русский</a>
</p>

**Vero** (from Latin *verus* + *zero*) is a lightweight Python framework for building LLM-based intelligent agents from scratch.

It provides a clean, extensible abstraction over the OpenAI Python SDK, supporting both **streaming** and **non-streaming** chat completions, tool calling, and agent-based reasoning workflows.

---

## Features

* Minimal wrapper around the OpenAI Python SDK
* Supports **streaming** and **non-streaming** chat completions
* Unified `Message` abstraction for conversation management
* Configuration via `.env` with sensible defaults
* Fully testable with `pytest`
* **Agent system**

  * Abstract `Agent` base class
  * `SimpleAgent`: lightweight TOOL_CALL protocol
  * `OpenAIFunctionAgent`: OpenAI function calling
* **Tool system**

  * Declarative tool definition via decorator
  * OpenAI-compatible function schemas
  * Built-in tools (math evaluation, DuckDuckGo search)
  * Easy extension with type annotations

---

## Project Structure

```
.
├── assets/
│   └── logo.png
├── examples/
│   └── agents/
│       └── main.py          # Agent usage examples
├── main.py
├── pyproject.toml
├── README.md
├── tests/
│   ├── test_chat_openai.py
│   ├── test_message.py
│   └── test_tool.py
├── uv.lock
├── vero/
│   ├── agents/
│   │   ├── openai_function_agent.py
│   │   └── simple_agent.py
│   ├── config/
│   │   └── config.py        # Environment-based settings
│   ├── core/
│   │   ├── agent.py         # Base Agent abstraction
│   │   ├── chat_openai.py   # OpenAI chat wrapper
│   │   ├── exceptions.py
│   │   └── message.py       # Message abstraction
│   ├── tool/
│   │   ├── buildin/
│   │   │   ├── ddg_search.py
│   │   │   └── math_calculator.py
│   │   └── tool.py          # Tool base class and decorator
│   └── __init__.py
├── .env.example
```

---

## Core Concepts

### Message

Represents a single message in a conversation.

* Roles: `system`, `user`, `assistant`
* Optional metadata (tool calls, tokens, reasoning)
* Helper constructors:

  * `Message.user(text)`
  * `Message.system(text)`
  * `Message.assistant(text)`
* `to_dict()` produces an OpenAI-compatible message

---

### ChatOpenAI

A thin wrapper around the OpenAI Python SDK for chat models.

**Key attributes**

* `model_name`
* `temperature`
* `max_tokens`
* `api_key`
* `base_url`

**Methods**

* `generate(messages, stream=False)` → full response
* `generate(messages, stream=True)` → streaming iterator

All API errors are wrapped in `LLMCallError` exceptions.

---

### Agent

Abstract base class for LLM-powered agents with tool usage.

**Responsibilities**

* Maintain conversation history
* Manage available tools
* Execute reasoning loops
* Handle tool invocation

**Key methods**

* `run(input_text)`
* `add_message(message)`
* `clear_history()`

**Properties**

* `tool_descriptions`
* `tool_names`
* `tool_by_names`

---

### Tool

A callable capability that agents can invoke.

* Human-readable name and description
* OpenAI-compatible function schema
* Defined via a decorator

```python
from vero.tool import tool

@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

---

## Configuration

Vero loads configuration from environment variables or a `.env` file.

Minimum required configuration:

```dotenv
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://api.openai.com/v1
```

Copy the example configuration:

```bash
cp .env.example .env
```

Optional settings:

```dotenv
DEBUG=False
TIMEOUT=60
MODEL_NAME=Qwen/Qwen3-32B
TEMPERATURE=0.7
```

All settings are loaded via `Settings` in `vero/config/config.py`.

---

## Running with uv

Vero uses **uv** as the recommended environment and dependency manager.

```bash
uv sync        # create virtual environment and install dependencies
uv run main.py
```

Make sure a `.env` file exists in the project root before running.

---

## Example Usage

### Basic LLM Usage

```python
from vero.core import ChatOpenAI, Message

llm = ChatOpenAI()

# Non-streaming
messages = [Message.user("Who are you?")]
response = llm.generate(messages)
print(response)

# Streaming
messages = [Message.user("Tell me a short joke.")]
for chunk in llm.generate(messages, stream=True):
    print(chunk, end="")
```

---

### Agents with Built-in Tools

```python
import time
from vero.core import ChatOpenAI
from vero.agents import OpenAIFunctionAgent
from vero.tool.buildin import math_evaluate, duckduckgo_search

llm = ChatOpenAI()

agent = OpenAIFunctionAgent(
    name="example-agent",
    llm=llm,
    tools=[duckduckgo_search, math_evaluate],
)

start = time.perf_counter()
answer = agent.run(
    "What is 123 + 456, and when was Python first released?"
)
print(f"Answer: {answer}")
print(f"Elapsed: {time.perf_counter() - start:.2f}s")
```

---

### Custom Tools

```python
from vero.tool import tool
from vero.core import ChatOpenAI
from vero.agents import SimpleAgent

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

llm = ChatOpenAI()
agent = SimpleAgent(
    name="math-agent",
    llm=llm,
    tools=[add, multiply],
)

result = agent.run("Add 5 and 3, then multiply the result by 10.")
print(result)
```

---

## Testing

Run all tests with:

```bash
pytest
```

---

## License

MIT License