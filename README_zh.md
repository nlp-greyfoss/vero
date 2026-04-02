# Vero 🚀

![](assets/logo.png)

[English](./README.md) | 中文

**Vero**（源自拉丁语 *verus* + *zero*）是一个轻量级 Python 框架，用于从零开始构建基于 LLM 的智能代理。

它在 OpenAI Python SDK 之上提供了简洁且可扩展的抽象，支持 **流式** 与 **非流式** 聊天补全、工具调用以及基于 Agent 的推理工作流。

---

## 特性

* 对 OpenAI Python SDK 的轻量封装
* 支持 **流式** 与 **非流式** 聊天补全
* 统一的 `Message` 抽象，用于对话管理
* 通过 `.env` 配置，提供合理默认值
* 可通过 `pytest` 完整测试
* **Agent 系统**
  * 抽象 `Agent` 基类
  * `SimpleAgent`：轻量级工具调用循环
  * `OpenAIFunctionAgent`：OpenAI 原生函数调用
  * `ReActAgent`：逐步 Thought / Action / Observation 循环
  * `ReWooAgent`：planner / worker / solver 流水线，支持依赖感知执行
  * `LLMCompilerAgent`：基于任务 DAG 的规划、依赖驱动调度与重规划
* **Tool 系统**
  * 通过装饰器声明工具
  * OpenAI 兼容的函数 schema
  * 内置工具：`calculate_math_expression`、`duckduckgo_search`、`google_search`、`bocha_search`
  * 结合类型注解，易于扩展

---

## 项目结构

```text
.
├── assets/
│   └── logo.png
├── examples/
│   └── agents/
│       └── main.py          # Agent 使用示例
├── main.py
├── pyproject.toml
├── README.md
├── README_zh.md
├── tests/
│   ├── test_chat_openai.py
│   ├── test_message.py
│   └── test_tool.py
├── uv.lock
├── vero/
│   ├── agents/
│   │   ├── react_agent.py
│   │   ├── rewoo_agent.py
│   │   ├── llm_compiler_agent.py
│   │   ├── openai_function_agent.py
│   │   └── simple_agent.py
│   ├── config/
│   │   └── config.py        # 基于环境变量的配置
│   ├── core/
│   │   ├── agent.py         # Agent 基类抽象
│   │   ├── chat_openai.py   # OpenAI 聊天封装
│   │   ├── exceptions.py
│   │   ├── mixins.py
│   │   └── message.py       # Message 抽象
│   ├── tool/
│   │   ├── buildin/
│   │   │   ├── bocha_search.py
│   │   │   ├── google_search.py
│   │   │   ├── ddg_search.py
│   │   │   └── math_calculator.py
│   │   └── tool.py          # Tool 基类与装饰器
│   └── __init__.py
├── .env.example
```

---

## 核心概念

### Message

表示对话中的一条消息。

* 角色：`system`、`user`、`assistant`
* 可选元数据（工具调用、token、推理信息）
* 辅助构造方法：
  * `Message.user(text)`
  * `Message.system(text)`
  * `Message.assistant(text)`
* `to_dict()` 会生成 OpenAI 兼容的消息格式

---

### ChatOpenAI

对 OpenAI Python SDK 中聊天模型的轻量封装。

**关键属性**

* `model_name`
* `temperature`
* `max_tokens`
* `api_key`
* `base_url`

**方法**

* `generate(messages, stream=False)` → 返回完整响应
* `generate(messages, stream=True)` → 返回流式迭代器

所有 API 错误都会被包装成 `LLMCallError` 异常。

---

### Agent

支持工具使用的 LLM Agent 抽象基类。

**职责**

* 维护对话历史
* 管理可用工具
* 执行推理循环
* 处理工具调用

**关键方法**

* `run(input_text)`
* `add_message(message)`
* `clear_history()`

**属性**

* `tool_descriptions`
* `tool_names`
* `tool_by_names`

### 内置 Agent

* `SimpleAgent`：使用轻量自定义协议进行简单工具调度
* `OpenAIFunctionAgent`：依赖原生 function calling
* `ReActAgent`：逐轮推理，每次只执行一步工具调用
* `ReWooAgent`：先整体规划，再分层执行 evidence；彼此独立的 evidence 步骤可并行运行
* `LLMCompilerAgent`：先编译任务 DAG，依赖满足后立即调度任务执行，并在证据不足时支持重规划

---

### Tool

Agent 可调用的能力单元。

* 可读的名称与描述
* OpenAI 兼容的函数 schema
* 通过装饰器定义

```python
from vero.tool import tool

@tool
def calculate_sum(a: int, b: int) -> int:
    """将两个数字相加。"""
    return a + b
```

---

## 配置

Vero 会从环境变量或 `.env` 文件中加载配置。

最少需要的配置：

```dotenv
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
```

复制示例配置：

```bash
cp .env.example .env
```

可选配置：

```dotenv
DEBUG=False
TIMEOUT=60
MODEL_NAME=Qwen/Qwen3-32B
TEMPERATURE=0.7
TAVILY_API_KEY=
BOCHA_API_KEY=
```

所有配置都通过 `vero/config/config.py` 中的 `Settings` 加载。

---

## 使用 uv 运行

Vero 推荐使用 **uv** 作为环境与依赖管理器。

```bash
uv sync        # 创建虚拟环境并安装依赖
uv run main.py
```

运行前请确保项目根目录下已经存在 `.env` 文件。

---

## 使用示例

### 基础 LLM 使用

```python
from vero.core import ChatOpenAI, Message

llm = ChatOpenAI()

# 非流式
messages = [Message.user("Who are you?")]
response = llm.generate(messages)
print(response)

# 流式
messages = [Message.user("Tell me a short joke.")]
for chunk in llm.generate(messages, stream=True):
    print(chunk, end="")
```

---

### 搭配内置工具的 Agent

```python
import time
from vero.core import ChatOpenAI
from vero.agents import OpenAIFunctionAgent
from vero.tool.buildin import calculate_math_expression, duckduckgo_search

llm = ChatOpenAI()

agent = OpenAIFunctionAgent(
    name="example-agent",
    llm=llm,
    tools=[duckduckgo_search, calculate_math_expression],
)

start = time.perf_counter()
answer = agent.run(
    "What is 123 + 456, and when was Python first released?"
)
print(f"Answer: {answer}")
print(f"Elapsed: {time.perf_counter() - start:.2f}s")
```

### ReWOO 示例

`ReWooAgent` 适合那些可以先拆成多个彼此独立的证据收集步骤，再进行最终综合回答的任务。可运行示例见 [examples/agents/main.py](examples/agents/main.py)。

```python
from vero.core import ChatOpenAI
from vero.agents import ReWooAgent
from vero.tool.buildin import duckduckgo_search

llm = ChatOpenAI()
agent = ReWooAgent(
    name="rewoo-agent",
    llm=llm,
    tools=[duckduckgo_search],
)

answer = agent.run(
    "Find the 2025 market capitalization of Microsoft, Apple, Nvidia, Amazon, and Meta. "
    "Rank them from highest to lowest, and identify which pair has the smallest gap."
)
print(answer)
```

---

### 自定义工具

```python
from vero.tool import tool
from vero.core import ChatOpenAI
from vero.agents import SimpleAgent

@tool
def add(a: int, b: int) -> int:
    """两个数字相加。"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """两个数字相乘。"""
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

## 测试

运行全部测试：

```bash
pytest
```

---

## 许可证

MIT License
