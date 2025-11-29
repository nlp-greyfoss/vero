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

Vero (from Latin *verus* + *zero*) is a lightweight Python framework for building LLM-based intelligent agents from scratch. It leverages the OpenAI Python SDK to provide a clean, extensible interface for both streaming and non-streaming interactions with chat-based language models.

---

## Features

- Simple wrapper around OpenAI Python SDK for chat models
- Supports both **streaming** and **non-streaming** outputs
- Unified `Message` class for conversation management
- Easy configuration via `.env` or `.env.example`
- Fully testable with `pytest`

---

## Project Structure

```

/
├── main.py
├── pyproject.toml
├── README.md
├── tests/
│   ├── test_chat_openai.py
│   └── test_message.py
├── uv.lock
├── vero/
│   ├── config/
│   │   └── config.py        # Settings class for environment and defaults
│   └── core/
│       ├── chat_openai.py   # ChatOpenAI LLM wrapper
│       ├── exceptions.py    # Custom exceptions (LLMCallError, LLMConfigError)
│       └── message.py       # Message class for role-based messages
├── .env.example             # Example environment configuration

````

---

## Core Classes

### `Message`
Encapsulates a single message in a conversation.

- Roles: `system`, `user`, `assistant`
- Metadata support for tools, token counts, reasoning, etc.
- Methods:
  - `Message.user("text")`
  - `Message.system("text")`
  - `Message.assistant("text")`
  - `to_dict()` → OpenAI-compatible dict

### `ChatOpenAI`
Wrapper around OpenAI Python SDK for chat-based LLMs.

- Attributes:
  - `model_name`, `temperature`, `max_tokens`, `api_key`, `base_url`
- Methods:
  - `generate(messages, stream=False)` → full response string
  - `generate(messages, stream=True)` → iterator for streaming chunks
- Handles exceptions and returns `LLMCallError` if API fails

---

## Configuration

Vero reads configuration from a `.env` file. You must set at least:

```dotenv
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://api.openai.com/v1
````

You can also copy `.env.example` as a template:

```bash
cp .env.example .env
```

Optional:

```dotenv
DEBUG=False
TIMEOUT=60
MODEL_NAME=Qwen/Qwen3-32B
TEMPERATURE=0.7
```

The `Settings` class in `vero/config/config.py` will load these automatically.

---

## Running with UV

You can run Vero via `uv` environment (virtual environment manager):

```bash
uv install    # install dependencies from pyproject.toml
uv run main.py
```

Make sure `.env` is present in the root directory.

The following examples assume you have created a .env file and set `OPENAI_API_KEY` and `OPENAI_API_BASE`.

---

## Example Usage


```python
from vero.core import ChatOpenAI, Message

# Initialize client
llm = ChatOpenAI()

# Non-streaming
messages = [Message.user("Who are you?")]
response = llm.generate(messages, stream=False)
print("Full response:")
print(response)

# Streaming
messages = [{"role": "user", "content": "Tell me a short joke."}]
for chunk in llm.generate(messages, stream=True):
    print(chunk, end="")
```

---

## Testing

Run the tests using `pytest`:

```bash
pytest
```

---

## License

MIT License