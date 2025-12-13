import pytest
from unittest.mock import MagicMock, patch
from vero.core import Message, ChatOpenAI, LLMConfigError, LLMCallError


def test_init_missing_config(monkeypatch):
    """
    Test ChatOpenAI raises LLMConfigError if required configuration is missing.
    """
    from vero.config import settings

    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_base_url", None)
    monkeypatch.setattr(settings, "model_name", None)

    with pytest.raises(LLMConfigError):
        ChatOpenAI(api_key=None, base_url=None, model_name=None)


@patch("vero.core.chat_openai.OpenAI")
def test_generate_returns_message(mock_openai_class):
    """
    Test that generate() returns a Message when stream=False.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # ---- mock response ----
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello, world!"
    mock_choice.message.tool_calls = None

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # mock token usage
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    mock_client.chat.completions.create.return_value = mock_response

    chat = ChatOpenAI(api_key="dummy", base_url="https://dummy", model_name="test-model")
    messages = [Message.user("Hi")]

    result = chat.generate(messages, stream=False)

    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "Hello, world!"
    assert "usage" in result.metadata
    assert result.metadata["usage"]["total_tokens"] == 15


@patch("vero.core.chat_openai.OpenAI")
def test_generate_stream_yields_chunks(mock_openai_class):
    """
    Test that generate() returns a generator yielding text chunks when stream=True.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = "Hello"

    mock_client.chat.completions.create.return_value = iter([mock_chunk])

    chat = ChatOpenAI(api_key="dummy", base_url="https://dummy", model_name="test-model")
    messages = [Message.user("Hi")]

    result_gen = chat.generate(messages, stream=True)

    assert hasattr(result_gen, "__iter__")
    assert list(result_gen) == ["Hello"]


@patch("vero.core.chat_openai.OpenAI")
def test_generate_raises_llm_call_error_non_stream(mock_openai_class):
    """
    Test that generate() raises LLMCallError if OpenAI client throws an exception (non-streaming).
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API failure")

    chat = ChatOpenAI(api_key="dummy", base_url="https://dummy", model_name="test-model")
    messages = [Message.user("Hi")]

    with pytest.raises(LLMCallError):
        chat.generate(messages, stream=False)


@patch("vero.core.chat_openai.OpenAI")
def test_generate_raises_llm_call_error_stream(mock_openai_class):
    """
    Test that generate() raises LLMCallError if OpenAI client throws an exception (streaming).
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API failure")

    chat = ChatOpenAI(api_key="dummy", base_url="https://dummy", model_name="test-model")
    messages = [Message.user("Hi")]

    with pytest.raises(LLMCallError):
        list(chat.generate(messages, stream=True))
