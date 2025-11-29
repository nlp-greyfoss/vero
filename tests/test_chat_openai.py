import pytest
from unittest.mock import MagicMock, patch
from vero.core import Message, ChatOpenAI, LLMConfigError, LLMCallError


def test_init_missing_config(monkeypatch):
    """
    Test ChatOpenAI raises LLMConfigError if required configuration is missing.
    """
    from vero.config import settings

    # Patch settings to None to simulate missing configuration
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openai_base_url", None)
    monkeypatch.setattr(settings, "model_name", None)

    with pytest.raises(LLMConfigError):
        ChatOpenAI(api_key=None, base_url=None, model_name=None)


@patch("vero.core.chat_openai.OpenAI")
def test_generate_returns_text(mock_openai_class):
    """
    Test that generate() returns the full response string when stream=False.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Mock response for non-streaming call
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello, world!"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response

    chat = ChatOpenAI(api_key="dummy", base_url="https://dummy", model_name="test-model")
    messages = [Message.user("Hi")]

    # Non-streaming
    result = chat.generate(messages, stream=False)
    assert isinstance(result, str)
    assert result == "Hello, world!"


@patch("vero.core.chat_openai.OpenAI")
def test_generate_stream_yields_chunks(mock_openai_class):
    """
    Test that generate() returns a generator yielding chunks when stream=True.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Mock streaming response
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = "Hello"
    mock_client.chat.completions.create.return_value = iter([mock_chunk])

    chat = ChatOpenAI(api_key="dummy", base_url="https://dummy", model_name="test-model")
    messages = [Message.user("Hi")]

    result_gen = chat.generate(messages, stream=True)
    assert hasattr(result_gen, "__iter__")  # It should be a generator
    chunks = list(result_gen)
    assert chunks == ["Hello"]


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
