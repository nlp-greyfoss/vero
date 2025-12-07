from typing import Optional, Iterator, List, Union
from openai import OpenAI

from .message import Message
from vero.config import settings
from vero.core.exceptions import LLMCallError, LLMConfigError


class ChatOpenAI:
    """
    Wrapper around OpenAI's Python SDK to interact with chat-based LLMs.

    Attributes:
        model_name: The name of the LLM model to use.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens for the response.
        timeout: Request timeout in seconds.
        api_key: OpenAI API key.
        base_url: OpenAI API base URL.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the LLM wrapper. Falls back to default settings if parameters are not provided.

        Raises:
            LLMConfigError: If any of api_key, base_url, or model_name is missing.
        """

        self.model_name = model_name or settings.model_name
        print(f"🤖 Initializing LLM with model: {self.model_name}")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or getattr(settings, "timeout", None)
        self.kwargs = kwargs

        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url

        if not all([self.api_key, self.base_url, self.model_name]):
            raise LLMConfigError(
                "Missing api_key, base_url, or model_name for LLM client"
            )

        self._client = self._create_client()

    def _create_client(self) -> OpenAI:
        """
        Create and return an OpenAI client instance.
        """
        return OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    def generate(
        self,
        messages: List[Union[Message, dict]],
        stream: bool = False,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Union[str, Iterator[str]]:
        """
        Generate a response from the LLM, supporting both streaming and full output.

        Args:
            messages: A list of Message objects or dicts representing conversation history.
            stream: If True, return an iterator yielding text chunks as they are generated.
            temperature: Optional override of the default temperature.
            **kwargs: Additional generation parameters (e.g., max_tokens).

        Returns:
            str: Complete response if stream=False.
            Iterator[str]: Streamed response chunks if stream=True.

        Raises:
            LLMCallError: If the LLM API call fails.
        """
        # Convert Message objects to dicts if needed
        messages_dict = [
            msg.to_dict() if isinstance(msg, Message) else msg for msg in messages
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages_dict,
                temperature=temperature or self.temperature,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=stream,
                **{k: v for k, v in kwargs.items() if k not in {"temperature", "max_tokens"}},
            )

            if stream:
                # Use a helper generator to yield chunks; avoids making the whole function a generator
                def _stream_generator():
                    # Yield chunks if streaming
                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        if content:
                            yield content
                return _stream_generator()
            else:
                # Return complete response
                return response.choices[0].message.content

        except Exception as e:
            raise LLMCallError(f"LLM call failed: {str(e)}") from e