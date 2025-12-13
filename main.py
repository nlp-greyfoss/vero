from vero.core import ChatOpenAI, Message

# Simple demonstration of using ChatOpenAI
# Full test coverage is in the `tests` folder

def test_chat_openai():
    """
    Demo usage of the ChatOpenAI class:
    - Generates a full response from a single user message.
    - Demonstrates streaming generation using the unified `generate` method.
    """
    # Initialize the LLM client (uses settings defaults if no arguments)
    llm = ChatOpenAI()

    # -------------------------
    # Full response generation
    # -------------------------
    messages = [Message.user("Who are you?")]
    response = llm.generate(messages, stream=False).content
    print("Full response:")
    print(response)

    # -------------------------
    # Streaming generation
    # -------------------------
    print("\nStreaming response:")
    messages = [{"role": "user", "content": "Tell me a short joke."}]
    for chunk in llm.generate(messages, stream=True):
        print(chunk, end="")  # print chunks as they arrive
    print()  # newline after streaming


if __name__ == "__main__":
    test_chat_openai()
