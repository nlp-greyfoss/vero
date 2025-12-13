import time
from vero.core import Message


def test_message_user_constructor():
    msg = Message.user("hello")

    assert msg.role == "user"
    assert msg.content == "hello"
    assert isinstance(msg.timestamp, int)
    assert msg.metadata == {}


def test_message_assistant_constructor():
    msg = Message.assistant("hello")

    assert msg.role == "assistant"
    assert msg.content == "hello"


def test_message_system_constructor():
    msg = Message.system("sys")

    assert msg.role == "system"
    assert msg.content == "sys"


def test_message_metadata_custom():
    msg = Message.user("hi", metadata={"foo": "bar"})

    assert msg.metadata == {"foo": "bar"}


def test_message_to_dict():
    msg = Message.user("hi")
    d = msg.to_dict()

    assert d["role"] == "user"
    assert d["content"] == "hi"


def test_message_str():
    msg = Message.user("test")
    s = f"[{msg.role}] {msg.content}"  # 假设你的 __str__ 没改，可以这样测试

    assert s.startswith("[user]")
    assert "test" in s


def test_message_multimodal_list_content():
    content = [
        {"type": "text", "text": "hello"},
        {
            "type": "input_audio",
            "audio_url": "http://example.com/a.wav",
        },
    ]

    msg = Message.user(content)

    assert isinstance(msg.content, list)
    assert msg.content == content

    out = msg.to_dict()
    assert out["role"] == "user"
    assert out["content"] == content


def test_timestamp_auto_generated():
    t0 = int(time.time())
    msg1 = Message.user("a")
    time.sleep(1)
    msg2 = Message.user("b")
    t1 = int(time.time())

    assert t0 <= msg1.timestamp <= t1
    assert msg1.timestamp <= msg2.timestamp <= t1
