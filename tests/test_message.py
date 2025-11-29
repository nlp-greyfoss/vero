import re
from datetime import datetime
from vero.core import Message

def test_message_user_constructor():
    msg = Message.user("hello")

    assert msg.role == Message.Role.user
    assert msg.content == "hello"
    assert isinstance(msg.timestamp, datetime)
    assert msg.metadata == {}


def test_message_assistant_constructor():
    msg = Message.assistant("hello")

    assert msg.role == Message.Role.assistant
    assert msg.content == "hello"


def test_message_system_constructor():
    msg = Message.system("sys")

    assert msg.role == Message.Role.system
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
    s = str(msg)

    assert s.startswith("[user]")
    assert "test" in s


def test_message_multimodal_list_content():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "input_audio", "audio_url": "http://example.com/a.wav"}
    ]

    msg = Message.user(content)

    assert isinstance(msg.content, list)
    assert msg.content == content

    out = msg.to_dict()
    assert out["content"] == content
    assert out["role"] == "user"


def test_timestamp_auto_generated():
    msg1 = Message.user("a")
    msg2 = Message.user("b")

    assert msg1.timestamp <= datetime.now()
    assert msg2.timestamp >= msg1.timestamp  
