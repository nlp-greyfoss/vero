from typing import Dict, Any, Optional, List, Self, Union, Literal
import time
from pydantic import BaseModel, Field


class Message(BaseModel):
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    role: Literal["system", "user", "assistant", "tool"]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="token counts"
    )

    @classmethod
    def user(cls, content: str, **kw) -> Self:
        return cls(role="user", content=content, **kw)

    @classmethod
    def system(cls, content: str, **kw) -> Self:
        return cls(role="system", content=content, **kw)

    @classmethod
    def assistant(
        cls, content: Optional[str] = None, tool_calls: List[dict] = None, **kw
    ) -> Self:
        return cls(role="assistant", content=content, tool_calls=tool_calls, **kw)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, **kw) -> Self:
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            **kw,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("timestamp", "metadata") and v is not None
        }

        return d