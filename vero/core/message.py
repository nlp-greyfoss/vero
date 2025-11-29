from typing import Dict, Any, List, Union, Self
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class Message(BaseModel):
    class Role(str, Enum):
        system = "system"
        user = "user"
        assistant = "assistant"

    content: Union[str, List[Dict[str, Any]]]
    role: Role
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def new(cls, role: Role, content: str, **kwargs) -> Self:
        return cls(role=role, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kw):
        return cls(role=Message.Role.user, content=content, **kw)

    @classmethod
    def system(cls, content: str, **kw):
        return cls(role=Message.Role.system, content=content, **kw)

    @classmethod
    def assistant(cls, content: str, **kw):
        return cls(role=Message.Role.assistant, content=content, **kw)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}
    
    def __str__(self):
        return f"[{self.role.value}] {self.content}"