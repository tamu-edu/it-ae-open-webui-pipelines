from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class OpenAIChatMessage(BaseModel):
    role: str
    content: str | List

    model_config = ConfigDict(extra="allow")


class OpenAIChatCompletionForm(BaseModel):
    stream: bool = True
    model: str
    messages: List[OpenAIChatMessage]

    model_config = ConfigDict(extra="allow")


class FilterForm(BaseModel):
    body: dict
    user: Optional[dict] = None
    model_config = ConfigDict(extra="allow")


class OpenAIEmbeddingForm(BaseModel):
    model: str
    input: str | List[str]
    encoding_format: Optional[str] = "float"
    model_config = ConfigDict(extra="allow")
