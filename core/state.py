from typing import Optional

from pydantic import BaseModel
from langgraph.graph.message import MessagesState


class Language(BaseModel):
    id: int
    name: str
    version: str
    is_deleted: bool


class QuestionMetadata(BaseModel):
    question_id: int
    question_title: str
    question_description: str
    question_difficulty: str
    question_tags: list[str]
    languages: list[Language]


class SmartOJMessagesState(MessagesState):
    assistant: Optional[str]
    task_description: Optional[str]
    question_metadata: Optional[QuestionMetadata]
