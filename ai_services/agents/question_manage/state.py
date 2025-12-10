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


class QuestionManageMessagesState(MessagesState):
    # 题目元信息
    question_metadata: Optional[QuestionMetadata]
    # 下一个需要执行任务的Agent和任务描述
    assistant: Optional[str]
    task_description: Optional[str]
