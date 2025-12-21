from typing import Optional, Literal, Annotated
from operator import add

from pydantic import BaseModel
from langgraph.graph.message import MessagesState, add_messages
from langchain.messages import AnyMessage


class QuestionMetadata(BaseModel):
    question_id: int
    question_title: str
    question_description: str
    question_difficulty: str
    question_tags: list[str]


class Step(BaseModel):
    assistant: Literal["judge_template_for_python", "memory_time_limit", "solving_framework", "test", "planner", None]
    task_description: str


class QuestionManageMessagesState(MessagesState):
    # 题目元信息
    question_metadata: Optional[QuestionMetadata]
    # 任务执行计划
    plan: Annotated[list[Step], add]
    # 展示给客户端的对话信息
    display_messages: Annotated[list[AnyMessage], add_messages]
