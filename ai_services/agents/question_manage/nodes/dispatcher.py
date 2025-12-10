from typing import Literal

from langchain.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from pydantic import BaseModel

from ..config import agent_config
from ..state import QuestionManageMessagesState
from core.model import create_model
from ai_services.agents.generic.json_parser import parse_json


path_map = [
    "judge_template",
    "memory_time_limit",
    "solving_framework",
    "test",
    "planner",
    END
]
NextNodeType = Literal["judge_template", "memory_time_limit", "solving_framework", "test", "planner", None]


class StructuredOutput(BaseModel):
    assistant: NextNodeType
    task_description: str


async def dispatcher_node(state: QuestionManageMessagesState, config: RunnableConfig) -> QuestionManageMessagesState:
    dispatcher_config = agent_config["dispatcher"]
    model = create_model(dispatcher_config.model)
    messages = [SystemMessage(dispatcher_config.original_prompt)] + state["messages"]
    response = await model.ainvoke(messages, config)
    return await parse_json(response.content, StructuredOutput)


def dispatch_next_node(state: QuestionManageMessagesState):
    next_assistant = state.get("assistant")
    if next_assistant is None:
        return END
    return next_assistant
