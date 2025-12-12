from typing import Literal

from langchain.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.config import get_stream_writer
from pydantic import BaseModel

from .node_log import create_node_call_log
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
    writer = get_stream_writer()
    writer(create_node_call_log("dispatcher", "任务调度助手开始分配任务", "entry"))
    dispatcher_config = agent_config["dispatcher"]
    model = create_model(dispatcher_config.model)
    messages = [SystemMessage(dispatcher_config.original_prompt)] + state["messages"]
    response = await model.ainvoke(messages, config)
    writer(create_node_call_log("dispatcher", "任务调度助手分配任务完毕", "finish"))
    return await parse_json(response.content, StructuredOutput)


def dispatch_next_node(state: QuestionManageMessagesState):
    next_assistant = state.get("assistant")
    if next_assistant is None:
        return END
    return next_assistant
