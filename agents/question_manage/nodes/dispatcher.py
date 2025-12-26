from langchain.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.config import get_stream_writer

from .node_log import create_node_call_log
from ..config import agent_config
from ..state import QuestionManageMessagesState, Step
from core.model import create_model
from agents.generic.json_parser import parse_json


path_map = [
    "judge_template_for_python",
    "memory_time_limit",
    "solving_framework",
    "test",
    "planner",
    END
]


async def dispatcher_node(state: QuestionManageMessagesState, config: RunnableConfig) -> QuestionManageMessagesState:
    # 仅聊天模式，不需要执行任务
    question_metadata = state.get("question_metadata")
    if question_metadata is None:
        return {"plan": [Step(assistant=None, task_description="")]}
    writer = get_stream_writer()
    writer(create_node_call_log("dispatcher", "任务调度助手开始分配任务", "entry"))
    dispatcher_config = agent_config["dispatcher"]
    model = create_model(dispatcher_config.model)
    messages = [SystemMessage(dispatcher_config.original_prompt)] + state["messages"]
    response = await model.ainvoke(messages, config)
    writer(create_node_call_log("dispatcher", "任务调度助手分配任务完毕", "finish"))
    step = await parse_json(response.content, Step)
    return {"plan": [step]}


def dispatch_next_node(state: QuestionManageMessagesState):
    plan = state.get("plan")
    if plan is None:
        return END
    assistant = plan[-1].assistant
    if assistant is None:
        return END
    return assistant
