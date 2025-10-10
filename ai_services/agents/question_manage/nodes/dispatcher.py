from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from pydantic import BaseModel

from core.node import SmartOJNode, SmartOJMessagesState
from core.config import settings


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


class DispatcherNode(SmartOJNode):
    model = settings.QUESTION_MANAGE_DISPATCHER_MODEL
    prompt_key = "question_manage.dispatcher"

    def __init__(self):
        super().__init__()
        self.llm = self.llm.with_structured_output(StructuredOutput)

    async def __call__(self, state: SmartOJMessagesState, config: RunnableConfig):
        self.build_prompt(state, use_original_prompt=True)
        messages = [self.prompt] + state["messages"]
        response = await self.llm.ainvoke(messages, config)
        return {"assistant": response.assistant, "task_description": response.task_description}


def dispatch_next_node(state: SmartOJMessagesState):
    next_assistant = state.get("assistant")
    if next_assistant is None:
        return END
    return next_assistant
