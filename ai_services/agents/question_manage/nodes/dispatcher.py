from typing import Literal

from langgraph.constants import END
from pydantic import BaseModel

from core.node import SmartOJNode, SmartOJMessagesState
from core.config import settings


path_map = [
    "judge_template",
    "memory_time_limit",
    "question",
    "solving_framework",
    "test",
    "planner",
    END
]


NextNodeType = Literal["question", "judge_template", "memory_time_limit", "solving_framework", "test", "planner"]


class StructuredOutput(BaseModel):
    next_node: NextNodeType


class DispatcherMessagesState(SmartOJMessagesState):
    next_node: NextNodeType


class DispatcherNode(SmartOJNode):
    model = settings.QUESTION_MANAGE_DISPATCHER_MODEL
    prompt_key = "question_manage.dispatcher"

    def __init__(self):
        super().__init__()
        self.llm = self.llm.with_structured_output(StructuredOutput)

    async def __call__(self, state: DispatcherMessagesState):
        messages = [self.prompt] + state["messages"]
        response = await self.llm.ainvoke(messages)
        return {"next_node": response.next_node}


def dispatch_next_node(state: DispatcherMessagesState):
    if not (next_node := state.get("next_node")):
        return END
    return next_node
