from typing import Literal

from langgraph.constants import END
from pydantic import BaseModel

from core.node import SmartOJNode, SmartOJMessagesState
from core.config import settings


target_node_mapping = {
    "judge_template": "judge_template",
    "memory_time_limit": "memory_time_limit",
    "question": "question",
    "solving_framework": "solving_framework",
    "test": "test"
}


NextNodeType = Literal["question", "judge_template", "language", "memory_time_limit", "solving_framework", "tag", "test"]


class StructuredOutput(BaseModel):
    next_node: NextNodeType


class DispatcherMessagesState(SmartOJMessagesState):
    next_node: NextNodeType


class DispatcherNode(SmartOJNode):
    model = settings.QUESTION_MANAGE_DISPATCHER_MODEL
    api_key = settings.OPENAI_API_KEY
    base_url = settings.OPENAI_BASE_URL
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
