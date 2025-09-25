from langgraph.constants import END

from core.node import SmartOJNode, SmartOJMessagesState


target_node_mapping = {
    "judge_template": "judge_template",
    "language": "language",
    "memory_time_limit": "memory_time_limit",
    "question": "question",
    "solving_framework": "solving_framework",
    "tag": "tag",
    "test": "test"
}


class DispatcherMessagesState(SmartOJMessagesState):
    next_node: str


class DispatcherNode(SmartOJNode):
    async def __call__(self, state: DispatcherMessagesState):
        return {"next_node": "test"}


def dispatch_next_node(state: DispatcherMessagesState):
    if not (next_node := state.get("next_node")):
        return END
    return next_node
