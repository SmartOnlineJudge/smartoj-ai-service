from core.node import SmartOJNode, SmartOJMessagesState


class TestNode(SmartOJNode):
    async def __call__(self, state: SmartOJMessagesState):
        return {"messages": [{"role": "assistant", "content": "test"}]}
