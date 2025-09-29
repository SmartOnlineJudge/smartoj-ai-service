from typing import Optional

from langgraph.graph.message import MessagesState


class SmartOJMessagesState(MessagesState):
    assistant: Optional[str]
    description: Optional[str]
