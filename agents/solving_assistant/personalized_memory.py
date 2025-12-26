from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from core.model import create_model
from core.config import settings


agent_config = settings.get_agent_config("solving_assistant")
personalized_memory = agent_config["personalized_memory"]


class PersonalizedMemory(BaseModel):
    id: int | None = None
    content: str


class StructuredOutput(BaseModel):
    levels: list[PersonalizedMemory] = []
    abilities: list[PersonalizedMemory] = []
    preferences: list[PersonalizedMemory] = []


def deserialize(output: StructuredOutput):
    def _process(memory: list[PersonalizedMemory], memory_type: str) -> list[dict]:
        return [{"id": item.id, "content": item.content, "type": memory_type} for item in memory]

    return _process(output.levels, "level") + _process(output.abilities, "ability") + _process(output.preferences, "preference")


async def summarize_personalized_memory(conversations: str, memory: str = ""):
    model = create_model(personalized_memory.model, streaming=False)
    model = model.with_structured_output(StructuredOutput)
    humam_message = f"从下面的新对话中提取相关信息：\n{conversations}"
    if memory:
        humam_message = f"该用户已经存在部分信息了：{memory}\n" + humam_message
    inputs = [SystemMessage(personalized_memory.original_prompt), HumanMessage(humam_message)]
    output = await model.ainvoke(inputs)
    return deserialize(output)
