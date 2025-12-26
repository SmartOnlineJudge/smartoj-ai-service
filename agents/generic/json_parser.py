from pydantic import BaseModel
from langchain.messages import SystemMessage, HumanMessage

from core.model import create_model
from core.config import settings


agent_config = settings.get_agent_config("generic")["json_parser"]


async def parse_json(
    content: str, 
    schema: BaseModel | dict, 
    streaming: bool = False,
    method: str = "json_schema"
):
    model = create_model(agent_config.model, streaming=streaming)
    structured_model = model.with_structured_output(schema, method=method)
    messages = [
        SystemMessage(agent_config.original_prompt),
        HumanMessage("从下面的文本中提取出指定schema的JSON数据：\n" + content)
    ]
    return await structured_model.ainvoke(messages)
