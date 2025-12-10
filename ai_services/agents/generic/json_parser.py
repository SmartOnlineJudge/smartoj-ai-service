from pydantic import BaseModel
from langchain.messages import SystemMessage, HumanMessage

from core.model import create_model
from core.config import settings


agent_config = settings.get_agent_config("generic")["json_parser"]


async def parse_json(content: str, schema: BaseModel | dict):
    model = create_model(agent_config.model, output_version=None).with_structured_output(schema)
    messages = [
        SystemMessage(agent_config.original_prompt),
        HumanMessage("请将下面的文本转换成标准的JSON字符串：\n" + content)
    ]
    return await model.ainvoke(messages)
