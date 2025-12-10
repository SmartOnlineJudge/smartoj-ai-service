from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent

from ai_services.agents.generic.json_parser import parse_json
from utils.tool import load_tools_from_config
from core.model import create_model
from ..state import QuestionMetadata, QuestionManageMessagesState
from ..config import agent_config


async def data_preheat_node(state: QuestionManageMessagesState, config: RunnableConfig) -> QuestionManageMessagesState:
    # 获取配置并创建模型
    data_preheat_config = agent_config["data_preheat"]
    model = create_model(data_preheat_config.model)
    # 加载工具列表
    tools = await load_tools_from_config(config, data_preheat_config.tools)
    # 创建 Agent
    agent = create_agent(model, tools, system_prompt=data_preheat_config.original_prompt)
    output_state = await agent.ainvoke(state, config)
    output_messages = output_state["messages"]
    # 将结果解析为标准JSON字符串
    question_metadata = await parse_json(output_messages[-1].content, QuestionMetadata)
    return {"question_metadata": question_metadata}
