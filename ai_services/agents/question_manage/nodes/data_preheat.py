from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

from ai_services.agents.generic.json_parser import parse_json
from utils.tool import load_tools_from_config
from core.model import create_model
from .node_log import create_node_call_log
from ..state import QuestionMetadata, QuestionManageMessagesState
from ..config import agent_config, tool_call_node_middlewares


async def data_preheat_node(state: QuestionManageMessagesState, config: RunnableConfig) -> QuestionManageMessagesState:
    writer = get_stream_writer()
    writer(create_node_call_log("data_preheat", "数据预助手开始处理任务", "entry"))
    # 获取配置并创建模型
    data_preheat_config = agent_config["data_preheat"]
    model = create_model(data_preheat_config.model)
    # 加载工具列表
    tools = await load_tools_from_config(config, data_preheat_config.tools)
    # 创建 Agent
    agent = create_agent(
        model, 
        tools, 
        system_prompt=data_preheat_config.original_prompt,
        middleware=tool_call_node_middlewares
    )
    output_state = await agent.ainvoke(state, config)
    output_messages = output_state["messages"]
    # 将结果解析为标准JSON字符串
    question_metadata = await parse_json(output_messages[-1].content, QuestionMetadata)
    writer(create_node_call_log("data_preheat", "数据预助手处理任务完成", "finish"))
    return {"question_metadata": question_metadata}
