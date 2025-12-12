from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langgraph.config import get_stream_writer

from utils.tool import load_tools_from_config
from core.model import create_model
from ..state import QuestionManageMessagesState
from ..config import agent_config, tool_call_node_middlewares
from .node_log import create_node_call_log


async def solving_framework_node(state: QuestionManageMessagesState, config: RunnableConfig) -> QuestionManageMessagesState:
    writer = get_stream_writer()
    writer(create_node_call_log("solving_framework", "解题框架助手开始执行任务", "entry"))
    solving_framework_config = agent_config["solving_framework"]
    model = create_model(solving_framework_config.model)
    tools = await load_tools_from_config(config, solving_framework_config.tools)
    # 从已有的题目数据构建系统提示词
    question_metadata = state["question_metadata"]
    prompt_template = SystemMessagePromptTemplate.from_template(solving_framework_config.original_prompt)
    system_prompt = prompt_template.format(**question_metadata.model_dump())
    # 创建 Agent
    agent = create_agent(model, tools, system_prompt=system_prompt, middleware=tool_call_node_middlewares)
    output_state = await agent.ainvoke({"messages": [HumanMessage(state["task_description"])]}, config)
    # 拿到 Agent 的最终执行结果并返回
    last_message = output_state["messages"][-1]
    response_content = last_message.content
    message = f"我是<solving_framework>助手，以下是我对这个任务的完成结果：\n{response_content}"
    writer(create_node_call_log("solving_framework", "解题框架助手执行任务完成", "finish"))
    return {"messages": [AIMessage(message)]}
