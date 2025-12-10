from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

from utils.tool import load_tools_from_config
from core.model import create_model
from ..state import QuestionManageMessagesState
from ..config import agent_config


async def test_node(state: QuestionManageMessagesState, config: RunnableConfig) -> QuestionManageMessagesState:
    test_config = agent_config["test"]
    model = create_model(test_config.model)
    tools = await load_tools_from_config(config, test_config.tools)
    # 从已有的题目数据构建系统提示词
    question_metadata = state["question_metadata"]
    prompt_template = SystemMessagePromptTemplate.from_template(test_config.original_prompt)
    system_prompt = prompt_template.format(**question_metadata.model_dump())
    # 创建 Agent
    agent = create_agent(model, tools, system_prompt=system_prompt)
    output_state = await agent.ainvoke({"messages": [HumanMessage(state["task_description"])]}, config)
    # 拿到 Agent 的最终执行结果并返回
    last_message = output_state["messages"][-1]
    response_content = last_message.content
    message = f"我是<test>助手，以下是我对这个任务的完成结果：\n{response_content}"
    return {"messages": [AIMessage(message)]}
