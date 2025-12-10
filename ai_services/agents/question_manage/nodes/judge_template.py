from typing import Literal

from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import RunnableConfig

from utils.tool import load_tools_from_config
from ..state import QuestionManageMessagesState
from core.model import create_model
from ..config import agent_config
from core.config import settings


class TargetLanguage(BaseModel):
    language: Literal["c", "cpp", "java", "python", "javascript", "golang", None]


async def get_language(state: QuestionManageMessagesState, config: RunnableConfig):
    judge_template_dispatcher_config = agent_config["judge_template_dispatcher"]
    model = create_model(judge_template_dispatcher_config.model, output_version=None).with_structured_output(TargetLanguage)
    messages = [
        SystemMessage(judge_template_dispatcher_config.original_prompt), 
        HumanMessage(state["task_description"])
    ]
    response = await model.ainvoke(messages, config)
    return response.language


async def judge_template_node(state: QuestionManageMessagesState, config: RunnableConfig) -> QuestionManageMessagesState:
    language = await get_language(state, config)
    if language is None:
        return {"messages": [AIMessage("需要先指定编程语言才能进行后续操作")]}
    judge_template_config = agent_config["judge_template"]
    model = create_model(judge_template_config.model)
    # 加载工具
    tools = await load_tools_from_config(config, judge_template_config.tools)
    # 从已有的题目数据构建系统提示词
    question_metadata = state["question_metadata"]
    original_prompt = settings.get_prompt(judge_template_config.prompt_key % language)
    prompt_template = SystemMessagePromptTemplate.from_template(original_prompt)
    system_prompt = prompt_template.format(**question_metadata.model_dump())
    # 创建 Agent
    agent = create_agent(model, tools, system_prompt=system_prompt)
    output_state = await agent.ainvoke({"messages": [HumanMessage(state["task_description"])]}, config)
    # 拿到 Agent 的最终执行结果并返回
    last_message = output_state["messages"][-1]
    response_content = last_message.content
    message = f"我是<judge_template>助手，以下是我对这个任务的完成结果：\n{response_content}"
    return {"messages": [AIMessage(message)]}
