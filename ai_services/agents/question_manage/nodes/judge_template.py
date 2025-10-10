from typing import Literal

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from core.node import SmartOJToolNode, SmartOJMessagesState
from core.config import settings


ProgrammingLanguageType = Literal["c", "cpp", "java", "python", "javascript", "golang", None]


class TargetLanguage(BaseModel):
    language: ProgrammingLanguageType


class JudgeTemplateNode(SmartOJToolNode):
    effective_tools = {
        "query_solving_frameworks_of_question",
        "query_tests_of_question",
        "create_judge_template_for_question",
        "query_judge_templates_of_question",
        "update_judge_template_for_question"
    }
    model = settings.QUESTION_MANAGE_JUDGE_TEMPLATE_MODEL

    def __init__(self):
        super().__init__()
        self.dispatcher_llm = ChatOpenAI(
            model=settings.QUESTION_MANAGE_JUDGE_TEMPLATE_DISPATCHER_MODEL,
            api_key=self.api_key,
            base_url=self.base_url,
            extra_body={"enable_thinking": False}
        ).with_structured_output(TargetLanguage)
        self.dispathcer_llm_prompt = SystemMessage(settings.prompt_manager.get_prompt("question_manage.judge_template.dispatcher"))

    async def __call__(self, state: SmartOJMessagesState, config: RunnableConfig):
        messages = [self.dispathcer_llm_prompt] + state["messages"]
        response = await self.dispatcher_llm.ainvoke(messages, config)
        target_language = response.language
        if target_language is None:
            return {"messages": [AIMessage(content="在操作判题模板前，需要先指定一个编程语言")]}
        self.original_prompt = settings.prompt_manager.get_prompt(f"question_manage.judge_template.{target_language}")
        return await super().__call__(state, config)
