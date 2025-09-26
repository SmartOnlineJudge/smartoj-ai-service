from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from core.node import SmartOJNode, SmartOJMessagesState
from core.config import settings


class MemoryTimeLimitNode(SmartOJNode):
    effective_tools = {
        "query_question_info", 
        "query_all_programming_languages", 
        "create_memory_time_limit_for_question",
        "query_memory_time_limits_of_question"
    }

    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model=settings.QUESTION_MANAGE_MEMORY_TIME_LIMIT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            extra_body={"enable_thinking": False}
        )
        self.prompt = SystemMessage(settings.prompt_manager.get_prompt("question_manage.memory_time_limit"))

    async def __call__(self, state: SmartOJMessagesState, config: RunnableConfig):
        tools = await self.load_tools(config)
        if not tools:
            return {"messages": [AIMessage("No backend session id")]}
        agent = create_react_agent(self.llm, tools, prompt=self.prompt)
        return await agent.ainvoke(state, config)
