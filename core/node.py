from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from mcp_tool_adapter import load_mcp_tools
from core.config import settings
from .state import SmartOJMessagesState


MCP_SERVER_URL = settings.MCP_SERVER_URL


class SmartOJNode:
    effective_tools: set[str] = set()
    model: str = ""
    api_key: str = settings.OPENAI_API_KEY
    base_url: str = settings.OPENAI_BASE_URL
    llm_type: BaseChatModel = ChatOpenAI
    prompt_key: str = ""

    def __init__(self):
        self._tools: list[BaseTool] = []
        # 初始化 LLM
        self.llm = self.llm_type(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            extra_body={"enable_thinking": False}
        )
        # 初始化系统提示词
        self.prompt = SystemMessage(settings.prompt_manager.get_prompt(self.prompt_key))
        self.mcp_connection_config = {
            "url": MCP_SERVER_URL, 
            "transport": "streamable_http"
        }

    async def __call__(self, state: SmartOJMessagesState, config: RunnableConfig):
        tools = await self.load_tools(config)
        if not tools:
            return {"messages": [AIMessage("No backend session id")]}
        agent = create_react_agent(self.llm, tools, prompt=self.prompt)
        return await agent.ainvoke(state, config)

    def filter_tools(self, tools: list[BaseTool]):
        """过滤出该结点应该使用的工具"""
        filtered_tools = []
        for tool in tools:
            if tool.name in self.effective_tools:
                filtered_tools.append(tool)
        return filtered_tools

    async def load_tools(self, config: RunnableConfig):
        if self._tools:
            return self._tools
        backend_session_id = config["metadata"].get("backend-session-id")
        if not backend_session_id:
            return []
        connection_config = self.mcp_connection_config.copy()
        connection_config.update(
            {
                "headers": {
                    "backend-session-id": backend_session_id
                }
            }
        )
        _tools = await load_mcp_tools(connection_config)
        self._tools = self.filter_tools(_tools)
        return self._tools
