import asyncio

from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from mcp_tool_adapter import load_mcp_tools
from core.config import settings
from .state import SmartOJMessagesState


MCP_SERVER_URL = settings.MCP_SERVER_URL


class SmartOJNode:
    """普通节点，不需要调用工具"""

    model: str = ""
    api_key: str = settings.OPENAI_API_KEY
    base_url: str = settings.OPENAI_BASE_URL
    llm_type: type[BaseChatModel] = ChatOpenAI
    prompt_key: str = ""

    def __init__(self):
        # 初始化 LLM
        self.llm = self.llm_type(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            extra_body={"enable_thinking": False}
        )
        # 初始化系统提示词
        self.prompt = SystemMessage(settings.prompt_manager.get_prompt(self.prompt_key))

    async def __call__(self, state: SmartOJMessagesState, config: RunnableConfig):
        raise NotImplementedError


class SmartOJToolNode(SmartOJNode):
    """需要调用工具的节点"""

    effective_tools: set[str] = set()

    def __init__(self):
        super().__init__()
        self._tools: list[BaseTool] = []
        self._name2tool: dict[str, BaseTool] = {}
        self.mcp_connection_config = {
            "url": MCP_SERVER_URL, 
            "transport": "streamable_http"
        }

    async def __call__(self, state: SmartOJMessagesState, config: RunnableConfig):
        response = await self.call_llm_with_tools([HumanMessage(state["description"])], config)
        assitant_name = state["assistant"]
        response_content = response.content
        message = f"我是<{assitant_name}>助手，以下是我对这个任务的完成结果：\n{response_content}"
        return {"messages": [AIMessage(message)]}

    @classmethod
    async def process_tool_call(cls, tool_call: dict, name2tool: dict[str, BaseTool]):
        tool = name2tool[tool_call['name']]
        result = await tool.ainvoke(tool_call['args'])
        return result, tool_call["id"]

    async def call_all_tools(self, tool_calls: list[dict], name2tool: dict[str, BaseTool]):
        tool_call_tasks = [self.process_tool_call(tool_call, name2tool) for tool_call in tool_calls]
        tool_call_results = await asyncio.gather(*tool_call_tasks)
        tool_messages = []
        for result, tool_call_id in tool_call_results:
            tool_messages.append(ToolMessage(result, tool_call_id=tool_call_id))
        return tool_messages

    async def call_llm_with_tools(self, messages: list[BaseMessage], config: RunnableConfig):
        copied_messages = [self.prompt] + messages.copy()
        tools, name2tool = await self.load_tools(config)
        if not tools:
            return {"messages": [AIMessage("No backend session id")]}
        llm_with_tools = self.llm.bind_tools(tools)
        # ReAct 过程：思考 - 行动 - 观察
        response = await llm_with_tools.ainvoke(copied_messages, config)
        tool_calls = response.tool_calls
        copied_messages.append(response)
        while tool_calls:
            tool_messages = await self.call_all_tools(tool_calls, name2tool)
            copied_messages.extend(tool_messages)
            response = await llm_with_tools.ainvoke(copied_messages, config)
            tool_calls = response.tool_calls
            copied_messages.append(response)
        return response  # 只返回最后一次模型的响应

    def filter_tools(self, tools: list[BaseTool]):
        """过滤出该结点应该使用的工具"""
        filtered_tools = []
        for tool in tools:
            if tool.name in self.effective_tools:
                filtered_tools.append(tool)
        return filtered_tools

    async def load_tools(self, config: RunnableConfig):
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
        tools = await load_mcp_tools(connection_config)
        tools = self.filter_tools(tools)
        name2tool = {}
        for tool in tools:
            name2tool[tool.name] = tool
        return tools, name2tool
