from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage

from mcp_tool_adapter import load_mcp_tools
from core.config import settings
from .state import SmartOJMessagesState


MCP_SERVER_URL = settings.MCP_SERVER_URL


class SmartOJNode:
    effective_tools: set[str] = set()

    def __init__(self):
        self._tools: list[BaseTool] = []
        self.prompt: SystemMessage = None
        self.mcp_connection_config = {
            "url": MCP_SERVER_URL, 
            "transport": "streamable_http"
        }

    async def __call__(self, state: SmartOJMessagesState, *args, **kwargs):
        raise NotImplementedError

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
