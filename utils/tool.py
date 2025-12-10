from langchain.tools import BaseTool

from core.config import settings
from mcp_tool_adapter import load_mcp_tools


mcp_connection_config = {
    "url": settings.MCP_SERVER_URL, 
    "transport": "streamable_http"
}


async def load_tools(backend_session_id: str, effective_tools: set[str]) -> list[BaseTool]:
    connection_config = mcp_connection_config.copy()
    connection_config.update(
        {
            "headers": {
                "backend-session-id": backend_session_id
            }
        }
    )
    all_mcp_tools = await load_mcp_tools(connection_config)
    if not all_mcp_tools:
        return []
    tools: list[BaseTool] = []
    for mcp_tool in all_mcp_tools:
        if mcp_tool.name in effective_tools:
            tools.append(mcp_tool)
    return tools


async def load_tools_from_config(config: dict, effective_tools: set[str]) -> list[BaseTool]:
    backend_session_id = config["metadata"].get("backend-session-id")
    return await load_tools(backend_session_id, effective_tools)
