"""
代码声明：
    该包下的`sessions.py`和`tools.py`这两个模块的代码并非本人编写
    上述两个模块的绝大部分代码均来自于：https://github.com/langchain-ai/langchain-mcp-adapters
"""
from .sessions import Connection
from .tools import load_mcp_tools as _load_mcp_tools, BaseTool


async def load_mcp_tools(connection: Connection) -> list[BaseTool]:
    return await _load_mcp_tools(None, connection=connection)
