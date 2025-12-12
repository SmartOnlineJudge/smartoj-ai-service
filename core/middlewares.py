from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langgraph.config import get_stream_writer


class ToolCallMonitorMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        writer = get_stream_writer()
        writer(request.tool_call)
        result = await handler(request)
        writer({"name": request.tool_call["name"], "result": result.content, "type": "tool_call_result"})
        return result
