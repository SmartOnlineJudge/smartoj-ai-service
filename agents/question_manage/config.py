from langchain.agents.middleware import ToolRetryMiddleware, AgentMiddleware

from core.config import settings
from core.middlewares import ToolCallMonitorMiddleware

# 智能体的配置信息
agent_config = settings.get_agent_config("question_manage")
# 工具节点需要使用的中间件列表
tool_call_node_middlewares: list[AgentMiddleware] = [ToolCallMonitorMiddleware(), ToolRetryMiddleware()]
