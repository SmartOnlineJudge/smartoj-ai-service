from core.node import SmartOJToolNode
from core.config import settings


class MemoryTimeLimitNode(SmartOJToolNode):
    effective_tools = {
        "create_memory_time_limit_for_question",
        "query_memory_time_limits_of_question",
        "update_memory_time_limit_for_question"
    }
    model = settings.QUESTION_MANAGE_MEMORY_TIME_LIMIT_MODEL
    prompt_key = "question_manage.memory_time_limit"
