from core.node import SmartOJNode
from core.config import settings


class MemoryTimeLimitNode(SmartOJNode):
    effective_tools = {
        "query_question_info", 
        "query_all_programming_languages", 
        "create_memory_time_limit_for_question",
        "query_memory_time_limits_of_question"
    }
    model = settings.QUESTION_MANAGE_MEMORY_TIME_LIMIT_MODEL
    prompt_key = "question_manage.memory_time_limit"
