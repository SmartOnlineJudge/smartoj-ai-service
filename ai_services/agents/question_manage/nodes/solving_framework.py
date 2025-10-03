from core.node import SmartOJToolNode
from core.config import settings


class SolvingFrameworkNode(SmartOJToolNode):
    effective_tools = {
        "query_question_info", 
        "query_all_programming_languages", 
        "create_solving_framework_for_question",
        "query_solving_frameworks_of_question",
        "update_solving_framework_for_question"
    }
    model = settings.QUESTION_MANAGE_SOLVING_FRAMEWORK_MODEL
    prompt_key = "question_manage.solving_framework"
    