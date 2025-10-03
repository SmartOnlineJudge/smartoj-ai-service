from core.node import SmartOJToolNode
from core.config import settings


class QuestionNode(SmartOJToolNode):
    effective_tools = {
        "query_question_info", 
        "create_question",
        "query_all_tags"
    }
    model = settings.QUESTION_MANAGE_QUESTION_MODEL
    prompt_key = "question_manage.question"
