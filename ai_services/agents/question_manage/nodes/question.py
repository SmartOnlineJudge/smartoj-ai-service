from core.node import SmartOJNode
from core.config import settings


class QuestionNode(SmartOJNode):
    effective_tools = {
        "query_question_info", 
        "create_question",
        "query_all_tags"
    }
    model = settings.QUESTION_MANAGE_QUESTION_MODEL
    prompt_key = "question_manage.question"
