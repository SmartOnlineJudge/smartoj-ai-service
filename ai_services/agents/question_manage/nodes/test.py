from core.node import SmartOJNode
from core.config import settings


class TestNode(SmartOJNode):
    effective_tools = {
        "query_question_info", 
        "query_tests_of_question", 
        "create_test_for_question"
    }
    model = settings.QUESTION_MANAGE_TEST_MODEL
    prompt_key = "question_manage.test"
