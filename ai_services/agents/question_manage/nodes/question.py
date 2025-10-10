from langchain_core.runnables import RunnableConfig

from core.node import SmartOJToolNode
from core.config import settings
from core.state import QuestionMetadata, SmartOJMessagesState


class QuestionNode(SmartOJToolNode):
    effective_tools = {
        "query_question_info", 
        "create_question",
        "query_all_tags",
        "query_all_programming_languages"
    }
    model = settings.QUESTION_MANAGE_QUESTION_MODEL
    prompt_key = "question_manage.question"

    async def __call__(self, state: SmartOJMessagesState, config: RunnableConfig):
        self.build_prompt(state, use_original_prompt=True)
        response = await self.call_llm_with_tools(state["messages"], config)
        question_metadata = QuestionMetadata.model_validate_json(response.content)
        return {"question_metadata": question_metadata}
