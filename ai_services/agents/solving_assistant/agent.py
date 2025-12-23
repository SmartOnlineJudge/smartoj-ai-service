from langchain.agents import create_agent
from langgraph.checkpoint.mysql.aio_base import BaseAsyncMySQLSaver
from langgraph.store.mysql.aio_base import BaseAsyncMySQLStore
from langchain_core.prompts import SystemMessagePromptTemplate

from core.model import create_model
from core.config import settings


agent_config = settings.get_agent_config("solving_assistant")
solving_assistant = agent_config["solving_assistant"]


def create_solving_assistant(
    question_description: str,
    code: str,
    checkpointer: BaseAsyncMySQLSaver = None, 
    store: BaseAsyncMySQLStore = None, 
    **kwargs
):
    model = create_model(solving_assistant.model)
    prompt_template = SystemMessagePromptTemplate.from_template(solving_assistant.original_prompt)
    system_prompt = prompt_template.format(question_description=question_description, code=code)
    return create_agent(
        model, 
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        store=store,
        **kwargs
    )
