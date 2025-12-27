from langchain.messages import HumanMessage
from langgraph.checkpoint.mysql.aio_base import BaseAsyncMySQLSaver
from langgraph.store.mysql.aio_base import BaseAsyncMySQLStore
from langgraph.graph import StateGraph, MessagesState
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import RunnableConfig

from core.model import create_model
from core.config import settings


agent_config = settings.get_agent_config("solving_assistant")
solving_assistant = agent_config["solving_assistant"]


class SolvingAssistantMessagesState(MessagesState):
    question_description: str
    code: str
    user_profile: str
    user_memory: str
    username: str


async def node(state: SolvingAssistantMessagesState, config: RunnableConfig):
    # 构造系统提示词
    prompt_template = SystemMessagePromptTemplate.from_template(solving_assistant.original_prompt)
    system_prompt = prompt_template.format(
        question_description=state["question_description"],
        user_profile=state["user_profile"],
        user_memory=state["user_memory"],
        username=state["username"]
    )
    # 处理用户最后一条输入
    messages = state["messages"]
    last_message = messages[-1]
    code = state["code"]
    processed_input = f"<用户当前解题代码>:\n{code}\n<用户输入>:\n{last_message.content}"
    # 构造输入
    input_messages = [system_prompt] + messages[:-1] + [HumanMessage(processed_input)]
    model = create_model(solving_assistant.model)
    output = await model.ainvoke(input_messages, config)
    return {"messages": [last_message, output]}


def create_solving_assistant(checkpointer: BaseAsyncMySQLSaver = None, store: BaseAsyncMySQLStore = None, **kwargs):
    graph_builder = StateGraph(SolvingAssistantMessagesState)

    graph_builder.add_node("node", node)
    graph_builder.set_entry_point("node")

    return graph_builder.compile(checkpointer, store=store, **kwargs)
