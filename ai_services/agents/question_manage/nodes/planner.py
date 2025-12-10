from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain.messages import HumanMessage, AIMessage

from ..config import agent_config
from ..state import QuestionManageMessagesState
from core.model import create_model


class Step(BaseModel):
    assistant: str = Field(description="需要使用的助手名字")
    task_description: str = Field(description="需要完成的任务的描述")


class StructuredOutput(BaseModel):
    plan: list[Step]


async def planner_node(state: QuestionManageMessagesState, config: RunnableConfig) -> QuestionManageMessagesState:
    planner_config = agent_config["planner"]
    model = create_model(planner_config.model).with_structured_output(StructuredOutput)
    messages = [planner_config.original_prompt] + [HumanMessage(state["task_description"])]
    response = await model.ainvoke(messages, config)
    plan_description = []
    for i, step in enumerate(response.plan, start=1):
        step_description = f"{i}.assistant: {step.assistant}, task_description: {step.task_description}\n"
        plan_description.append(step_description)
    plan = "我是<planner>助手，我已经帮你规划好了各个助手的调用顺序了，请按照以下顺序来调用助手：\n" + "".join(plan_description)
    return {"messages": [AIMessage(plan)]}
