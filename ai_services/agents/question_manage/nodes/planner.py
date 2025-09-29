from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from core.node import SmartOJNode, SmartOJMessagesState
from core.config import settings


class Step(BaseModel):
    assistant: str = Field(description="需要使用的助手名字")
    description: str = Field(description="需要完成的任务的描述")


class StructuredOutput(BaseModel):
    plan: list[Step]


class PlannerNode(SmartOJNode):
    model = settings.QUESTION_MANAGE_PLANNER_MODEL
    prompt_key = "question_manage.planner"

    def __init__(self):
        super().__init__()
        self.llm = self.llm.with_structured_output(StructuredOutput)

    async def __call__(self, state: SmartOJMessagesState, config: RunnableConfig):
        messages = [self.prompt] + [HumanMessage(state["description"])]
        response = await self.llm.ainvoke(messages, config)
        plan_desctiption = []
        for i, step in enumerate(response.plan, start=1):
            step_description = f"{i}. assistant: {step.assistant}, description: {step.description}\n"
            plan_desctiption.append(step_description)
        plan_prompt = HumanMessage("请按照以下顺序来调用助手：\n" + "".join(plan_desctiption))
        return {"messages": [plan_prompt]}
