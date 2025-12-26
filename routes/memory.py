import asyncio

from langchain_core.runnables import RunnableConfig
from langchain.messages import HumanMessage
from fastapi import APIRouter, Depends, Body, HTTPException

from core.database import (
    langgraph_persistence_context,
    create_memories,
    batch_update_memories,
    get_memories_by_user,
    get_conversation_by_thread_id,
    delete_memory
)
from core.user import get_current_user
from agents.solving_assistant.agent import create_solving_assistant
from agents.solving_assistant.personalized_memory import summarize_personalized_memory


router = APIRouter(prefix="/memory")


@router.get("/list")
async def get_memories(user: dict = Depends(get_current_user)):
    memories = await get_memories_by_user(user["user_id"])
    for memory in memories:
        memory["created_at"] = memory["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        memory["updated_at"] = memory["updated_at"].strftime("%Y-%m-%d %H:%M:%S")
    return {"memories": memories}


@router.delete("")
async def delete_memory_(
    _: dict = Depends(get_current_user),
    memory_id: int = Body(embed=True)
):
    await delete_memory(memory_id)
    return {"message": "OK"}


@router.post("")
async def create_or_update_memories(
    user: dict = Depends(get_current_user),
    thread_id: str = Body(embed=True)
):
    # 验证身份
    conversation = await get_conversation_by_thread_id(thread_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Not Found")
    user_id = user["user_id"]
    if conversation["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    # 获取历史对话
    config = RunnableConfig(configurable={"thread_id": thread_id})
    async with langgraph_persistence_context() as (checkpointer, store):
        solving_assistant = create_solving_assistant("", "", checkpointer, store)
        state = await solving_assistant.aget_state(config)
        messages = state.values["messages"]
    # 构造对话字符串
    conversations = []
    for message in messages:
        prefix = "Q：" if isinstance(message, HumanMessage) else "A："
        conversations.append(prefix + message.content)
    conversations = "\n".join(conversations)
    # 获取旧记忆
    memories = await get_memories_by_user(user_id)
    old_memories = []
    for memory in memories:
        old_memories.append(str({
            "id": memory["id"],
            "content": memory["content"],
            "type": memory["type"]
        }))
    old_memories = "\n".join(old_memories)
    # 新增或更新记忆
    output_memories = await summarize_personalized_memory(conversations, old_memories)
    should_create_memories, should_update_memoties = [], []
    for memory in output_memories:
        if memory["id"]:
            should_update_memoties.append(memory)
        else:
            should_create_memories.append(memory)
    await asyncio.gather(
        create_memories(should_create_memories, user_id),
        batch_update_memories(should_update_memoties)
    )
    return {
        "message": "OK", 
        "data": {
            "updated": len(should_update_memoties), 
            "created": len(should_create_memories)
        }
    }
