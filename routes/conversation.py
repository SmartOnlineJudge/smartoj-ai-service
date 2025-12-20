from langchain_core.runnables import RunnableConfig
from fastapi import APIRouter, Depends, HTTPException, Query, Body

from core.database import (
    get_conversation_by_thread_id, 
    get_conversations_by_user_and_question,
    update_conversation_title,
    delete_conversation,
    langgraph_persistence_context
)
from core.user import get_admin_user, get_current_user
from ai_services.agents.question_manage.agent import build_question_manage_graph


router = APIRouter(prefix="/conversation")


@router.get("/list")
async def get_conversations(
    user: dict = Depends(get_current_user), 
    question_id: int = Query(None)
):
    conversations = await get_conversations_by_user_and_question(user["user_id"], question_id)
    for conversation in conversations:
        conversation["created_at"] = conversation["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        conversation["updated_at"] = conversation["updated_at"].strftime("%Y-%m-%d %H:%M:%S")
    return {"conversations": conversations}


@router.get("")
async def get_conversation(thread_id: str = Query()):
    conversation = await get_conversation_by_thread_id(thread_id)
    if not conversation:
        return {"conversation": {}}
    conversation["created_at"] = conversation["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    conversation["updated_at"] = conversation["updated_at"].strftime("%Y-%m-%d %H:%M:%S")
    return {"conversation": conversation}


@router.patch("")
async def modify_conversation_title(
    title: str = Body(),
    thread_id: str = Body(),
    user: dict = Depends(get_current_user)
):
    conversation = await get_conversation_by_thread_id(thread_id)
    if not conversation:
        return {"success": False}
    if conversation["user_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not await update_conversation_title(conversation["id"], title):
        return {"success": False}
    return {"success": True}


@router.delete("")
async def destory_conversation(
    thread_id: str = Body(embed=True),
    user: dict = Depends(get_current_user)
):
    conversation = await get_conversation_by_thread_id(thread_id)
    if not conversation:
        return {"success": False}
    if conversation["user_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not await delete_conversation(conversation["id"]):
        return {"success": False}
    return {"success": True}


@router.get("/detail/question-manage")
async def get_question_manage_agent_conversation_detail(
    thread_id: str = Query(),
    admin: dict = Depends(get_admin_user),
):
    conversation = await get_conversation_by_thread_id(thread_id)
    if not conversation:
        return {"details": []}
    if conversation["user_id"] != admin["user_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    config = RunnableConfig(configurable={"thread_id": thread_id})
    async with langgraph_persistence_context() as (checkpointer, store):
        graph = build_question_manage_graph(checkpointer, store)
        snapshot = await graph.aget_state(config)
    message_type_mapping = {
        "human": "user",
        "ai": "assistant",
        "tool_call": "tool_call",
        "tool": "tool_call_result",
    }
    details = []
    for message in snapshot.values["display_messages"]:
        message = message.model_dump(include={"content", "type", "id", "tool_calls", "tool_call_id"})
        message["type"] = message_type_mapping[message.pop("type")]
        if message["type"] == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                details.append({
                    "name": tool_call["name"], 
                    "args": tool_call["args"], 
                    "id": tool_call["id"],
                    "type": "tool_call",
                })
            continue
        if message["type"] == "tool_call_result":
            message["id"] = message["tool_call_id"]
            message["result"] = message.pop("content")
        details.append(message)
    return {"details": details}
