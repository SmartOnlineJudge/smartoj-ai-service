import asyncio
import json
import traceback

from langchain_core.runnables import RunnableConfig
from langchain.messages import HumanMessage, AIMessage
from fastapi import APIRouter, Cookie, Depends, Body, HTTPException
from fastapi.responses import StreamingResponse
from uvicorn.config import logger

from core.database import (
    get_conversation_by_thread_id, 
    create_conversation, 
    update_conversation_title,
    langgraph_persistence_context
)
from core.user import get_admin_user, get_current_user
from core.request_states import (
    get_interrupted_tasks,
    get_stream_queues,
    get_stream_tasks
)
from utils.checkpointer import generate_thread_id
from agents.generic import generate_title
from agents.question_manage.agent import build_question_manage_graph
from agents.solving_assistant.agent import create_solving_assistant


router = APIRouter(prefix="/chat")


@router.post("/stream")
async def chat_stream(
    thread_id: str = Body(embed=True), 
    user: dict = Depends(get_current_user),
    stream_queues: dict[str, asyncio.Queue] = Depends(get_stream_queues),
    interrupted_tasks: set[str] = Depends(get_interrupted_tasks),
    stream_tasks: dict[str, asyncio.Task] = Depends(get_stream_tasks)
):
    process_id = thread_id + "-" + user["user_id"]
    stream_queue = stream_queues.get(process_id)
    if stream_queue is None:
        raise HTTPException(status_code=404, detail="The target chat is not found")
    async def stream_generator():
        while True:
            if process_id in interrupted_tasks:
                stream_task = stream_tasks.get(process_id)
                if stream_task:
                    stream_task.cancel()
                interrupted_tasks.remove(process_id)
                break
            data = await stream_queue.get()
            if data is None:
                break
            json_data = json.dumps(data)
            yield f"data: {json_data}\n\n"
        yield "data: DONE\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.post("/interrupt")
async def interrupt_chat(
    thread_id: str = Body(embed=True),
    user: dict = Depends(get_current_user),
    interrupted_tasks: set[str] = Depends(get_interrupted_tasks)
):
    interrupted_tasks.add(thread_id + "-" + user["user_id"])
    return {"message": "OK"}


@router.post("/question-manage")
async def invoke_question_manage_agent(
    session_id: str = Cookie(),
    admin: dict = Depends(get_admin_user),
    query: str = Body(),
    thread_id: str = Body(default_factory=generate_thread_id),
    stream_queues: dict[str, asyncio.Queue] = Depends(get_stream_queues),
    stream_tasks: dict[str, asyncio.Task] = Depends(get_stream_tasks)
):
    config = RunnableConfig(configurable={"thread_id": thread_id, "backend-session-id": session_id})
    agent_input = {"messages": [HumanMessage(query)]}
    stream_queue = asyncio.Queue(200)
    user_id = admin["user_id"]
    process_id = thread_id + "-" + user_id
    stream_queues[process_id] = stream_queue

    async def agent_invoke():
        async with langgraph_persistence_context() as (checkpointer, store):
            graph = build_question_manage_graph(checkpointer, store)
            try:
                async for namespace, stream_mode, data in graph.astream(
                    agent_input, config, stream_mode=["messages", "custom"], subgraphs=True
                ):
                    if stream_mode == "custom":
                        await stream_queue.put(data)
                        continue
                    if not namespace:
                        continue
                    message_chunk, _ = data
                    message_id = message_chunk.id
                    if message_id[:6] != "lc_run":
                        continue
                    content = message_chunk.content
                    if not content:
                        continue
                    name = namespace[-1].split(":")[0]    
                    await stream_queue.put({
                        "content": content,
                        "id": message_id,
                        "node": name,
                        "type": "assistant"
                    })
                conversation = await get_conversation_by_thread_id(thread_id)
                if conversation:
                    title = conversation["title"]
                    await update_conversation_title(conversation["id"], title)  # 更新对话活跃时间
                    return
                # 新对话
                snapshot = await graph.aget_state(config)
                assistant_messages = []
                for message in snapshot.values["messages"]:
                    if isinstance(message, AIMessage):
                        assistant_messages.append(message.content)
                answer = "".join(assistant_messages)
                title = await generate_title(query, answer)
                await create_conversation(title, user_id, None, thread_id)
            finally:
                await stream_queue.put(None)

    def task_done_callback(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("任务<%s>被取消", process_id)
        except Exception as _:
            logger.info("任务<%s>异常", process_id)
            traceback.print_exc()
        else:
            logger.info("任务<%s>完成", process_id)
        finally:
            stream_queues.pop(process_id, None)
            stream_tasks.pop(process_id, None)

    task = asyncio.create_task(agent_invoke())
    task.add_done_callback(task_done_callback)
    stream_tasks[process_id] = task

    return {"thread_id": thread_id}


@router.post("/solving-assistant")
async def invoke_solving_assistant_agent(
    user: dict = Depends(get_current_user),
    query: str = Body(),
    question_description: str = Body(),
    code: str = Body(),
    question_id: int = Body(),
    thread_id: str = Body(default_factory=generate_thread_id),
    stream_queues: dict[str, asyncio.Queue] = Depends(get_stream_queues),
    stream_tasks: dict[str, asyncio.Task] = Depends(get_stream_tasks)
):
    config = RunnableConfig(configurable={"thread_id": thread_id})
    agent_input = {"messages": [HumanMessage(query)]}
    stream_queue = asyncio.Queue(200)
    user_id = user["user_id"]
    process_id = thread_id + "-" + user_id
    stream_queues[process_id] = stream_queue

    async def agent_invoke():
        async with langgraph_persistence_context() as (checkpointer, store):
            agent = create_solving_assistant(question_description, code, checkpointer, store)
            try:
                async for message_chunk, _ in agent.astream(agent_input, config, stream_mode="messages"):
                    content = message_chunk.content
                    if not content:
                        continue
                    await stream_queue.put({
                        "content": content,
                        "id": message_chunk.id,
                        "node": "solving_assistant",
                        "type": "assistant"
                    })
                conversation = await get_conversation_by_thread_id(thread_id)
                if conversation:
                    return
                await create_conversation("", user_id, question_id, thread_id)
            finally:
                await stream_queue.put(None)

    def task_done_callback(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("任务<%s>被取消", process_id)
        except Exception as _:
            logger.info("任务<%s>异常", process_id)
            traceback.print_exc()
        else:
            logger.info("任务<%s>完成", process_id)
        finally:
            stream_queues.pop(process_id, None)
            stream_tasks.pop(process_id, None)

    task = asyncio.create_task(agent_invoke())
    task.add_done_callback(task_done_callback)
    stream_tasks[process_id] = task

    return {"thread_id": thread_id}
