from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from core.database import ConnectionManager, langgraph_persistence_context
from core.config import settings
from routes.chat import router as chat_router
from routes.conversation import router as conversation_router
from routes.memory import router as memory_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    await ConnectionManager.initialize(settings.DATABASE_URI)
    async with langgraph_persistence_context() as (checkpointer, _):
        await checkpointer.setup()
        yield {
            "stream_queues": {},
            "stream_tasks": {},
            "interrupted_tasks": set()
        }
    await ConnectionManager.close()


app = FastAPI(lifespan=lifespan)

app.include_router(chat_router)
app.include_router(conversation_router)
app.include_router(memory_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_config="log-config.json")
