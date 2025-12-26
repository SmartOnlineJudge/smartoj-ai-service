from urllib.parse import urlparse
from typing import Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import aiomysql
from aiomysql.cursors import DictCursor
from aiomysql.pool import Pool
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from langgraph.store.mysql import AIOMySQLStore


class ConnectionManager:
    _generic_pool: Optional[Pool] = None
    _langgraph_pool: Optional[Pool] = None
    
    @classmethod
    def _parse_conn_string(cls, conn_string: str) -> dict[str, Any]:
        parsed = urlparse(conn_string)
        return {
            "host": parsed.hostname or "localhost",
            "user": parsed.username,
            "password": parsed.password or "",
            "db": parsed.path[1:] or None,
            "port": parsed.port or 3306
        }

    @classmethod
    async def initialize(cls, connection_string: str, minsize: int = 1, maxsize: int = 15):
        """初始化连接池"""
        _conn_config = cls._parse_conn_string(connection_string)
        if cls._generic_pool is None:
            cls._generic_pool = await aiomysql.create_pool(
                **_conn_config,
                minsize=minsize,
                maxsize=maxsize,
                autocommit=False,
                cursorclass=DictCursor,
                pool_recycle=30,
                connect_timeout=10
            )
        if cls._langgraph_pool is None:
            cls._langgraph_pool = await aiomysql.create_pool(
                **_conn_config,
                minsize=minsize,
                maxsize=maxsize,
                autocommit=True,
                cursorclass=DictCursor,
                pool_recycle=30,
                connect_timeout=10
            )
    
    @classmethod
    async def close(cls):
        """关闭连接池"""
        if cls._generic_pool:
            cls._generic_pool.close()
            await cls._generic_pool.wait_closed()
            cls._generic_pool = None
        if cls._langgraph_pool:
            cls._langgraph_pool.close()
            await cls._langgraph_pool.wait_closed()
            cls._langgraph_pool = None

    @classmethod
    @asynccontextmanager
    async def connection(cls, pool_type: str = "generic") -> AsyncGenerator[aiomysql.Connection, None]:
        target_pool = cls._langgraph_pool if pool_type == "langgraph" else cls._generic_pool
        async with target_pool.acquire() as conn:
            yield conn


@asynccontextmanager
async def langgraph_persistence_context():
    async with ConnectionManager.connection(pool_type="langgraph") as conn:
        yield AIOMySQLSaver(conn), AIOMySQLStore(conn)


async def create_conversation(
    title: str, 
    user_id: str, 
    question_id: int, 
    thread_id: str
) -> int:
    sql = "INSERT INTO conversations (title, user_id, question_id, thread_id) VALUES (%s, %s, %s, %s)"
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, (title, user_id, question_id, thread_id))
            await conn.commit()
            return cursor.lastrowid


async def delete_conversation(conversation_id: int):
    sql = "UPDATE conversations SET is_deleted = TRUE WHERE id = %s"
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, (conversation_id,))
            await conn.commit()
            return cursor.rowcount > 0


async def get_conversations_by_user_and_question(user_id: str, question_id: int | None = None) -> list[dict]:
    if question_id is None:
        sql = """
            SELECT id, title, created_at, updated_at, user_id, question_id, thread_id 
            FROM conversations 
            WHERE user_id = %s AND question_id is NULL AND is_deleted = FALSE
            ORDER BY updated_at DESC
        """
        args = (user_id,)
    else:
        sql = """
            SELECT id, title, created_at, updated_at, user_id, question_id, thread_id 
            FROM conversations 
            WHERE user_id = %s AND question_id = %s AND is_deleted = FALSE
            ORDER BY updated_at DESC
        """
        args = (user_id, question_id)
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, args)
            return await cursor.fetchall()


async def update_conversation_title(conversation_id: int, title: str) -> bool:
    sql = "UPDATE conversations SET title = %s WHERE id = %s"
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, (title, conversation_id))
            await conn.commit()
            return cursor.rowcount > 0


async def get_conversation_by_thread_id(thread_id: str) -> dict:
    sql = """
        SELECT id, title, created_at, updated_at, user_id, question_id, thread_id 
        FROM conversations 
        WHERE thread_id = %s AND is_deleted = FALSE
    """
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, (thread_id,))
            conversation = await cursor.fetchone()
            return conversation or {}


async def create_memories(memories: list[dict], user_id: str) -> list[int]:
    sql = "INSERT INTO memories (user_id, content, type) VALUES (%s, %s, %s)"
    results = []
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            for memory in memories:
                await cursor.execute(sql, (user_id, memory["content"], memory["type"]))
                results.append(cursor.lastrowid)
            await conn.commit()
            return results


async def get_memories_by_user(user_id: str) -> list[dict]:
    sql = """
        SELECT id, user_id, created_at, updated_at, content, type
        FROM memories
        WHERE user_id = %s
        ORDER BY updated_at DESC
    """
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, (user_id,))
            return await cursor.fetchall()


async def delete_memory(memory_id: int) -> bool:
    sql = "DELETE FROM memories WHERE id = %s"
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, (memory_id,))
            await conn.commit()
            return cursor.rowcount > 0


async def update_memory_content(memory_id: int, content: str) -> bool:
    sql = "UPDATE memories SET content = %s WHERE id = %s"
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, (content, memory_id))
            await conn.commit()
            return cursor.rowcount > 0


async def batch_update_memories(memories: list[dict]):
    if not memories:
        return
    sql = "UPDATE memories SET content = %s WHERE id = %s"
    async with ConnectionManager.connection() as conn:
        async with conn.cursor() as cursor:
            for memory in memories:
                await cursor.execute(sql, (memory["content"], memory["id"]))
            await conn.commit()
