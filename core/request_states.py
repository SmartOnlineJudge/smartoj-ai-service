import asyncio

from fastapi import Request


def get_stream_queues(request: Request) -> dict[str, asyncio.Queue]:
    return getattr(request.state, "stream_queues")


def get_stream_tasks(request: Request) -> dict[str, asyncio.Task]:
    return getattr(request.state, "stream_tasks")


def get_interrupted_tasks(request: Request) -> set[str]:
    return getattr(request.state, "interrupted_tasks")
