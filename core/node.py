from .state import SmartOJMessagesState


class SmartOJNode:
    async def __call__(self, state: SmartOJMessagesState, *args, **kwargs):
        raise NotImplementedError
