from langchain_openai import ChatOpenAI

from core.config import settings


def create_model(
    model_name: str,
    api_key: str = settings.OPENAI_API_KEY,
    base_url: str = settings.OPENAI_BASE_URL,
    extra_body: dict = None,
    streaming: bool = True,
    **kwargs
) -> ChatOpenAI:
    if extra_body is None:
        extra_body = {"enable_thinking": False}
    else:
        if "enable_thinking" not in extra_body:
            extra_body.update({"enable_thinking": False})
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body,
        streaming=streaming,
        **kwargs
    )
