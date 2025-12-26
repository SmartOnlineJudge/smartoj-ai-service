from core.config import settings
from core.model import create_model


async def generate_title(question: str, answer: str):
    model = create_model(
        settings.GENERIC_CHAT_TITLE_GENERATOR_MODEL,
        settings.OPENAI_API_KEY,
        settings.OPENAI_BASE_URL,
        streaming=False
    )
    prompt = """
        为以下对话生成一个简短的标题：\n
        Q：%s\n
        A: %s\n
        输出格式：你只需要输出一个标题，不要输出其他内容。
    """
    response = await model.ainvoke(prompt % (question, answer))
    return response.content
