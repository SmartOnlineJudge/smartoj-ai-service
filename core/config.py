from pydantic_settings import BaseSettings

from prompts.manager import PromptManager


class Settings(BaseSettings):
    # 提示词管理器
    __prompt_manager = None

    # AI 服务相关配置
    OPENAI_BASE_URL: str
    OPENAI_API_KEY: str

    # Graph Node LLM 模型配置
    # 每个图的每个节点对应的的 LLM 模型
    QUESTION_MANAGE_DISPATCHER_MODEL: str
    QUESTION_MANAGE_MEMORY_TIME_LIMIT_MODEL: str
    QUESTION_MANAGE_TEST_MODEL: str
    QUESTION_MANAGE_SOLVING_FRAMEWORK_MODEL: str
    QUESTION_MANAGE_JUDGE_TEMPLATE_MODEL: str
    QUESTION_MANAGE_JUDGE_TEMPLATE_DISPATCHER_MODEL: str
    QUESTION_MANAGE_PLANNER_MODEL: str
    QUESTION_MANAGE_QUESTION_MODEL: str

    # MCP 连接配置
    MCP_SERVER_URL: str

    class Config:
        env_file = ".env"

    @property
    def prompt_manager(self):
        if self.__prompt_manager is None:
            self.__prompt_manager = PromptManager()
        return self.__prompt_manager


settings = Settings()
