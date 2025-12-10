from pydantic import BaseModel
from pydantic_settings import BaseSettings

from prompts.manager import PromptManager


class AgentConfig(BaseModel):
    model: str
    prompt_key: str
    original_prompt: str
    tools: set[str] = set()


class Settings(BaseSettings):
    # 提示词管理器
    __prompt_manager = None

    # Agents 配置
    __agents_config: dict[str, dict[str, AgentConfig]] = {}

    # AI 服务相关配置
    OPENAI_BASE_URL: str
    OPENAI_API_KEY: str

    # Graph Node LLM 模型配置
    # 题目信息管理 Agent 的每个节点对应的的 LLM 模型
    QUESTION_MANAGE_DISPATCHER_MODEL: str
    QUESTION_MANAGE_MEMORY_TIME_LIMIT_MODEL: str
    QUESTION_MANAGE_TEST_MODEL: str
    QUESTION_MANAGE_SOLVING_FRAMEWORK_MODEL: str
    QUESTION_MANAGE_JUDGE_TEMPLATE_MODEL: str
    QUESTION_MANAGE_JUDGE_TEMPLATE_DISPATCHER_MODEL: str
    QUESTION_MANAGE_PLANNER_MODEL: str
    QUESTION_MANAGE_DATA_PREHEAT_MODEL: str
    # 通用 Agent LLM 配置
    GENERIC_JSON_PARSER_MODEL: str

    # MCP 连接配置
    MCP_SERVER_URL: str

    class Config:
        env_file = ".env"

    @property
    def prompt_manager(self):
        if self.__prompt_manager is None:
            self.__prompt_manager = PromptManager()
        return self.__prompt_manager

    @property
    def agents_config(self):
        if self.__agents_config:
            return self.__agents_config
        self.__agents_config = {
            "question_manage": {
                "data_preheat": AgentConfig(
                    model=self.QUESTION_MANAGE_DATA_PREHEAT_MODEL, 
                    prompt_key="question_manage.data_preheat",
                    original_prompt=self.get_prompt("question_manage.data_preheat"),
                    tools={
                        "query_question_info", 
                        "create_question", 
                        "query_all_tags", 
                        "query_all_programming_languages"
                    }
                ),
                "planner": AgentConfig(
                    model=self.QUESTION_MANAGE_PLANNER_MODEL,
                    prompt_key="question_manage.planner",
                    original_prompt=self.get_prompt("question_manage.planner"),
                ),
                "dispatcher": AgentConfig(
                    model=self.QUESTION_MANAGE_DISPATCHER_MODEL,
                    prompt_key="question_manage.dispatcher",
                    original_prompt=self.get_prompt("question_manage.dispatcher"),
                ),
                "solving_framework": AgentConfig(
                    model=self.QUESTION_MANAGE_SOLVING_FRAMEWORK_MODEL,
                    prompt_key="question_manage.solving_framework",
                    original_prompt=self.get_prompt("question_manage.solving_framework"),
                    tools={
                        "create_solving_framework_for_question",
                        "query_solving_frameworks_of_question",
                        "update_solving_framework_for_question"
                    }
                ),
                "judge_template": AgentConfig(
                    model=self.QUESTION_MANAGE_JUDGE_TEMPLATE_MODEL,
                    prompt_key="question_manage.judge_template.%s",
                    original_prompt="",
                    tools={
                        "query_solving_frameworks_of_question",
                        "query_tests_of_question",
                        "create_judge_template_for_question",
                        "query_judge_templates_of_question",
                        "update_judge_template_for_question"
                    }
                ),
                "judge_template_dispatcher": AgentConfig(
                    model=self.QUESTION_MANAGE_JUDGE_TEMPLATE_DISPATCHER_MODEL,
                    prompt_key="question_manage.judge_template.dispatcher",
                    original_prompt=self.get_prompt("question_manage.judge_template.dispatcher"),
                ),
                "memory_time_limit": AgentConfig(
                    model=self.QUESTION_MANAGE_MEMORY_TIME_LIMIT_MODEL,
                    prompt_key="question_manage.memory_time_limit",
                    original_prompt=self.get_prompt("question_manage.memory_time_limit"),
                    tools={
                        "create_memory_time_limit_for_question",
                        "query_memory_time_limits_of_question",
                        "update_memory_time_limit_for_question"
                    }
                ),
                "test": AgentConfig(
                    model=self.QUESTION_MANAGE_TEST_MODEL,
                    prompt_key="question_manage.test",
                    original_prompt=self.get_prompt("question_manage.test"),
                    tools={
                        "query_tests_of_question", 
                        "create_test_for_question"
                    }
                )
            },
            "generic": {
                "json_parser": AgentConfig(
                    model=self.GENERIC_JSON_PARSER_MODEL,
                    prompt_key="generic.json_parser",
                    original_prompt=self.get_prompt("generic.json_parser"),
                )
            },
        }
        return self.__agents_config

    def get_prompt(self, prompt_key: str) -> str:
        return self.prompt_manager.get_prompt(prompt_key)

    def get_agent_config(self, agent_name: str) -> dict[str, AgentConfig]:
        return self.agents_config.get(agent_name, {})


settings = Settings()
