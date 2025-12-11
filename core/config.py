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

    AGENTS_CONFIG_TEMPLATE: dict[str, dict[str, dict]] = {
        "question_manage": {
            "data_preheat": {
                "model": "QUESTION_MANAGE_DATA_PREHEAT_MODEL",
                "prompt_key": "question_manage.data_preheat",
                "tools": {
                    "query_question_info", 
                    "create_question", 
                    "query_all_tags", 
                    "query_all_programming_languages"
                }
            },
            "planner": {
                "model": "QUESTION_MANAGE_PLANNER_MODEL",
                "prompt_key": "question_manage.planner"
            },
            "dispatcher": {
                "prompt_key": "question_manage.dispatcher",
                "model": "QUESTION_MANAGE_DISPATCHER_MODEL"
            },
            "solving_framework": {
                "prompt_key": "question_manage.solving_framework",
                "tools": {
                    "create_solving_framework_for_question",
                    "query_solving_frameworks_of_question",
                    "update_solving_framework_for_question"
                },
                "model": "QUESTION_MANAGE_SOLVING_FRAMEWORK_MODEL"
            },
            "judge_template": {
                "prompt_key": "question_manage.judge_template.%s",
                "tools": {
                    "query_solving_frameworks_of_question",
                    "query_tests_of_question",
                    "create_judge_template_for_question",
                    "query_judge_templates_of_question",
                    "update_judge_template_for_question"
                },
                "model": "QUESTION_MANAGE_JUDGE_TEMPLATE_MODEL"
            },
            "judge_template_dispatcher": {
                "prompt_key": "question_manage.judge_template.dispatcher",
                "model": "QUESTION_MANAGE_JUDGE_TEMPLATE_DISPATCHER_MODEL"
            },
            "memory_time_limit": {
                "prompt_key": "question_manage.memory_time_limit",
                "tools": {
                    "create_memory_time_limit_for_question",
                    "query_memory_time_limits_of_question",
                    "update_memory_time_limit_for_question"
                },
                "model": "QUESTION_MANAGE_MEMORY_TIME_LIMIT_MODEL"
            },
            "test": {
                "prompt_key": "question_manage.test",
                "tools": {
                    "query_tests_of_question", 
                    "create_test_for_question"
                },
                "model": "QUESTION_MANAGE_TEST_MODEL"
            }
        },
        "generic": {
            "json_parser": {
                "prompt_key": "generic.json_parser",
                "model": "GENERIC_JSON_PARSER_MODEL"
            }
        }
    }

    class Config:
        env_file = ".env"

    @property
    def prompt_manager(self):
        if self.__prompt_manager is None:
            self.__prompt_manager = PromptManager()
        return self.__prompt_manager

    @property
    def agents_config(self):
        """获取Agent配置（运行时构建）"""
        if self.__agents_config:
            return self.__agents_config
        
        for agent_type, agents in self.AGENTS_CONFIG_TEMPLATE.items():
            self.__agents_config[agent_type] = {}
            
            for agent_name, config_template in agents.items():
                # 从实例变量获取模型名
                model_field = config_template.get("model")
                model = getattr(self, model_field) if model_field else ""
                
                # 获取或生成原始提示词
                prompt_key = config_template["prompt_key"]
                if "%s" in prompt_key:  # 该提示词需要动态加载
                    original_prompt = ""
                else:
                    original_prompt = self.get_prompt(prompt_key)
                
                # 创建AgentConfig
                self.__agents_config[agent_type][agent_name] = AgentConfig(
                    model=model,
                    prompt_key=prompt_key,
                    original_prompt=original_prompt,
                    tools=config_template.get("tools", set())
                )
        return self.__agents_config

    def get_prompt(self, prompt_key: str) -> str:
        return self.prompt_manager.get_prompt(prompt_key)

    def get_agent_config(self, agent_name: str) -> dict[str, AgentConfig]:
        return self.agents_config.get(agent_name, {})


settings = Settings()
