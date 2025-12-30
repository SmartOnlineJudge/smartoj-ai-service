# 智能算法刷题平台——AI  服务

## 🖼️仓库介绍

本仓库为智能算法刷题平台提供了相关的 AI 服务，主要包括题目信息智能管理助手和智能刷题助手。前者聚焦在后台管理系统中，而后者主要用于在线刷题页面。

## ✨技术栈

1. LangChain：调用大模型实现简单智能体构建
2. LangGraph：基于图结构构建复杂智能体
3. FastAPI：向外部提供智能体调用 API

## 🤖相关智能体

### 题目信息管理助手

这是基于 LangGraph 构建的一个多智能体协同的工作流系统，相当于一个复杂的智能体。其架构如下：

![](https://cdn.nlark.com/yuque/0/2025/png/47866636/1760153373921-15478ebb-b997-47a0-8df2-0c977e4c3a11.png?x-oss-process=image%2Fformat%2Cwebp)

包括以下几个子智能体（节点）：

1. 数据预热助手（date_preheat）

   主要负责与用户对话，并为后续的节点提供数据支持。

2. 任务调用助手（dispatcher）

   根据用户的需求，动态调度其他的助手，以此来完成负责的任务。

3. 任务规划助手（planner）

   将一个负责任务拆分成多个可完成的小任务，并返回一个任务规划列表。

4. 测试用例助手（test）

   专门负责处理测试用例相关的信息。

5. 解题框架助手（solving_framework）

   专门负责处理解题框架相关的信息。

6. 内存时间限制助手（memory_time_limit）

   专门负责处理内存时间限制相关的信息。

7. Python判题模板助手（judge_template）

   专门负责处理Python判题模板相关的信息。

它专门用于管理系统中题目的相关数据，适合处理任务，不适合日常对话。

一个清晰、简洁的指令可以让它精准的完成任务。

例如：“帮我完善xxx题目的测试用例、判题模板、解题框架”。这条指令说明了需要操作的对象：“xxx题目”，指出了需要完成什么样的任务：“测试用例、判题模板、解题框架”。

当它收到这条指令的时候，会先由数据预热助手来处理，它会调用 MCP 工具来查询当前题目的相关数据，例如：当前题目的描述、当前题目的 ID 等等关键信息。最后将这些关键信息保存到 LangGraph 的 State 中；接下来由任务调度助手来执行任务，这个任务涉及到 3 个子任务，因此它会先让任务规划助手来规划这个任务，最终拿到一个任务执行列表，然后按照这个列表中的顺序来分别调用每一位数据处理助手，例如：测试用例助手、Python 判题模板助手、解题框架助手。

> 每一次执行任务最好只涉及 1 道题目的操作。

![](https://cdn.nlark.com/yuque/0/2025/png/47866636/1767088028065-3f35bc9a-7496-4931-be0a-7552e8d919b3.png?x-oss-process=image%2Fformat%2Cwebp)

在后台管理界面的题目信息管理中，可以在表格中勾选出想要操作的题目，然后将任务发送给它。

对话期间可能还有调用工具的消息：

![](https://cdn.nlark.com/yuque/0/2025/png/47866636/1767088140224-6cda332a-02cd-433a-af37-f86988a245ea.png?x-oss-process=image%2Fformat%2Cwebp)

总之它只适合处理任务，不适合日常对话。

### 智能刷题助手

基于用户记忆的个性化智能刷题助手，存在于客户端的刷题界面中，不同题目之间的对话相互隔离：

![](https://cdn.nlark.com/yuque/0/2025/png/47866636/1767088640225-f4973c61-38a4-4213-8485-9f30d677a9e9.png?x-oss-process=image%2Fformat%2Cwebp)

在对话的时候会被注入以下信息：

1. 当前的题目信息
2. 用户当前的解题代码
3. 用户名
4. 用户的记忆

这使得它在不同的题目、不同的用户中会表现出完全不同的风格。

其中当对话超过一定轮次的时候，会触发记忆更新的功能：

![](https://cdn.nlark.com/yuque/0/2025/png/47866636/1767089488406-f48b6c98-0764-4354-885c-87965ad55aa1.png?x-oss-process=image%2Fformat%2Cwebp)

### 通用模型

一共有两个通用模型：

1. JSON 结构化输出模型

   给定一个结构化 Schema，专门负责从文本中提取出结构化数据。

2. 聊天标题生成模型

   用于题目信息管理中生成对话标题。

## ⚒️启动项目

### 基本环境

在启动项目之前需要先将后端和 MCP 服务器启动，不然部分功能可能无法使用。

1. 后端仓库：[https://github.com/SmartOnlineJudge/smartoj-backend](https://github.com/SmartOnlineJudge/smartoj-backend)
2. MCP 仓库：[https://github.com/SmartOnlineJudge/smartoj-mcp-server](https://github.com/SmartOnlineJudge/smartoj-mcp-server)

项目使用 uv 来管理 Python 的第三方库，并且 Python 的版本要大于等于 3.11。否则项目将无法启动。

### 创建数据库及相关表结构

项目的数据库使用的是 MySQL，并且确保 MySQL 的版本大于等于 8.0。如果小于这个版本，那么将无法启动项目。

项目依赖一个名为`smartojai`的数据库，在创建的时候需要注意该数据库的编码格式和排序方式，特别是排序方式：

```sql
CREATE DATABASE smartojai
  DEFAULT COLLATE SET = 'utf8mb4_0900_ai_ci'  # 排序方式
  DEFAULT CHARACTER SET = 'utf8mb4';  # 编码方式
```

创建好数据库以后，还需要再创建两张数据表：

```sql
CREATE TABLE conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    user_id VARCHAR(13) NOT NULL,
    question_id INT DEFAULT NULL,
    thread_id VARCHAR(128) NOT NULL UNIQUE,
    INDEX idx_user_id_is_deleted_question_id (user_id, is_deleted, question_id),
    INDEX idx_updated_at (updated_at)
)

CREATE TABLE memories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(13) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    content VARCHAR(50) NOT NULL,
    type ENUM("level", "ability", "preference"),
    INDEX idx_user_id (user_id)
)
```

一张表用户保存对话记录，一张表用于保存用户的记忆。

### 编写配置文件

编写 .env 配置文件，先复制一份：

```bash
cp .env.example .env
```

将后端、MCP、MySQL的配置写入其中：

```
# MCP 连接配置
MCP_SERVER_URL=http://127.0.0.1:9000/mcp

# 后端接口地址
BACKEND_URL=http://127.0.0.1:8000

# MySQL URI
DATABASE_URI=mysql://root:password@localhost:3306/smartojai
```

然后就是模型配置：

```
# AI 服务相关配置
# 需要符合 OpenAI 接口的规范
OPENAI_API_KEY=<your api key>
OPENAI_BASE_URL=<your base url>

# Graph Node LLM 模型配置
# 题目信息管理 Agent 的每个节点对应的的 LLM 模型
QUESTION_MANAGE_DISPATCHER_MODEL=Qwen/Qwen3-8B
QUESTION_MANAGE_MEMORY_TIME_LIMIT_MODEL=deepseek-ai/DeepSeek-V3.1
QUESTION_MANAGE_TEST_MODEL=deepseek-ai/DeepSeek-V3.1
QUESTION_MANAGE_SOLVING_FRAMEWORK_MODEL=deepseek-ai/DeepSeek-V3.1
QUESTION_MANAGE_JUDGE_TEMPLATE_FOR_PYTHON_MODEL=deepseek-ai/DeepSeek-V3.1
QUESTION_MANAGE_PLANNER_MODEL=Qwen/Qwen3-8B
QUESTION_MANAGE_DATA_PREHEAT_MODEL=deepseek-ai/DeepSeek-V3.1
# 通用 Agent LLM 配置
GENERIC_JSON_PARSER_MODEL=deepseek-ai/DeepSeek-V3.1
GENERIC_CHAT_TITLE_GENERATOR_MODEL=qwen3-30b-a3b
# 智能刷题助手 LLM 配置
SOLVING_ASSISTANT_MODEL=deepseek-ai/DeepSeek-V3.1
# 个性化记忆 LLM 配置
PERSONALIZED_MEMORY_MODEL=deepseek-ai/DeepSeek-V3.1
```

大模型提供商的 API 需要符合 OpenAI 格式，否则无法调用大模型。

对于模型配置方面，以`QUESTION_MANAGE`开头的全部模型都推荐使用 Qwen3-Max。因为在开发的时候，这个模型是最稳定的，输出很符合预期结果，但是其他模型就不一定了。而以`GENERIC`开头的模型推荐使用参数两较小的模型即可，因为任务足够简单。而剩下的智能刷题助手、个性化记忆这两个用什么模型基本上问题不大。

编写好配置以后，安装项目依赖：

```bash
uv sync
```

启动项目：

```bash
uv run main.py

INFO:     2025-12-29 21:31:36,829 - Started server process [115855]
INFO:     2025-12-29 21:31:36,829 - Waiting for application startup.
/path/to/smartoj/smartoj-ai-service/.venv/lib/python3.11/site-packages/aiomysql/cursors.py:458: Warning: Table 'checkpoint_migrations' already exists
  await self._do_get_result()
INFO:     2025-12-29 21:31:36,855 - Application startup complete.
INFO:     2025-12-29 21:31:36,855 - Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

首次运行该项目的时候，程序会先创建 LangGraph 相关的数据库表结构，因此速度可能会比较慢一些。

看到上面的输出就代表项目已经启动了。