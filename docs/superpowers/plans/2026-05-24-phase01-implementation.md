# Phase 1: v1.x 新架构与核心 API 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 LangChain v1.x 学习环境，掌握 `create_agent`、LCEL、结构化输出、流式输出等核心 API，完成多模型切换聊天器 Checkpoint 项目

**Architecture:** 使用 UV 管理 Python 项目，按阶段分目录组织代码（`phase01/`），每个知识点对应一个练习脚本，最终整合为 Checkpoint 项目

**Tech Stack:** Python 3.12+, UV, langchain v1.x, langchain-openai, langchain-anthropic, langchain-google-genai, langgraph, pydantic v2

---

## File Structure

```
lzy-langgraph/
├── pyproject.toml                    # UV 项目配置，依赖声明
├── .env.example                      # API Key 模板
├── .gitignore
├── phase01/
│   ├── 01_hello_agent.py             # Task 1: 第一个 create_agent
│   ├── 02_model_strings.py           # Task 2: 统一模型字符串
│   ├── 03_lcel_basics.py             # Task 3: LCEL 基础
│   ├── 04_structured_output.py       # Task 4: 结构化输出
│   ├── 05_streaming.py              # Task 5: 流式输出
│   ├── 06_deep_agents_intro.py       # Task 6: Deep Agents 初探
│   └── checkpoint/
│       ├── chat_multi_model.py       # Checkpoint: 多模型切换聊天器
│       └── README.md                 # Checkpoint 项目说明
├── docs/
│   └── superpowers/
│       ├── specs/                    # 已有
│       └── plans/                    # 已有
```

---

### Task 0: 项目初始化与环境搭建

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`

- [ ] **Step 1: 初始化 UV 项目**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv init --name lzy-langgraph --python 3.12
```

- [ ] **Step 2: 配置 pyproject.toml 依赖**

在 `pyproject.toml` 的 `[project.dependencies]` 中添加：

```toml
[project]
name = "lzy-langgraph"
version = "0.1.0"
description = "LangChain v1.x + LangGraph learning project"
requires-python = ">=3.12"
dependencies = [
    "langchain>=0.1.0",
    "langchain-openai>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-google-genai>=2.1.0",
    "langgraph>=0.4.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]
```

- [ ] **Step 3: 安装依赖**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv sync
```

Expected: 依赖安装成功，生成 `uv.lock`

- [ ] **Step 4: 创建 .env.example**

```env
# 复制此文件为 .env 并填入你的 API Key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=lzy-langgraph-phase01
```

- [ ] **Step 5: 创建 .gitignore**

```gitignore
.env
__pycache__/
*.pyc
.venv/
.uv/
uv.lock
```

- [ ] **Step 6: 创建 .env 文件（从模板复制）**

Run:
```bash
cp .env.example .env
```

然后手动填入真实 API Key。

- [ ] **Step 7: 创建 phase01 目录**

Run:
```bash
mkdir -p phase01/checkpoint
```

- [ ] **Step 8: 验证环境**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python -c "from langchain.agents import create_agent; print('langchain OK')"
uv run python -c "import langgraph; print('langgraph OK')"
uv run python -c "import pydantic; print(f'pydantic {pydantic.__version__}')"
```

Expected: 三个包都能正常导入

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .env.example .gitignore phase01/.gitkeep
git commit -m "chore: init project with UV and dependencies"
```

---

### Task 1: 第一个 create_agent（Day 1-2）

**学习目标：** 理解 v1.x `create_agent` API，跑通第一个 Agent

**Files:**
- Create: `phase01/01_hello_agent.py`

- [ ] **Step 1: 编写第一个 Agent 脚本**

```python
"""
Phase01 - Task 1: 第一个 create_agent
学习目标：用 v1.x 新 API 创建一个简单的工具调用 Agent

运行：uv run python phase01/01_hello_agent.py
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()


def get_current_time(city: str) -> str:
    """获取指定城市的当前时间（模拟）"""
    from datetime import datetime, timezone, timedelta

    timezones = {
        "北京": "Asia/Shanghai",
        "上海": "Asia/Shanghai",
        "东京": "Asia/Tokyo",
        "纽约": "America/New_York",
        "伦敦": "Europe/London",
        "旧金山": "America/Los_Angeles",
    }

    tz_name = timezones.get(city, "UTC")
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    return f"{city} 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})"


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def main():
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[get_current_time, calculate],
        system_prompt="你是一个友好的助手，可以查询时间和做简单计算。用中文回答。",
    )

    print("=== 第一个 LangChain v1.x Agent ===")
    print("输入 'quit' 退出\n")

    messages = []
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "quit":
            print("再见！")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]

        ai_content = result["messages"][-1]
        if hasattr(ai_content, "content_blocks"):
            for block in ai_content.content_blocks:
                if hasattr(block, "text"):
                    print(f"AI: {block.text}")
        elif hasattr(ai_content, "content"):
            print(f"AI: {ai_content.content}")
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行测试**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python phase01/01_hello_agent.py
```

测试输入：
- "北京现在几点？"
- "123 * 456 等于多少？"

Expected: Agent 正确调用工具并返回结果

- [ ] **Step 3: 记录观察笔记**

在脚本同目录下创建一个简单注释，记录：
- `create_agent` 返回的对象类型
- 工具被调用的流程（先选择工具 → 执行工具 → 整合结果）
- 与旧版 `AgentExecutor` 的区别（如已知）

- [ ] **Step 4: Commit**

```bash
git add phase01/01_hello_agent.py
git commit -m "feat(phase01): add first create_agent example"
```

---

### Task 2: 统一模型字符串与多模型切换（Day 2-3）

**学习目标：** 掌握 v1.x 统一模型字符串格式，理解如何在提供商之间切换

**Files:**
- Create: `phase01/02_model_strings.py`

- [ ] **Step 1: 编写多模型切换脚本**

```python
"""
Phase01 - Task 2: 统一模型字符串
学习目标：掌握 v1.x 模型字符串格式，在多个提供商间切换

运行：uv run python phase01/02_model_strings.py
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

MODELS = {
    "openai": "openai:gpt-4o-mini",
    "claude": "claude-sonnet-4-6",
    "gemini": "google_genai:gemini-2.5-flash-lite",
}


def get_current_time(city: str) -> str:
    """获取指定城市的当前时间"""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    timezones = {
        "北京": "Asia/Shanghai",
        "纽约": "America/New_York",
        "伦敦": "Europe/London",
    }
    tz_name = timezones.get(city, "UTC")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    return f"{city} 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"


def main():
    print("=== 统一模型字符串演示 ===")
    print(f"可用模型：{list(MODELS.keys())}")
    print("输入 'quit' 退出\n")

    while True:
        model_choice = input("选择模型 (openai/claude/gemini): ").strip().lower()
        if model_choice == "quit":
            break
        if model_choice not in MODELS:
            print(f"无效模型，请选择：{list(MODELS.keys())}")
            continue

        model_string = MODELS[model_choice]
        print(f"\n使用模型：{model_string}")
        print("创建 Agent 中...")

        try:
            agent = create_agent(
                model=model_string,
                tools=[get_current_time],
                system_prompt="你是一个时间查询助手。用中文回答。",
            )

            user_input = input("你: ").strip()
            if not user_input:
                continue

            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )

            ai_msg = result["messages"][-1]
            if hasattr(ai_msg, "content_blocks"):
                for block in ai_msg.content_blocks:
                    if hasattr(block, "text"):
                        print(f"AI: {block.text}")
            elif hasattr(ai_msg, "content"):
                print(f"AI: {ai_msg.content}")

        except Exception as e:
            print(f"错误：{e}")
            print("可能的原因：API Key 未配置或模型不可用")
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 逐个模型测试**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python phase01/02_model_strings.py
```

分别测试 openai、claude、gemini，确认：
- 统一模型字符串格式生效
- 模型切换无需改代码，只改字符串
- API Key 正确时能正常调用

- [ ] **Step 3: 体会观察**

记录以下要点：
- `"openai:gpt-4o-mini"` vs 旧版 `ChatOpenAI(model="gpt-4o-mini")` 的区别
- 模型字符串中 `:` 左右的含义（provider:model）
- 哪个提供商响应最快？哪个工具调用最稳定？

- [ ] **Step 4: Commit**

```bash
git add phase01/02_model_strings.py
git commit -m "feat(phase01): add unified model string demo"
```

---

### Task 3: LCEL 基础（Day 4-5）

**学习目标：** 掌握 LCEL 管道语法、Runnable 协议、链式组合

**Files:**
- Create: `phase01/03_lcel_basics.py`

- [ ] **Step 1: 编写 LCEL 练习脚本**

```python
"""
Phase01 - Task 3: LCEL 基础
学习目标：掌握 LangChain Expression Language 管道语法

运行：uv run python phase01/03_lcel_basics.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def demo_basic_chain():
    """基础链：prompt | model | parser"""
    print("=== 1. 基础链：prompt | model | parser ===\n")

    prompt = ChatPromptTemplate.from_template("用一句话解释{topic}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"topic": "LangChain"})
    print(f"结果：{result}\n")


def demo_streaming():
    """流式输出"""
    print("=== 2. 流式输出 ===\n")

    prompt = ChatPromptTemplate.from_template("写一首关于{topic}的短诗")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    print("逐字输出：", end="")
    for chunk in chain.stream({"topic": "Python"}):
        print(chunk, end="", flush=True)
    print("\n")


def demo_batch():
    """批量调用"""
    print("=== 3. 批量调用 ===\n")

    prompt = ChatPromptTemplate.from_template("用三个词形容{language}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    results = chain.batch(
        [
            {"language": "Python"},
            {"language": "Rust"},
            {"language": "JavaScript"},
        ]
    )

    for lang, result in zip(["Python", "Rust", "JavaScript"], results):
        print(f"{lang}: {result}")
    print()


def demo_parallel():
    """并行链（RunnableParallel）"""
    print("=== 4. 并行链 ===\n")

    from langchain_core.runnables import RunnableParallel

    prompt = ChatPromptTemplate.from_template("用简短的话总结：{text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    summary_chain = prompt | model | parser

    joke_prompt = ChatPromptTemplate.from_template("根据这个主题写一个笑话：{topic}")
    joke_chain = joke_prompt | model | parser

    parallel_chain = RunnableParallel(
        summary=summary_chain,
        joke=joke_chain,
    )

    result = parallel_chain.invoke(
        {"text": "LangChain 是一个 LLM 应用开发框架", "topic": "程序员"}
    )
    print(f"摘要：{result['summary']}")
    print(f"笑话：{result['joke']}")
    print()


def demo_fallback():
    """带降级的链（RunnableWithFallbacks）"""
    print("=== 5. 带降级的链 ===\n")

    prompt = ChatPromptTemplate.from_template("翻译成英文：{text}")
    primary_model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_retries=2,
    )

    chain = prompt | primary_model | StrOutputParser()

    try:
        result = chain.invoke({"text": "你好世界"})
        print(f"翻译：{result}")
    except Exception as e:
        print(f"主模型失败：{e}")
    print()


def main():
    print("=== LCEL 基础演示 ===\n")
    demo_basic_chain()
    demo_streaming()
    demo_batch()
    demo_parallel()
    demo_fallback()
    print("=== LCEL 核心概念总结 ===")
    print("| 操作 | 语法 |")
    print("|------|------|")
    print("| 管道组合 | chain = prompt \\| model \\| parser |")
    print("| 调用 | chain.invoke({...}) |")
    print("| 流式 | chain.stream({...}) |")
    print("| 批量 | chain.batch([{...}, {...}]) |")
    print("| 并行 | RunnableParallel({...}) |")
    print("| 异步 | await chain.ainvoke({...}) |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行并逐个验证**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python phase01/03_lcel_basics.py
```

Expected: 5 个演示全部运行成功

- [ ] **Step 3: 动手修改实验**

尝试以下修改并记录结果：
1. 修改 prompt 模板，添加 system message
2. 尝试 `chain.ainvoke()` 异步调用
3. 尝试用 `|` 连接更多组件（如 `RunnableLambda`）

- [ ] **Step 4: Commit**

```bash
git add phase01/03_lcel_basics.py
git commit -m "feat(phase01): add LCEL basics demo"
```

---

### Task 4: 结构化输出（Day 5-6）

**学习目标：** 掌握 `with_structured_output()` + Pydantic v2 模式定义

**Files:**
- Create: `phase01/04_structured_output.py`

- [ ] **Step 1: 编写结构化输出脚本**

```python
"""
Phase01 - Task 4: 结构化输出
学习目标：掌握 with_structured_output() + Pydantic v2

运行：uv run python phase01/04_structured_output.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


class MovieReview(BaseModel):
    """电影评价结构化输出"""

    movie_name: str = Field(description="电影名称")
    rating: float = Field(description="评分 0-10")
    summary: str = Field(description="一句话总结")
    recommend: bool = Field(description="是否推荐")
    genres: list[str] = Field(description="类型标签列表")


class PersonInfo(BaseModel):
    """人物信息结构化输出"""

    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")
    skills: list[str] = Field(description="技能列表")
    bio: str = Field(description="简短介绍")


def demo_structured_output():
    """方式一：model.with_structured_output()"""
    print("=== 1. with_structured_output() ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(MovieReview)

    result = structured_llm.invoke("请评价电影《盗梦空间》")
    print(f"电影：{result.movie_name}")
    print(f"评分：{result.rating}")
    print(f"总结：{result.summary}")
    print(f"推荐：{result.recommend}")
    print(f"类型：{', '.join(result.genres)}")
    print(f"\n类型：{type(result)}")
    print()


def demo_structured_with_prompt():
    """方式二：prompt + chain + structured output"""
    print("=== 2. Prompt Chain + 结构化输出 ===\n")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个专业的猎头，根据描述提取人物信息。"),
            ("human", "{description}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(PersonInfo)

    chain = prompt | structured_llm

    result = chain.invoke(
        {"description": "张三，28岁，全栈工程师，精通 Python、Go、Kubernetes，喜欢开源"}
    )
    print(f"姓名：{result.name}")
    print(f"年龄：{result.age}")
    print(f"职业：{result.occupation}")
    print(f"技能：{', '.join(result.skills)}")
    print(f"介绍：{result.bio}")
    print()


def demo_json_mode():
    """方式三：JSON 模式（不使用 schema 约束）"""
    print("=== 3. JSON 模式 ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
        response_format={"type": "json_object"}
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "总是以 JSON 格式回复。"),
            ("human", "列出三种编程语言及其创建者"),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({})
    print(result.content)
    print()


def demo_enum_field():
    """方式四：带枚举约束的结构化输出"""
    print("=== 4. 带枚举约束 ===\n")

    from enum import Enum

    class Sentiment(str, Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

    class SentimentAnalysis(BaseModel):
        sentiment: Sentiment = Field(description="情感倾向")
        confidence: float = Field(description="置信度 0-1")
        explanation: str = Field(description="分析理由")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(SentimentAnalysis)

    result = structured_llm.invoke("这个产品还不错，但价格有点贵")
    print(f"情感：{result.sentiment.value}")
    print(f"置信度：{result.confidence}")
    print(f"理由：{result.explanation}")
    print()


def main():
    print("=== 结构化输出演示 ===\n")
    demo_structured_output()
    demo_structured_with_prompt()
    demo_json_mode()
    demo_enum_field()

    print("=== 结构化输出总结 ===")
    print("| 方式 | 适用场景 |")
    print("|------|----------|")
    print("| with_structured_output() | 需要 Pydantic 对象，最推荐 |")
    print("| prompt + chain + structured | 需要自定义 prompt |")
    print("| JSON 模式 | 简单 JSON，不需要强类型 |")
    print("| 枚举约束 | 需要限制字段值范围 |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行并验证**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python phase01/04_structured_output.py
```

Expected: 4 个演示全部成功，输出包含正确的 Pydantic 对象

- [ ] **Step 3: 实验要点**

记录以下观察：
- `with_structured_output()` 底层使用的是 Function Calling 还是 JSON Mode？
- 如果 LLM 输出不符合 schema 会发生什么？
- 枚举约束是否真的限制了输出？

- [ ] **Step 4: Commit**

```bash
git add phase01/04_structured_output.py
git commit -m "feat(phase01): add structured output demo"
```

---

### Task 5: 流式输出（Day 6-7）

**学习目标：** 掌握 `astream_events` v2、token 级流式、自定义流式事件

**Files:**
- Create: `phase01/05_streaming.py`

- [ ] **Step 1: 编写流式输出脚本**

```python
"""
Phase01 - Task 5: 流式输出
学习目标：掌握 astream_events v2、token 级流式输出

运行：uv run python phase01/05_streaming.py
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


async def demo_basic_streaming():
    """基础流式输出"""
    print("=== 1. 基础流式输出 (stream) ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    prompt = ChatPromptTemplate.from_template("详细解释{topic}，至少200字")
    chain = prompt | llm | StrOutputParser()

    print("结果：", end="")
    async for chunk in chain.astream({"topic": "为什么 Python 这么流行"}):
        print(chunk, end="", flush=True)
    print("\n")


async def demo_astream_events():
    """astream_events v2：事件级流式"""
    print("=== 2. astream_events v2 ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    prompt = ChatPromptTemplate.from_template("写一首关于{topic}的诗")
    chain = prompt | llm | StrOutputParser()

    event_types_seen = set()

    async for event in chain.astream_events(
        {"topic": "AI"},
        version="v2",
    ):
        kind = event["event"]
        event_types_seen.add(kind)

        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            print(token, end="", flush=True)
        elif kind == "on_chain_start":
            print(f"\n[链开始] {event['name']}")
        elif kind == "on_chain_end":
            print(f"\n[链结束] {event['name']}")

    print(f"\n\n观察到的事件类型：{sorted(event_types_seen)}")


async def demo_streaming_with_agent():
    """带工具调用的 Agent 流式输出"""
    print("=== 3. Agent 流式输出 ===\n")

    from langchain.agents import create_agent

    def get_weather(city: str) -> str:
        """获取天气"""
        weathers = {
            "北京": "晴天，25°C",
            "上海": "多云，28°C",
            "深圳": "雨，30°C",
        }
        return weathers.get(city, f"{city}：暂无数据")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather],
        system_prompt="你是一个天气助手。用中文回答。",
    )

    print("问：北京天气怎么样？")
    print("答：", end="")

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
            elif hasattr(chunk, "tool_call_chunks"):
                for tc_chunk in chunk.tool_call_chunks:
                    if tc_chunk.get("name"):
                        print(f"\n[调用工具: {tc_chunk['name']}]")
        elif kind == "on_tool_start":
            print(f"\n[工具开始: {event['name']}]")
        elif kind == "on_tool_end":
            print(f"\n[工具结束: {event['name']}]")

    print("\n")


async def main():
    await demo_basic_streaming()
    await demo_astream_events()
    await demo_streaming_with_agent()

    print("=== 流式输出总结 ===")
    print("| 方法 | 用途 |")
    print("|------|------|")
    print("| .stream() | 同步流式，逐 token |")
    print("| .astream() | 异步流式 |")
    print("| .astream_events(v2) | 事件级流式，可观察链内部 |")
    print("| Agent 流式 | 可观察工具调用过程 |")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 运行并验证**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python phase01/05_streaming.py
```

Expected: 三种流式模式都能工作，特别验证 `astream_events` v2 能观察到链的内部事件

- [ ] **Step 3: 观察要点**

- `on_chat_model_stream` 事件的 `data` 结构是什么？
- Agent 流式中工具调用的事件顺序是什么？
- `astream_events` v1 和 v2 的区别？（查看文档）

- [ ] **Step 4: Commit**

```bash
git add phase01/05_streaming.py
git commit -m "feat(phase01): add streaming output demo"
```

---

### Task 6: Deep Agents 初探（Day 7-8）

**学习目标：** 了解 Deep Agents 概念，尝试内置能力

**Files:**
- Create: `phase01/06_deep_agents_intro.py`

- [ ] **Step 1: 编写 Deep Agents 探索脚本**

```python
"""
Phase01 - Task 6: Deep Agents 初探
学习目标：了解 Deep Agents 的自动上下文压缩、虚拟文件系统

注意：Deep Agents 是 v1.x 新特性，需确认已安装最新版本

运行：uv run python phase01/06_deep_agents_intro.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def explore_create_agent_capabilities():
    """探索 create_agent 的高级能力"""
    print("=== 1. create_agent 高级配置 ===\n")

    from langchain.agents import create_agent

    def search_web(query: str) -> str:
        """搜索网页（模拟）"""
        return f"搜索结果：关于'{query}'的最新信息..."

    def read_file(path: str) -> str:
        """读取文件内容（模拟）"""
        return f"文件 {path} 的内容..."

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[search_web, read_file],
        system_prompt=(
            "你是一个研究助手。你可以搜索网页和读取文件。"
            "请先分析用户需求，再选择合适的工具。"
            "用中文回答。"
        ),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我搜索一下 LangChain 的最新版本信息"}]}
    )

    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            role = getattr(msg, "type", "unknown")
            print(f"[{role}] {msg.content[:200]}")
        elif hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                print(f"[tool_call] {tc['name']}({tc['args']})")
    print()


def explore_agent_state():
    """探索 Agent 状态结构"""
    print("=== 2. Agent 返回结构分析 ===\n")

    from langchain.agents import create_agent

    def add_numbers(a: int, b: int) -> str:
        """将两个数字相加"""
        return str(a + b)

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[add_numbers],
        system_prompt="你是一个计算助手。",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "3加5等于多少？"}]}
    )

    print(f"结果类型：{type(result)}")
    print(f"结果键：{result.keys()}")
    print(f"消息数量：{len(result['messages'])}")
    print("\n消息流程：")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content = ""
        if hasattr(msg, "content") and msg.content:
            content = msg.content[:100]
        elif hasattr(msg, "tool_calls"):
            content = str(msg.tool_calls)[:100]
        elif hasattr(msg, "name"):
            content = f"tool={msg.name}, output={str(msg.content)[:100]}"
        print(f"  [{i}] {msg_type}: {content}")
    print()


def explore_langGraph_vs_create_agent():
    """对比 create_agent 和 LangGraph 直接构建"""
    print("=== 3. create_agent vs LangGraph ===\n")

    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI

    def multiply(a: int, b: int) -> str:
        """两数相乘"""
        return str(a * b)

    llm = ChatOpenAI(model="gpt-4o-mini")

    langgraph_agent = create_react_agent(
        model=llm,
        tools=[multiply],
    )

    result = langgraph_agent.invoke(
        {"messages": [{"role": "user", "content": "3乘以7等于多少？"}]}
    )

    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            msg_type = type(msg).__name__
            print(f"  [{msg_type}] {msg.content[:100]}")
    print()

    print("对比要点：")
    print("| 方式 | 优点 | 缺点 |")
    print("|------|------|------|")
    print("| create_agent | 极简、10行搞定 | 灵活性低 |")
    print("| create_react_agent | 可自定义模型对象 | 需要更多配置 |")
    print("| 自定义 StateGraph | 完全控制 | 代码量大 |")


def main():
    print("=== Deep Agents 初探 ===\n")

    try:
        explore_create_agent_capabilities()
    except Exception as e:
        print(f"create_agent 探索失败：{e}")
        print("可能需要更新 langchain 版本\n")

    try:
        explore_agent_state()
    except Exception as e:
        print(f"状态分析失败：{e}\n")

    try:
        explore_langGraph_vs_create_agent()
    except Exception as e:
        print(f"对比演示失败：{e}\n")

    print("\n=== Deep Agents 关键概念 ===")
    print("1. create_agent：最简 API，底层由 LangGraph 驱动")
    print("2. 自动上下文压缩：（Deep Agents 特性）自动管理长对话")
    print("3. 虚拟文件系统：（Deep Agents 特性）Agent 可读写虚拟文件")
    print("4. 子 Agent 生成：（Deep Agents 特性）自动创建子 Agent 处理子任务")
    print("5. create_react_agent：LangGraph 的预构建 Agent，比 create_agent 灵活")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行并验证**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python phase01/06_deep_agents_intro.py
```

Expected: 三个实验都能运行，理解 Agent 返回的消息结构

- [ ] **Step 3: 观察要点**

- `create_agent` 返回的 `result` 中 `messages` 的顺序和类型
- Human Message → AI Message (tool_calls) → Tool Message → AI Message (final) 的流程
- Deep Agents 特性在当前版本中的支持程度

- [ ] **Step 4: Commit**

```bash
git add phase01/06_deep_agents_intro.py
git commit -m "feat(phase01): add Deep Agents intro demo"
```

---

### Task 7: Checkpoint 项目 — 多模型切换聊天器（Day 8-14）

**学习目标：** 整合第一阶段所有知识，完成多模型切换聊天器

**Files:**
- Create: `phase01/checkpoint/chat_multi_model.py`
- Create: `phase01/checkpoint/README.md`

- [ ] **Step 1: 编写 Checkpoint 项目 README**

```markdown
# Phase 01 Checkpoint: 多模型切换聊天器

## 功能
- 支持 OpenAI / Claude / Gemini 模型热切换
- 流式输出
- 结构化信息提取
- 工具调用（时间查询、计算器）
- LangSmith 追踪

## 使用
```bash
# 确保已配置 .env 中的 API Key
uv run python phase01/checkpoint/chat_multi_model.py
```

## 命令
- `/model openai|claude|gemini` — 切换模型
- `/stream on|off` — 开关流式输出
- `/extract` — 进入结构化提取模式
- `/history` — 查看对话历史
- `/clear` — 清空对话
- `/quit` — 退出

## 验收标准
- [x] 能在三个模型之间切换
- [x] 流式输出正常
- [x] 结构化提取模式可用
- [x] 工具调用正常
```

- [ ] **Step 2: 编写多模型切换聊天器**

```python
"""
Phase01 Checkpoint: 多模型切换聊天器
整合：统一模型字符串、LCEL、结构化输出、流式输出、工具调用

运行：uv run python phase01/checkpoint/chat_multi_model.py
"""

import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo
from datetime import datetime

load_dotenv()

MODELS = {
    "openai": "openai:gpt-4o-mini",
    "claude": "claude-sonnet-4-6",
    "gemini": "google_genai:gemini-2.5-flash-lite",
}

DISPLAY_NAMES = {
    "openai": "GPT-4o-mini",
    "claude": "Claude Sonnet 4.6",
    "gemini": "Gemini 2.5 Flash Lite",
}


def get_current_time(city: str) -> str:
    """获取指定城市当前时间"""
    timezones = {
        "北京": "Asia/Shanghai",
        "上海": "Asia/Shanghai",
        "东京": "Asia/Tokyo",
        "纽约": "America/New_York",
        "伦敦": "Europe/London",
        "旧金山": "America/Los_Angeles",
        "巴黎": "Europe/Paris",
        "悉尼": "Australia/Sydney",
    }
    tz_name = timezones.get(city, "UTC")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    return f"{city} 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})"


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误：包含不允许的字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


class StructuredInfo(BaseModel):
    topic: str = Field(description="讨论的主题")
    key_points: list[str] = Field(description="关键要点列表")
    sentiment: str = Field(description="情感倾向：positive/negative/neutral")
    summary: str = Field(description="一句话总结")


class MultiModelChat:
    def __init__(self):
        self.current_model = "openai"
        self.streaming = True
        self.messages = []
        self.agent = None
        self._create_agent()

    def _create_agent(self):
        model_string = MODELS[self.current_model]
        try:
            self.agent = create_agent(
                model=model_string,
                tools=[get_current_time, calculate],
                system_prompt=(
                    "你是一个智能助手，可以查询时间和做计算。"
                    "用中文回答。切换模型时告知用户。"
                ),
            )
        except Exception as e:
            print(f"创建 Agent 失败（模型：{model_string}）：{e}")
            print("请检查 API Key 配置")

    def switch_model(self, model_key: str):
        if model_key not in MODELS:
            print(f"无效模型，可选：{list(MODELS.keys())}")
            return
        self.current_model = model_key
        self._create_agent()
        print(f"已切换到 {DISPLAY_NAMES[model_key]}")

    def extract_structured(self, text: str):
        llm_map = {
            "openai": ChatOpenAI(model="gpt-4o-mini", temperature=0),
        }
        llm = llm_map.get(self.current_model)
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        structured_llm = llm.with_structured_output(StructuredInfo)
        result = structured_llm.invoke(text)
        return result

    def chat(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        if self.streaming and self.agent:
            print(f"\n{DISPLAY_NAMES[self.current_model]}：", end="")
            full_response = ""
            try:
                for event in self.agent.stream({"messages": self.messages}):
                    if hasattr(event, "content") and event.content:
                        print(event.content, end="", flush=True)
                        full_response += event.content
                print()
            except Exception:
                try:
                    result = self.agent.invoke({"messages": self.messages})
                    for msg in result["messages"][len(self.messages):]:
                        if hasattr(msg, "content") and msg.content:
                            full_response += msg.content
                            print(msg.content)
                except Exception as e:
                    print(f"\n错误：{e}")
                    return
            if full_response:
                self.messages.append({"role": "assistant", "content": full_response})
        elif self.agent:
            try:
                result = self.agent.invoke({"messages": self.messages})
                ai_msg = result["messages"][-1]
                if hasattr(ai_msg, "content"):
                    print(f"\n{DISPLAY_NAMES[self.current_model]}：{ai_msg.content}")
                    self.messages.append({"role": "assistant", "content": ai_msg.content})
            except Exception as e:
                print(f"错误：{e}")

    def show_history(self):
        if not self.messages:
            print("暂无对话历史")
            return
        for msg in self.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            print(f"  [{role}] {content}")

    def run(self):
        print("=" * 50)
        print("  多模型切换聊天器 — Phase 01 Checkpoint")
        print("=" * 50)
        print(f"当前模型：{DISPLAY_NAMES[self.current_model]}")
        print(f"流式输出：{'开启' if self.streaming else '关闭'}")
        print()
        print("命令：")
        print("  /model openai|claude|gemini  切换模型")
        print("  /stream on|off               开关流式输出")
        print("  /extract <text>              结构化提取")
        print("  /history                     查看历史")
        print("  /clear                       清空对话")
        print("  /quit                        退出")
        print()

        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue

            if user_input == "/quit":
                print("再见！")
                break
            elif user_input.startswith("/model"):
                parts = user_input.split()
                if len(parts) == 2:
                    self.switch_model(parts[1])
                else:
                    print(f"可用模型：{list(MODELS.keys())}")
            elif user_input.startswith("/stream"):
                parts = user_input.split()
                if len(parts) == 2:
                    self.streaming = parts[1].lower() == "on"
                    print(f"流式输出：{'开启' if self.streaming else '关闭'}")
            elif user_input.startswith("/extract"):
                text = user_input[len("/extract"):].strip()
                if text:
                    print("\n结构化提取结果：")
                    result = self.extract_structured(text)
                    print(f"  主题：{result.topic}")
                    print(f"  要点：{', '.join(result.key_points)}")
                    print(f"  情感：{result.sentiment}")
                    print(f"  总结：{result.summary}")
                else:
                    print("用法：/extract <文本>")
            elif user_input == "/history":
                self.show_history()
            elif user_input == "/clear":
                self.messages = []
                print("对话已清空")
            else:
                self.chat(user_input)


if __name__ == "__main__":
    chat = MultiModelChat()
    chat.run()
```

- [ ] **Step 3: 运行并逐一验证功能**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python phase01/checkpoint/chat_multi_model.py
```

验证清单：
- 输入 "北京现在几点？" → 工具调用正常
- 输入 "123*456 等于多少？" → 计算工具正常
- `/model claude` → 切换到 Claude 模型
- `/model gemini` → 切换到 Gemini
- `/stream off` → 关闭流式，再次聊天
- `/stream on` → 开启流式
- `/extract LangChain是一个LLM应用框架，它提供了模块化的工具` → 结构化提取
- `/history` → 查看历史
- `/clear` → 清空
- `/quit` → 退出

- [ ] **Step 4: Commit**

```bash
git add phase01/checkpoint/
git commit -m "feat(phase01): add multi-model chat checkpoint project"
```

---

### Task 8: LangSmith 追踪与调试（Day 13-14）

**学习目标：** 配置 LangSmith，理解追踪和调试流程

**Files:**
- Create: `phase01/07_langsmith_setup.py`

- [ ] **Step 1: 编写 LangSmith 追踪演示脚本**

```python
"""
Phase01 - Task 8: LangSmith 追踪与调试
学习目标：配置 LangSmith，理解追踪流程

前置条件：
1. 在 .env 中设置 LANGSMITH_API_KEY
2. 设置 LANGSMITH_TRACING=true
3. 设置 LANGSMITH_PROJECT=lzy-langgraph-phase01

运行：uv run python phase01/07_langsmith_setup.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent

load_dotenv()


def demo_basic_tracing():
    """基础追踪：查看在 LangSmith 上的记录"""
    print("=== 1. 基础追踪 ===\n")

    if not os.environ.get("LANGSMITH_API_KEY"):
        print("请先在 .env 中设置 LANGSMITH_API_KEY")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("用3个词形容{concept}")
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"concept": "LangChain"})
    print(f"结果：{result}")
    print(f"请在 LangSmith 中查看项目：lzy-langgraph-phase01")
    print(f"地址：https://smith.langchain.com")
    print()


def demo_agent_tracing():
    """Agent 追踪：观察工具调用链"""
    print("=== 2. Agent 追踪 ===\n")

    def search_info(query: str) -> str:
        """搜索信息（模拟）"""
        return f"关于'{query}'的信息：这是一个模拟搜索结果"

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[search_info],
        system_prompt="你是一个信息搜索助手。",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我搜索一下 LangGraph"}]}
    )

    ai_msg = result["messages"][-1]
    if hasattr(ai_msg, "content_blocks"):
        for block in ai_msg.content_blocks:
            if hasattr(block, "text"):
                print(f"AI: {block.text}")
    elif hasattr(ai_msg, "content"):
        print(f"AI: {ai_msg.content}")

    print("\n请在 LangSmith 中查看追踪详情：")
    print("  - 可以看到完整的 LLM 调用")
    print("  - 可以看到工具调用的输入和输出")
    print("  - 可以看到 Token 使用量和延迟")
    print()


def demo_add_metadata():
    """添加元数据和标签"""
    print("=== 3. 元数据和标签 ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("翻译成英文：{text}")
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {"text": "今天天气很好"},
        config={
            "run_name": "翻译任务",
            "tags": ["翻译", "中文-英文"],
            "metadata": {"environment": "learning", "phase": "01"},
        },
    )
    print(f"结果：{result}")
    print("此追踪已标记为'翻译任务'，带有 tags 和 metadata")
    print("在 LangSmith 中可按标签过滤")
    print()


def main():
    print("=== LangSmith 追踪与调试 ===\n")
    print("当前配置：")
    print(f"  LANGSMITH_TRACING: {os.environ.get('LANGSMITH_TRACING', '未设置')}")
    print(f"  LANGSMITH_PROJECT: {os.environ.get('LANGSMITH_PROJECT', '未设置')}")
    print(f"  LANGSMITH_API_KEY: {'已设置' if os.environ.get('LANGSMITH_API_KEY') else '未设置'}")
    print()

    demo_basic_tracing()
    demo_agent_tracing()
    demo_add_metadata()

    print("=== LangSmith 使用要点 ===")
    print("1. 设置环境变量即可自动追踪，零代码侵入")
    print("2. 追踪中可以看到：输入、输出、延迟、Token 数")
    print("3. Agent 追踪可以看到完整的工具调用链")
    print("4. 可以添加 run_name / tags / metadata 便于过滤")
    print("5. LangSmith Engine 可自动检测问题并建议修复")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行并验证**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python phase01/07_langsmith_setup.py
```

Expected: 运行后能在 https://smith.langchain.com 看到追踪记录

- [ ] **Step 3: 在 LangSmith 中验证**

登录 LangSmith 后确认：
- 能看到 `lzy-langgraph-phase01` 项目
- 能看到每次 LLM 调用的详情
- 能看到工具调用链
- 能看到元数据和标签

- [ ] **Step 4: Commit**

```bash
git add phase01/07_langsmith_setup.py
git commit -m "feat(phase01): add LangSmith tracing demo"
```

---

### Task 9: 第一阶段总结与 Checkpoint 验收（Day 14）

**学习目标：** 回顾第一阶段知识，确保 Checkpoint 项目完善

- [ ] **Step 1: 运行 Checkpoint 项目完整功能测试**

Run:
```bash
cd /Users/tpy/workspace/agent/lzy-langgraph
uv run python phase01/checkpoint/chat_multi_model.py
```

逐一验证：
- [ ] 三个模型切换正常
- [ ] 流式输出正常
- [ ] 工具调用正常（时间、计算）
- [ ] 结构化提取正常
- [ ] LangSmith 追踪可见

- [ ] **Step 2: 创建第一阶段学习笔记**

创建 `phase01/NOTES.md`：

```markdown
# Phase 01 学习笔记

## 核心概念
- `create_agent`：v1.x 一步创建 Agent
- 统一模型字符串：`"provider:model"` 格式
- LCEL：`|` 管道语法组合链
- `with_structured_output()`：Pydantic v2 结构化输出
- `astream_events` v2：事件级流式
- LangSmith：零侵入追踪

## 关键发现
（在此记录自己的学习发现）

## 遇到的问题
（在此记录遇到的问题和解决方案）

## 下一步
- Phase 02: RAG 深入
```

- [ ] **Step 3: 最终 Commit**

```bash
git add phase01/NOTES.md
git commit -m "docs(phase01): add learning notes"
```