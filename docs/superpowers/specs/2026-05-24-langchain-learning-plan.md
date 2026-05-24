# LangChain v1.x + LangGraph 学习计划

> 背景：完全零基础，目标从零到独立开发 AI Agent 应用，偏好实战驱动，每天 1-2 小时
> 重点关注：LangChain v1.0–v1.2 新特性、RAG 深入、Tool Calling 深入、工程化实践
> 总时长：约 14 周

---

## 第一阶段：v1.x 新架构与核心 API（第 1-2 周）

### 目标
理解 LangChain v1.x 架构变化，掌握新核心 API，能跑通第一个 Agent

### 知识点
- **v1.x 架构总览**：LangChain / LangGraph / Deep Agents / LangSmith 四层关系
- **统一模型接口 v1.x**：`"openai:gpt-5.4"` / `"claude-sonnet-4-6"` 等字符串格式指定模型
- **`create_agent` 新 API**：10 行代码创建 Agent（取代旧版 AgentExecutor）
- **LCEL (LangChain Expression Language)**：`|` 管道语法、`Runnable` 协议
- **结构化输出 v2**：`with_structured_output()` + Pydantic v2
- **流式输出**：`astream_events` v2 / token 级流式
- **环境搭建**：Python 3.10+ / UV 包管理 / API Key / LangSmith 追踪配置

### v1.x 新特性重点
| 特性 | 说明 |
|------|------|
| `create_agent` | 一行创建 Agent，无需手动配置 AgentExecutor |
| 统一模型字符串 | `"openai:gpt-5.4"` 格式，统一所有提供商 |
| Deep Agents | 开箱即用 Agent：自动上下文压缩、虚拟文件系统、子 Agent 生成 |

### 实战项目
1. **Day 1-3**：环境搭建 + `create_agent` 第一个 Agent（天气查询）
2. **Day 4-7**：用 LCEL 构建流式问答链 + 结构化输出提取器
3. **Day 8-14**：对比 `create_agent` vs 手动 LangGraph 构建，理解底层机制

### Checkpoint 项目
> 多模型切换聊天器：用统一模型字符串在 OpenAI / Claude / Gemini 之间切换，支持流式输出

---

## 第二阶段：RAG 深入（第 3-5 周）

### 目标
从基础 RAG 到生产级高级检索系统，掌握 v1.x RAG 最佳实践

### 知识点
- **文档加载与分块**：新版 `Document` 对象、递归分块、语义分块
- **Embedding 模型**：OpenAI / 开源模型对比（`text-embedding-3-small`、`bge-m3` 等）
- **向量数据库实战对比**：Chroma / FAISS / Qdrant / Pinecone
- **基础 RAG 流水线**：`create_retrieval_chain` + `create_history_aware_retriever`
- **高级检索策略**：
  - 多查询检索（Multi-query）
  - 自查询（Self-query）
  - 上下文压缩（ContextualCompression）
  - 重排序（Re-ranking / Cohere / FlashRank）
  - 父文档检索（ParentDocumentRetriever）
  - 多向量检索（MultiVectorRetriever）
- **RAG 评估**：LangSmith Evaluation + RAGAS 指标

### v1.x 新特性重点
| 特性 | 说明 |
|------|------|
| `create_retrieval_chain` | v1.x 新的 RAG 链构建方式 |
| `create_history_aware_retriever` | 内置对话历史感知检索 |
| LangSmith Engine | 自动检测追踪问题并提议修复 |

### 实战项目
1. **Day 15-18**：PDF 对话应用（单个向量库入门）
2. **Day 19-23**：向量数据库对比实验（同一数据集，4 种向量库效果与性能对比）
3. **Day 24-35**：多策略检索对比 + 综合项目：多文档混合检索 + 重排序研究助手

### Checkpoint 项目
> 智能论文研读助手：加载多篇论文 PDF，支持混合检索 + 重排序，用 LangSmith 评估检索质量

---

## 第三阶段：Tool Calling 深入与 ReAct Agent（第 6-8 周）

### 目标
精通 v1.x Tool Calling 机制，构建可控 Agent

### 知识点
- **v1.x Tool Calling 新机制**：
  - `@tool` 装饰器 v2
  - `InjectedToolArg` 注入参数
  - 工具消息规范（ToolMessage）
  - 并行 Tool Calling
- **自定义工具开发**：Pydantic v2 参数验证、错误处理、异步工具
- **多模型 Tool Calling 差异**：OpenAI / Claude / Gemini 的行为差异
- **LangGraph ReAct Agent** vs 旧版 `AgentExecutor`
  - `create_react_agent` 高级 API
  - 自定义状态机精确控制
- **Human-in-the-loop**：审批、编辑、中断恢复
- **安全考量**：工具权限控制、输入清洗、沙箱执行

### v1.x 新特性重点
| 特性 | 说明 |
|------|------|
| `create_agent` + tools | 一行创建带工具的 Agent |
| Deep Agents 工具系统 | 内置虚拟文件系统、Shell 执行等 |
| MCP 工具集成 | 通过 Model Context Protocol 连接外部工具 |
| A2A 协议 | Agent-to-Agent 通信协议 |

### 实战项目
1. **Day 36-40**：多工具助手（搜索 + 数据库 + 代码执行）
2. **Day 41-49**：带 Human-in-the-loop 的审批流 Agent
3. **Day 50-56**：MCP 工具集成实践 + 安全加固

### Checkpoint 项目
> 智能代码审查 Agent：能自动搜索代码、运行测试、提出修改建议，人工审批后执行

---

## 第四阶段：LangGraph 高级特性（第 9-11 周）

### 目标
掌握子图、流式、持久化与多 Agent 协作

### 知识点
- **LangGraph 核心概念重温**：State / Node / Edge / Command
- **子图（Subgraph）与状态组合**：
  - 父子图状态映射
  - 多子图编排
- **`Command` 对象与动态路由**：条件分支、循环
- **流式输出进阶**：
  - token 级流式
  - 事件级流式（`astream_events`）
  - 自定义流式事件
- **持久化与检查点**：
  - PostgreSQL / SQLite Checkpoint
  - 会话恢复与时间旅行
- **多 Agent 协作模式**：
  - 主管-工人（Supervisor-Worker）
  - 层级式（Hierarchical）
  - 对等式（Peer-to-Peer）
- **LangGraph Platform 部署**：
  - `langgraph dev` 本地开发
  - `langgraph up` 部署到云
  - Agent Server API

### v1.x 新特性重点
| 特性 | 说明 |
|------|------|
| Deep Agents 子 Agent | Agent 可自动生成子 Agent 处理子任务 |
| 自动上下文压缩 | Deep Agent 自动管理上下文窗口 |
| Agent Server | 全新的 Agent 部署和运行方式 |
| Fleet | 无代码 Agent 平台（渠道、定时任务、Webhook） |

### 实战项目
1. **Day 57-62**：子图组合的多功能 Agent（研究 + 写作 + 审核三个子图）
2. **Day 63-72**：多 Agent 协作系统（代码审查团队：架构师 + 开发者 + 测试员）
3. **Day 73-77**：部署到 LangGraph Platform / Agent Server

### Checkpoint 项目
> 多 Agent 研究团队：多个专业 Agent 协作完成复杂研究任务，部署为可调用的 API

---

## 第五阶段：工程化与毕业项目（第 12-14 周）

### 目标
具备生产级开发、部署和维护 AI Agent 应用的能力

### 知识点
- **错误处理与重试**：`with_retry()`、断路器、超时控制、优雅降级
- **成本优化**：
  - Token 计算与监控
  - 语义缓存（Semantic Cache）
  - 模型路由（小问题用小模型、复杂问题用大模型）
- **LangSmith 深度使用**：
  - 追踪与调试
  - Evaluation：自动评估、人工评估
  - 标注与反馈
  - A/B 测试
  - LangSmith Engine 自动问题检测
  - LangSmith MCP Server
- **测试策略**：
  - Agent 单元测试
  - 回归测试
  - CI/CD 集成
- **生产监控与告警**：
  - LangSmith Insights
  - 成本追踪
  - 性能指标
- **安全**：
  - 提示注入防护
  - 数据脱敏
  - LLM Gateway 与 PII/Secrets 红线

### 实战项目
- **综合毕业项目**：从零设计并开发一个完整的 AI Agent 应用
  - 需求分析 → 架构设计 → 开发 → 测试 → 部署 → 监控 全流程
  - 自主选题（建议：智能研究助手 / 自动代码审查员 / 多模态内容生成器）

### 毕业项目要求
- [ ] 使用 `create_agent` 或 LangGraph 自定义状态机
- [ ] 包含 RAG 检索（至少一种高级检索策略）
- [ ] 包含 3+ 个自定义工具 + MCP 集成
- [ ] 包含 Human-in-the-loop 审批流
- [ ] 部署到 LangGraph Platform / Agent Server
- [ ] LangSmith 追踪 + 评估
- [ ] 包含错误处理、重试、降级策略

---

## 学习路线图

```
Week 1-2:  v1.x 新架构与核心 API ──── create_agent + LCEL + 流式
                                      │
Week 3-5:  RAG 深入 ───────────────── 文档分块 → 向量库 → 高级检索 → 评估
                                      │
Week 6-8:  Tool Calling + Agent ────── 工具定义 → ReAct → HITL → 安全
                                      │
Week 9-11: LangGraph 高级 ────────── 子图 → 流式 → 持久化 → 多Agent → 部署
                                      │
Week 12-14: 工程化 + 毕业项目 ──────── 重试/缓存/监控/测试 → 完整项目
```

## 每日学习节奏建议

| 时间 | 活动 |
|------|------|
| 0-20 min | 阅读文档 / 官方教程 |
| 20-80 min | 编码实践 |
| 80-120 min | 总结笔记 + 代码整理 |

## 推荐资源

- **官方文档**：https://python.langchain.com/docs/
- **LangGraph 文档**：https://langchain-ai.github.io/langgraph/
- **LangSmith 教程**：https://docs.langchain.com/langsmith/
- **LangChain GitHub**：https://github.com/langchain-ai/langchain
- **LangGraph GitHub**：https://github.com/langchain-ai/langgraph
- **Deep Agents 文档**：https://python.langchain.com/docs/deepagents/