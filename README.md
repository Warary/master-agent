## 项目简介

基于 LangChain 1.x 开发的算命大师 AI Agent，扮演「陈玉楼」角色与用户对话。
使用智谱 GLM-4-Flash 作为大模型，本地 RAG 知识库检索命理知识，
支持流式输出、工具调用、短期记忆管理。

## 技术栈

- **Agent 框架**：LangChain 1.x / LangGraph
- **大模型**：智谱 GLM-4-Flash（OpenAI 兼容接口）
- **RAG**：Chroma + BAAI/bge-small-zh-v1.5（本地 Embedding）
- **前端**：Streamlit
- **语言**：Python 3.13

## 核心功能

- **Agent 编排**：基于 create_agent 构建 ReAct 循环
- **RAG 检索**：本地向量库检索命理知识，避免模型幻觉
- **流式输出**：终端与网页双端支持打字机效果
- **短期记忆**：滑动窗口管理上下文，防止 Token 爆炸
- **工具调用**：模型自主决定何时调用 search_knowledge

## 项目结构
Code/
├── Demo2.py # 核心业务类 Master
├── Demo2_app.py # Streamlit 网页界面
├── build_rag.py # 向量库构建脚本
├── test_rag.py # 检索测试脚本
├── knowledge/ # 知识库文档
│ └── 命理知识.txt
├── chroma_db/ # 向量库（本地持久化）
└── .env # API Key 配置

## 踩坑记录

### 1. 智谱结构化输出不可用
- **现象**：`glm-4-flash` 用 `with_structured_output` 频繁返回 Markdown 包裹的 JSON，Pydantic 解析失败
- **原因**：轻量模型对结构化输出指令遵循能力弱
- **解法**：放弃结构化输出，改为在 SYSTEM_PROMPT 中用标记 + 正则解析

### 2. RAG 检索结果与问题不匹配
- **现象**：问「属马运势」，检索出的是「三合局」「五行相生」等理论
- **原因**：知识库只有理论没有具体答案
- **解法**：往知识库补充具体的生肖运势段落，让检索直接命中答案

### 3. 流式输出工具碎片污染
- **现象**：打字机输出时混入 JSON 参数碎片，界面混乱
- **原因**：Agent 流式输出中工具调用参数也是逐字吐出的
- **解法**：用 `getattr(chunk, "tool_call_chunks", None)` 过滤

