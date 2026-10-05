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

- **本地 RAG 检索**：Chroma + BAAI/bge-small-zh-v1.5，解决知识库问答
- **八字排盘工具**：引入 lunar_python，解决大模型推算干支出错的问题
- **工程兜底方案**：正则预判 + 代码强制调用，解决 GLM-4-Flash 工具调用失效
- **多会话隔离**：状态与逻辑分离，支持 ChatGPT 式会话切换
- **流式输出**：打字机效果，已过滤工具调用碎片

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

## ⚠️ 已知局限
- 命理知识库粒度较粗，小模型基于理论推理时易出现幻觉，后续可通过精细化知识库或换用更大的模型优化。
- 暂未做持久化存储，重启服务会丢失会话历史。