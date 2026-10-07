# 🔮 陈大师算命 Agent（Master Agent）

> 基于 LangChain + 智谱 GLM-4-Flash + FastAPI + Streamlit 构建的智能算命 Agent。
> 采用 **前后端分离** 与 **无状态（Stateless）后端** 架构，支持多用户并发隔离与流式打字机效果。

## ✨ 核心特性

- **无状态后端架构**：后端不保存任何用户状态。前端通过 HTTP 请求携带 `history` 和 `last_bazi`，彻底解决多用户并发时的数据串号问题，支持水平扩展。
- **工作流编排（Workflow Orchestration）**：代码强制接管排盘与 RAG 检索逻辑（基于正则与关键词兜底），大模型只负责将组装好的素材转化为自然语言，有效缓解了小模型的幻觉与工具调用失效问题。
- **SSE 流式输出**：基于 FastAPI `StreamingResponse` 实现打字机效果。手动处理了 UTF-8 中文在分块传输边界被截断导致的 `ChunkedEncodingError`。
- **系统隐式传参**：后端通过 `[SYSTEM_DATA:last_bazi:...]` 隐藏标记，在流式响应的最前端向前端传递排盘状态，前端拦截并保存，不展示给用户。
- **RAG 幻觉治理**：剥离检索结果中的 Markdown 格式，并利用元提示（Meta-Prompt）下"铁律"，约束小模型复述原文的倾向。
- **多会话隔离**：前端基于 Streamlit `session_state` 实现 ChatGPT 式侧边栏多会话切换，每个会话独立维护 `memory` 与 `last_bazi`。

## 🏗️ 架构设计

```text
┌──────────────────── 前端 app.py (Streamlit) ────────────────────┐
│                                                                 │
│   session_state.sessions = {                                    │
│       "会话A": {"memory": [...], "last_bazi": "庚辰..."},        │
│       "会话B": {"memory": [...], "last_bazi": None},             │
│   }                                                             │
│                                                                 │
│   用户输入  →  带上 memory + last_bazi  →  POST /chat            │
│                                                                 │
│   接收流：                                                       │
│     拦截 [SYSTEM_DATA]  →  存进 last_bazi                        │
│     正常文字   →  显示给用户  →  存进 memory                       │
└──────────────────────────────────────────────────────────────────┘
                              ↓  ↑  HTTP JSON / SSE
┌──────────────────── 后端 main.py (FastAPI) ─────────────────────┐
│                                                                 │
│   接收 question + history + last_bazi                           │
│   调用 master.ask_stream(question, history, last_bazi)          │
│   通过 StreamingResponse 流式返回给前端                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓  ↑
┌──────────────────── 核心 master.py (Agent) ─────────────────────┐
│                                                                 │
│   完全无状态！不存 self.memory / self.last_bazi                  │
│                                                                 │
│   ask_stream 干三件事：                                          │
│     1. 检测输入有日期 → 生成 [SYSTEM_DATA:last_bazi:...]         │
│     2. _enhance_question(question, last_bazi) 工作流编排         │
│     3. 先 yield 系统标记，再 yield 正常文字                       │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ 技术栈

| 层级 | 技术 |
|---|---|
| 框架 | LangChain 1.x (`create_agent`) |
| 模型 | 智谱 GLM-4-Flash (OpenAI 兼容接口) |
| 后端 | FastAPI + Pydantic + StreamingResponse (SSE) |
| 前端 | Streamlit |
| 向量库 | Chroma + BAAI/bge-small-zh-v1.5 |
| 八字排盘 | lunar_python |
| 语言 | Python 3.10+ |

## 📂 项目结构

.
├── main.py           # FastAPI 后端入口（接口层）
├── master.py         # Agent 核心业务逻辑（无状态）
├── app.py            # Streamlit 前端（UI 与状态管理）
├── tools_bazi.py     # 八字排盘工具（纯函数）
├── build_rag.py      # 知识库构建脚本
├── requirements.txt  # 依赖清单
├── knowledge/        # 命理知识库源文件
├── chroma_db/        # 本地向量库（.gitignore 忽略）
├── .env              # 环境变量（不提交）
├── .gitignore
└── README.md

## 🔍 关键实现细节

### 1. 工作流编排（`master.py::_enhance_question`）
代码通过正则与关键词强制接管决策权：
- 检测到日期 → 强制调用 `get_bazi` 排盘 + 强制 RAG 检索流年运势
- 检测到生肖 → 强制 RAG 检索生肖合冲刑害
- 检测到命理关键词 → 兜底强制检索
- 其余情况 → 放行给模型

### 2. 无状态状态传递（`master.py::ask_stream` + `app.py`）
# 后端：把排盘结果包进隐藏标记
bazi_marker = f"[SYSTEM_DATA:last_bazi: {bazi_result}]\n"
yield bazi_marker

# 前端：拦截标记，存入 session_state
if text.startswith("[SYSTEM_DATA:"):
    m = re.search(r"last_bazi: (.*)]", text, re.DOTALL)
    if m:
        current["last_bazi"] = m.group(1).strip()
    continue

### 3. UTF-8 中文截断修复（`app.py`）
for chunk in res.iter_content(chunk_size=None):
    text = chunk.decode('utf-8', errors='ignore')  # 关键：errors='ignore'


## ⚠️ 已知局限

- 命理知识库粒度较粗，小模型基于理论推理时仍可能出现幻觉。后续可通过精细化知识库或换用更强大的模型（如 GLM-4-Plus）优化。
- 关键词兜底机制基于正则与硬编码词表，存在漏网之鱼。后续可通过语义路由（Semantic Router）或轻量级意图识别模型替代。
- 尚未做持久化存储，重启服务会丢失会话历史。后续可引入 Redis 或 SQLite。
- 未做用户鉴权，仅用于个人演示与学习。
