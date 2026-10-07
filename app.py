import uuid
import re
import streamlit as st
from dotenv import load_dotenv
import requests

load_dotenv()
st.set_page_config(page_title="陈大师算命", page_icon="🔮", layout="wide")
st.title("🔮 陈大师在线算命")

def new_session():
    """新建会话，同时初始化 last_bazi 字段"""
    sid = str(uuid.uuid4())[:8]
    st.session_state.sessions[sid] = {"title": "新对话", "memory": [], "last_bazi": None}
    st.session_state.current_sid = sid

# 1. session_state 初始化
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    new_session()

# 2. 侧边栏多会话管理
with st.sidebar:
    st.header("💬 会话列表")
    if st.button("➕ 新建对话", use_container_width=True):
        new_session()
        st.rerun()

    for sid, data in st.session_state.sessions.items():
        is_active = (sid == st.session_state.current_sid)
        label = ("🟢 " if is_active else "⚪ ") + data["title"]
        if st.button(label, key=f"btn_{sid}", use_container_width=True):
            st.session_state.current_sid = sid
            st.rerun()

    st.divider()
    if st.button("🧹 清空当前会话", use_container_width=True):
        current_sid = st.session_state.current_sid
        st.session_state.sessions[current_sid] = {"title": "新对话", "memory": [], "last_bazi": None}
        st.rerun()

# 3. 主区域渲染
sid = st.session_state.current_sid
current = st.session_state.sessions[sid]

# 渲染历史聊天
for msg in current["memory"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 4. 接收输入并流式输出
if prompt := st.chat_input("你想问大师什么？"):
    if current["title"] == "新对话":
        current["title"] = prompt[:12] + ("…" if len(prompt) > 12 else "")

    current["memory"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("大师正在推算天机..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/chat",
                    json={
                        "question": prompt,
                        "history": current["memory"],
                        "last_bazi": current.get("last_bazi")  # 核心：带上历史排盘
                    },
                    timeout=60,
                    stream=True
                )
                if res.status_code == 200:
                    def stream_reader():
                        for chunk in res.iter_content(chunk_size=None):
                            if not chunk:
                                continue
                            text = chunk.decode('utf-8', errors='ignore')
                            # 核心：拦截系统标记，存入 session_state
                            if text.startswith("[SYSTEM_DATA:"):
                                m = re.search(r"last_bazi: (.*)]", text, re.DOTALL)
                                if m:
                                    current["last_bazi"] = m.group(1).strip()
                                continue  # 不把标记显示给用户
                            
                            yield text

                    answer = st.write_stream(stream_reader())
                    current["memory"].append({"role": "assistant", "content": answer})
                else:
                    st.error(f"请求失败，错误码：{res.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("大肥鱼提示：请先启动 FastAPI 后端（运行 fastapi dev）！")