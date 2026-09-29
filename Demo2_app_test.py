import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from Demo2 import Master

load_dotenv()
st.set_page_config(page_title="陈大师算命", page_icon="🔮", layout="wide")
st.title("🔮 陈大师在线算命")

# 1. 全局单例：只加载一次 RAG
@st.cache_resource
def get_master():
    return Master(api_key=os.getenv("Zhipu_api_key"))

master = get_master()

# 2. session_state 初始化
if "sessions" not in st.session_state:
    sid = str(uuid.uuid4())[:8]
    st.session_state.sessions = {sid: {"title": "新对话", "memory": []}}
    st.session_state.current_sid = sid

# 3. 侧边栏多会话管理
with st.sidebar:
    st.header("💬 会话列表")
    if st.button("➕ 新建对话", use_container_width=True):
        sid = str(uuid.uuid4())[:8]
        st.session_state.sessions[sid] = {"title": "新对话", "memory": []}
        st.session_state.current_sid = sid
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
        st.session_state.sessions[current_sid] = {"title": "新对话", "memory": []}
        st.rerun()

# 4. 主区域渲染
sid = st.session_state.current_sid
current = st.session_state.sessions[sid]

# 关键：把当前会话的 memory 塞给全局单例的 master
master.memory = list(current["memory"])

# 渲染历史聊天
for msg in current["memory"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 5. 接收输入并流式输出
if prompt := st.chat_input("你想问大师什么？"):
    with st.chat_message("user"):
        st.write(prompt)

    # 首条消息更新标题
    if current["title"] == "新对话":
        current["title"] = prompt[:12] + ("…" if len(prompt) > 12 else "")

    with st.chat_message("assistant"):
        # 【这里非常重要】直接调用我们封装好的 ask_stream，不要绕路！
        # 这样 _enhance_question 的日期检测、排盘注入才会生效！
        answer = st.write_stream(master.ask_stream(prompt))

    # 6. 把 master 更新后的 memory 存回当前会话
    current["memory"] = list(master.memory)