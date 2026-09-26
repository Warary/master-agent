import streamlit as st
import os
from dotenv import load_dotenv
from Demo2 import Master

load_dotenv()

st.set_page_config(page_title="陈大师算命", page_icon="🔮", layout="wide")
st.title("🔮 陈大师在线算命")

# === 1. 用侧边栏作为“探针控制台” ===
with st.sidebar:
    if st.button("🧹 开启新对话（清空当前记忆）"):
        st.session_state.messages = []
        st.session_state.master.memory = []
        if os.path.exists(st.session_state.master.MEMORY_KEY):
            os.remove(st.session_state.master.MEMORY_KEY)
        st.success("记忆已清空！")
        st.rerun()
    st.header("🔍 后台探针")
    st.write("这里可以看到大师每一次推演的内部逻辑。")
    probe_placeholder = st.empty()

# === 2. 初始化 Master 和聊天记录 ===
if "master" not in st.session_state:
    API_KEY = os.getenv("Zhipu_api_key")
    st.session_state.master = Master(api_key=API_KEY)
    st.session_state.messages = []

# 显示历史聊天记录（网页界面上的）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# === 3. 接收用户输入并处理 ===
if prompt := st.chat_input("你想问大师什么？"):
    # 1. 显示用户输入
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 调用大师（流式输出）
    with st.chat_message("assistant"):
        # 拼接历史记忆
        messages = []
        for msg in st.session_state.master.memory[-20:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 直接塞原始问题，生日已经由大师自己在对话里问
        messages.append({"role": "user", "content": prompt})

        # 用容器在生成器外面捕获最终的完整 state（用于探针）
        final_state_container = {}

        def agent_stream_wrapper():
            """包装生成器：yield 文字给前端，同时把最终 state 存进容器"""
            for mode, data in st.session_state.master.agent.stream(
                {"messages": messages},
                stream_mode=["messages", "values"]
            ):
                if mode == "messages":
                    chunk, metadata = data
                    # 过滤掉工具调用的参数碎片
                    if getattr(chunk, "tool_call_chunks", None):
                        continue
                    if chunk.content:
                        yield chunk.content
                elif mode == "values":
                    final_state_container["state"] = data

        # st.write_stream 会自动渲染打字机效果，并返回完整的字符串
        answer = st.write_stream(agent_stream_wrapper())
        
        # 获取最终 state 用于侧边栏探针展示
        final_state = final_state_container.get("state")

    # 3. 更新侧边栏探针（流式结束后展示）
    if final_state:
        with probe_placeholder.container():
            st.markdown("**Agent 执行轨迹:**")
            for idx, msg in enumerate(final_state["messages"]):
                content_preview = str(msg.content)[:40].replace("\n", " ") if msg.content else "(无文本内容)"
                tool_info = ""
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_info = f" 👉 调用: {msg.tool_calls[0]['name']}"
                st.text(f"  第{idx+1}步 | {msg.type}: {content_preview}...{tool_info}")
    # 4. 保存记忆
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.master.memory.append({"role": "user", "content": prompt})
    st.session_state.master.memory.append({"role": "assistant", "content": answer})
    st.session_state.master.memory = st.session_state.master.memory[-20:]
    st.session_state.master.save_memory()