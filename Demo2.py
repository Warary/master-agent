from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
from datetime import datetime
import json
import os

# 1. 禁用联网检查，防止卡在 WinError 10060
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. RAG 相关依赖
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
load_dotenv()

class Master:
    def __init__(self, api_key: str):

        today_date = datetime.now().strftime("%Y-%m-%d")

        # 智谱 OpenAI 兼容接口
        self.master_llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            temperature=0.9,
            streaming=True,
        )

        self.MEMORY_KEY = "chat_history.json"
        self.memory = []

        # ============ 新增 RAG 初始化 ============
        print("[初始化] 正在加载 RAG 向量库...")
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu", "local_files_only": True}, # 强制本地加载
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Chroma(
            collection_name="suanming_knowledge",
            embedding_function=embedding,
            persist_directory="./chroma_db"
        )

        # 系统人设
        self.SYSTEM_PROMPT = f"""你是一个非常厉害的算命先生，你叫陈玉楼，人称陈大师。【今天的真实日期是：{today_date}，如果有人问你今天是哪一天，你必须回答这个日期。】
        以下是你的个人设定：
        1. 你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉，看命运八字等。
        2. 你大约 60 岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明。
        3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
        4. 当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的
        5. 你总是用繁体中文来作答。
        6. 你从不说自己是一个人工智能助手或AI,而是以老夫、老朽等自称。
        7. 你需要敏锐地感知用户当前的情绪（开心、悲伤、愤怒、焦虑等），并在你的回答里自然而然地调整语气，比如愤怒时安抚，悲伤时宽慰。
        以下是你常说的一些口头禅：
        1. “命里有时终须有，命里无时莫强求。”
        2. “山重水复疑无路，柳暗花明又一村。”
        3. “金山竹影几千秋，云锁高飞水自流。”
        4. “伤情最是晚凉天，憔悴斯人不堪怜。”
        以下是你算命的过程：
        1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
        2. 当用户询问生肖运势、五行、八字、天干地支等命理知识时，必须优先调用 search_knowledge 工具检索知识库。
            检索结果只是【参考资料】，你必须：
            - 用中文重新组织语言，结合用户的具体问题来回答
            - 绝对不允许把参考资料原文直接复制粘贴给用户
            - 如果参考资料没有直接答案，就基于参考资料里的理论进行推理，并告诉用户这是老夫根据命理推演所得
            - 如果参考资料完全不相关，就说“老夫学艺不精，此事暂且算不出来
        3. 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
        4. 你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。
        5. 你会保存每一次的聊天记录，以便在后续的对话中使用。
        """

        # ============ 新增：利用闭包在类内定义 RAG 工具 ============
        @tool
        def search_knowledge(query: str) -> str:
            """从算命知识库中检索相关信息。
            当用户问到五行、天干地支、生肖运势、八字规则等内容时，
            优先调用此工具查询，不要凭记忆瞎编。
            query 参数是用户问题的核心关键词，例如 '属兔运势'、'五行相生'。
            """
            results = self.vector_store.similarity_search(query, k=3)
            if not results:
                return "知识库中未找到相关信息。"
            # 用 XML 标签包裹，明确告诉模型"这是资料，不是答案"
            content = "\n---\n".join([doc.page_content for doc in results])
            return f"<参考资料>\n{content}\n</参考资料>\n请基于上述资料，用你自己的话回答用户的问题，不要原文照抄。"

        self.tools = [search_knowledge]

        # 新版 create_agent 标准参数：model / tools / system_prompt
        self.agent = create_agent(
            model=self.master_llm,
            tools=self.tools,
            system_prompt=self.SYSTEM_PROMPT
        )

    def ask(self, question: str) -> str:
        # 只取最近 20 条发给大模型（读取限制）
        recent = self.memory[-20:]
        messages = []
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})
        print(f"[探针 2] 实际发给大师的纸条:\n{question}")
        messages.append({"role": "user", "content": question})
        answer = ""
        final_state = None
        print("\n陈大师：", end="", flush=True)

        for mode, data in self.agent.stream(
            {"messages": messages},
            stream_mode=["messages", "values"]
        ):
            if mode == "messages":
                chunk, metadata = data
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    answer += chunk.content
            elif mode == "values":
                final_state = data
        print()
        self.memory.append({"role": "user", "content": question})
        self.memory.append({"role": "assistant", "content": answer})
        # 关键：截断列表本身（存储限制）
        self.memory = self.memory[-20:]
        return answer

    def save_memory(self):
        with open(self.MEMORY_KEY, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    API_KEY = os.getenv("Zhipu_api_key")
    master = Master(api_key=API_KEY)
    print("✨ 陈大师已上线")
    print("  输入 'q' 退出 | 输入 'clear' 清空记忆")
    while True:
        q = input("你问：")
        if q.lower() == "q":
            break
        if q.lower() == "clear":
            master.memory = []
            if os.path.exists(master.MEMORY_KEY):
                os.remove(master.MEMORY_KEY)
            print("🧹 记忆已清空，大师重新开张\n")
            continue
        if not q.strip():
            continue
        master.ask(q)
    master.save_memory()