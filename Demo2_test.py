from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
from datetime import datetime
from lunar_python import Solar
from tools_bazi import get_bazi
import json
import os
import re

# 1. 禁用联网检查，防止卡在 WinError 10060
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. RAG 相关依赖
from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
load_dotenv()

# 命理相关关键词，用于判断用户是否在问命理
DESTINY_KEYWORDS = ["运势", "八字", "命", "今年", "事业", "财运", "感情", "健康", "流年"]


class Master:
    def __init__(self, api_key: str):

        today = datetime.now()
        today_date = today.strftime("%Y-%m-%d")
        this_year_ganzhi = Solar.fromYmdHms(
            today.year, today.month, today.day, 0, 0, 0
        ).getLunar().getYearInGanZhiByLiChun()

        self.master_llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            temperature=0.8,
            streaming=True,
        )

        self.MEMORY_KEY = "chat_history.json"
        self.memory = []
        self.last_bazi = None  # 缓存最近一次排盘结果

        # ============ RAG 初始化 ============
        print("[初始化] 正在加载 RAG 向量库...")
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Chroma(
            collection_name="suanming_knowledge",
            embedding_function=embedding,
            persist_directory="./chroma_db"
        )

        self.SYSTEM_PROMPT = f"""你是陈玉楼，人称陈大师，一位精通阴阳五行的算命先生。
        【时间锚点】（极其重要，不得算错）
        - 今天日期：{today_date}
        - 今年干支：{this_year_ganzhi}年
        - 若用户问今天日期或今年运势，必须以这两个信息为准，不得凭记忆推算年份干支。

        【身份】
        - 年约六旬，早年是湘西一带赫赫有名的土匪头子，因盗墓被毒气所伤，双目失明。
        - 好友有胡八一、雪莉杨、王胖子，皆为摸金校尉。
        - 精通八字、紫微斗数、姓名测算、占卜吉凶。

        【说话方式】
        - 全程使用繁体中文。
        - 自称"老夫""老朽"，绝不承认自己是 AI 或人工智能助手。
        - 回答时自然穿插一句口头禅，不要堆砌、不要复读：
        "命里有时终须有，命里无时莫强求。"
        "山重水复疑无路，柳暗花明又一村。"
        "金山竹影几千秋，云锁高飞水自流。"
        "伤情最是晚凉天，憔悴斯人不堪怜。"
        - 察觉用户情绪（喜、悲、怒、忧）时调整语气：怒则安抚，悲则宽慰，喜则同庆。

        【排盘数据使用规则】（重要）
        - 当用户消息里出现【系统排盘数据】标签时，说明系统已替你排好盘。
        - 你必须基于该数据作答，不得另行推算或编造干支。
        - 该数据是【内部资料】，不要原样输出给用户，请用陈大师的口吻重新组织。

        【工具调用规则】
        1. 用户询问五行、天干地支、生肖运势、命理规则等知识时，调用 search_knowledge 检索。
        2. 检索结果是【参考资料】而非答案：
        - 结合用户的具体问题，用自己的话重新组织
        - 绝不原文照抄
        - 资料没有直接答案时，基于其中理论推演，并说明"此乃老夫推演所得"
        - 资料完全不相关时，直接说"老夫学艺不精，此事暂且算不出来"，不要编造

        【对话流程】
        - 初次见面时，先问用户的姓名与出生年月日，以便后续排盘。
        """

        @tool
        def search_knowledge(query: str) -> str:
            """从算命知识库中检索相关信息。
            当用户问到五行、天干地支、生肖运势、八字规则等内容时，
            优先调用此工具查询，不要凭记忆瞎编。
            query 参数是用户问题的核心关键词，例如 '属兔运势'、'五行相生'。
            """
            print(f"  🔍 search_knowledge(\"{query}\")", end=" ")
            results = self.vector_store.similarity_search(query, k=3)
            print(f"→ 命中 {len(results)} 条")
            if not results:
                return "知识库中未找到相关信息。"
            content = "\n---\n".join([doc.page_content for doc in results])
            return (
                f"<参考资料>\n{content}\n</参考资料>\n"
                "回答要求：先用一句话总结资料的核心观点，再结合用户的具体问题展开。"
                "语气像老先生聊天，不要像念书。"
            )

        self.tools = [search_knowledge, get_bazi]

        self.agent = create_agent(
            model=self.master_llm,
            tools=self.tools,
            system_prompt=self.SYSTEM_PROMPT
        )

    def _enhance_question(self, question: str) -> str:
        """工程兜底：检测出生日期或命理提问，强制拼入排盘数据。"""
        date_match = re.search(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})", question)
        if date_match:
            y, m, d = date_match.groups()
            birth = f"{y}-{int(m):02d}-{int(d):02d} 12:00"
            bazi = get_bazi.invoke({"birth_datetime": birth})
            self.last_bazi = bazi
            print(f"  🔧 get_bazi({birth})")
            return (
                f"{question}\n\n"
                f"【系统排盘数据】\n{bazi}\n"
                f"【请基于以上数据用陈大师口吻作答，不要原样输出，不得自行推算】"
            )

        if self.last_bazi and any(kw in question for kw in DESTINY_KEYWORDS):
            print(f"  🔧 复用上次排盘数据")
            return (
                f"{question}\n\n"
                f"【系统排盘数据（该用户此前提供）】\n{self.last_bazi}\n"
                f"【请基于以上数据作答】"
            )

        return question

    def ask(self, question: str) -> str:
        recent = self.memory[-20:]
        messages = []
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

        enhanced = self._enhance_question(question)

        # 只在被增强时提示，否则不打印，免得刷屏
        if enhanced != question:
            print(f"  📎 已注入上下文 (+{len(enhanced) - len(question)} 字)")

        messages.append({"role": "user", "content": enhanced})

        answer = ""
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
        print()
        self.memory.append({"role": "user", "content": question})
        self.memory.append({"role": "assistant", "content": answer})
        self.memory = self.memory[-20:]
        return answer

    def ask_stream(self, question: str):
        """流式版本：yield token，供 Streamlit 使用。结束后更新 memory。"""
        recent = self.memory[-20:]
        messages = []
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

        enhanced = self._enhance_question(question)
        messages.append({"role": "user", "content": enhanced})

        full_answer = ""
        for mode, data in self.agent.stream(
            {"messages": messages},
            stream_mode=["messages", "values"]
        ):
            if mode == "messages":
                chunk, metadata = data
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                if chunk.content:
                    full_answer += chunk.content
                    yield chunk.content

        self.memory.append({"role": "user", "content": question})
        self.memory.append({"role": "assistant", "content": full_answer})
        self.memory = self.memory[-20:]

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
            master.last_bazi = None
            if os.path.exists(master.MEMORY_KEY):
                os.remove(master.MEMORY_KEY)
            print("🧹 记忆已清空，大师重新开张\n")
            continue
        if not q.strip():
            continue
        master.ask(q)
    master.save_memory()