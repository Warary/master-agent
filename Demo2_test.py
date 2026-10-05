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
DESTINY_KEYWORDS = ["运势", "八字", "命", "今年", "事业", "财运", "感情", "健康", "流年","五行"]


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
            results = self.vector_store.similarity_search(query, k=2)
            print(f"→ 命中 {len(results)} 条")
            if not results:
                return "知识库中未找到相关信息。"
            
            # 核心优化：剥离Markdown格式，让模型无法直接复制
            content_parts = []
            for doc in results:
                text = doc.page_content
                text = re.sub(r'#+\s*', '', text)          # 去掉标题符号 #
                text = re.sub(r'\*\*(.+?)\*\*', r'\1', text) # 去掉加粗 **
                text = re.sub(r'\n{3,}', '\n\n', text)       # 压缩多余空行
                content_parts.append(text.strip())
            
            content = "\n".join(content_parts)
            
            # 核心优化：下死命令，告诉它这是素材不是答案
            return (
                f"【内部参考素材】\n{content}\n\n"
                f"【铁律】以上是命理素材，不是给你的回答模板。"
                f"你必须像陈大师坐在茶桌旁闲聊一样，用自己的话把素材里的道理讲出来。"
                f"绝对禁止出现原文中的任何整句。如果讲不出来，就说'老夫学艺不精'。"
            )

        self.tools = [search_knowledge, get_bazi]

        self.agent = create_agent(
            model=self.master_llm,
            tools=self.tools,
            system_prompt=self.SYSTEM_PROMPT
        )

    def _enhance_question(self, question: str, last_bazi: str) -> str:
        """工作流编排：代码接管排盘与检索，模型只负责总结和说人话。"""
        
        # 1. 检测用户输入里是否包含出生日期
        date_match = re.search(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})", question)
        
        # 2. 检测用户输入里是否包含生肖（比如“属马”）
        zodiac_match = re.search(r"(属[鼠牛虎兔龙蛇马羊猴鸡狗猪])", question)
        
        # 获取当年的干支（用于组合检索词）
        this_year = datetime.now().year
        this_year_ganzhi = Solar.fromYmdHms(this_year, 1, 1, 0, 0, 0).getLunar().getYearInGanZhiByLiChun()

        # ========== 分支 A：用户给了出生日期，进入完整排盘流程 ==========
        if date_match:
            y, m, d = date_match.groups()
            birth = f"{y}-{int(m):02d}-{int(d):02d} 12:00"
            bazi_result = get_bazi.invoke({"birth_datetime": birth})
            print(f"  🔧 强排盘 get_bazi({birth})")
            
            # 代码强制构造检索词
            query = f"{y}年 {this_year_ganzhi}年 流年 运势 五行生克"
            docs = self.vector_store.similarity_search(query, k=3)
            rag_content = "\n---\n".join([doc.page_content for doc in docs]) if docs else "知识库暂无相关记录"
            print(f"  🔍 强制 RAG 检索 → 命中 {len(docs)} 条")
            
            return (
                f"{question}\n\n"
                f"【系统排盘数据】\n{bazi_result}\n\n"
                f"【已检索的命理资料】\n{rag_content}\n\n"
                f"【任务】请结合上述数据和资料，用陈大师的口吻回答。绝对不要原样输出数据和资料。"
            )

        # ========== 分支 B：用户给了生肖，没给日期 ==========
        if zodiac_match and any(kw in question for kw in DESTINY_KEYWORDS):
            zodiac = zodiac_match.group(1)[1]  # 提取“马”
            print(f"  🔧 识别生肖：{zodiac}，强制 RAG 检索...")
            
            query = f"{zodiac} {this_year_ganzhi}年 流年 运势 合冲刑害"
            docs = self.vector_store.similarity_search(query, k=3)
            rag_content = "\n---\n".join([doc.page_content for doc in docs]) if docs else "知识库暂无相关记录"
            print(f"  🔍 强制 RAG 检索 → 命中 {len(docs)} 条")
            
            return (
                f"{question}\n\n"
                f"【系统识别结果】用户生肖：{zodiac}，当年流年：{this_year_ganzhi}年\n\n"
                f"【已检索的命理资料】\n{rag_content}\n\n"
                f"【任务】请结合上述流年与生肖的生克关系，用陈大师口吻回答。严禁自行编造相冲相合关系，必须严格基于资料。"
            )

        # ========== 分支 C：万能兜底（覆盖历史排盘与一般命理问答） ==========
        # 只要涉及命理关键词，不管有没有历史排盘，一律强制检索，杜绝模型瞎编
        if any(kw in question for kw in DESTINY_KEYWORDS):
            print(f"  🔧 识别到命理问题，强制 RAG 检索...")
            
            query = f"八字 流年 运势 事业 财运 感情 五行 {question}" if last_bazi else question
            if last_bazi:
                print(f"  🔧 复用历史排盘数据（已注入检索词）")
            else:
                query = question
                
            docs = self.vector_store.similarity_search(query, k=3)
            rag_content = "\n---\n".join([doc.page_content for doc in docs]) if docs else "知识库暂无相关记录"
            print(f"  🔍 强制 RAG 检索 → 命中 {len(docs)} 条")
            
            context_block = f"【系统排盘数据（历史）】\n{last_bazi}\n\n" if last_bazi else ""
            
            return (
                f"{question}\n\n"
                f"{context_block}"
                f"【已检索的命理资料】\n{rag_content}\n\n"
                f"【任务】请结合上述资料，用陈大师口吻回答。绝对不要原样输出资料。"
            )

        # ========== 分支 D：完全无关的问题，放行给模型 ==========
        return question

    def ask_stream(self, question: str, history: list, last_bazi: str):
        """流式版本：yield token，附带系统标记，供前端保存状态。"""
        recent = history[-20:] if history else []
        messages = list(recent)

        # 1. 检测日期，生成系统标记（供前端保存 last_bazi）
        bazi_marker = ""
        date_match = re.search(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})", question)
        if date_match:
            y, m, d = date_match.groups()
            birth = f"{y}-{int(m):02d}-{int(d):02d} 12:00"
            bazi_result = get_bazi.invoke({"birth_datetime": birth})
            # 把排盘结果包装进系统标记，前端会拦截并保存
            bazi_marker = f"[SYSTEM_DATA:last_bazi: {bazi_result}]\n"

        # 2. 工作流编排（只用 last_bazi，不再用 self）
        enhanced = self._enhance_question(question, last_bazi)
        if enhanced != question:
            print(f"  📎 已注入上下文 (+{len(enhanced) - len(question)} 字)")
        messages.append({"role": "user", "content": enhanced})

        # 3. 优先发送系统标记
        if bazi_marker:
            yield bazi_marker

        # 4. 正常流式输出
        for mode, data in self.agent.stream(
            {"messages": messages},
            stream_mode=["messages", "values"]
        ):
            if mode == "messages":
                chunk, metadata = data
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                if chunk.content:
                    yield chunk.content

    def save_memory(self):
        with open(self.MEMORY_KEY, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    API_KEY = os.getenv("Zhipu_api_key")
    master = Master(api_key=API_KEY)
    print("✨ 陈大师已上线")
    print("  输入 'q' 退出 | 输入 'clear' 清空记忆")
    
    # 终端版本的本地状态
    terminal_memory = []
    terminal_last_bazi = None

    while True:
        q = input("你问：")
        if q.lower() == "q":
            break
        if q.lower() == "clear":
            terminal_memory = []
            terminal_last_bazi = None
            if os.path.exists(master.MEMORY_KEY):
                os.remove(master.MEMORY_KEY)
            print("🧹 记忆已清空，大师重新开张\n")
            continue
        if not q.strip():
            continue

        # 调用无状态的 ask_stream，并手动拼接本地状态
        full_answer = ""
        print("\n陈大师：", end="", flush=True)
        for chunk in master.ask_stream(q, terminal_memory, terminal_last_bazi):
            # 拦截系统标记，终端版不显示，但提取出 last_bazi
            if chunk.startswith("[SYSTEM_DATA:"):
                m = re.search(r"last_bazi: (.*)]", chunk, re.DOTALL)
                if m:
                    terminal_last_bazi = m.group(1).strip()
                continue
            
            # 正常文本内容
            print(chunk, end="", flush=True)
            full_answer += chunk
        print()

        # 更新终端本地记忆
        terminal_memory.append({"role": "user", "content": q})
        terminal_memory.append({"role": "assistant", "content": full_answer})
        terminal_memory = terminal_memory[-20:]  # 滑动窗口