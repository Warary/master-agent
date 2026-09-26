"""
RAG 知识库构建脚本
运行一次即可，之后检索会直接读取本地向量库
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv

# ========== 1. 文档加载 ==========
from langchain_community.document_loaders import TextLoader

loader = TextLoader("knowledge/mingli.txt", encoding="utf-8")
docs = loader.load()
print(f"[加载] 读取到 {len(docs)} 篇文档")

# ========== 2. 文本分割 ==========
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,          # 每块最多  字
    chunk_overlap=60,        # 相邻块重叠  字，保证语义不断
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"[切块] 切出 {len(chunks)} 个文本块")

# ========== 3. 本地嵌入模型 ==========
from langchain_huggingface import HuggingFaceEmbeddings

# 模型名要和下载的一致
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},          # 有 GPU 改成 "cuda"
    encode_kwargs={"normalize_embeddings": True}
)
print("[嵌入] 本地模型加载完成")

# ========== 4. 存入 Chroma ==========
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="suanming_knowledge",
    embedding_function=embedding,
    persist_directory="./chroma_db"          # 向量库存在这里
)

# 分批入库，避免内存爆炸
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vector_store.add_documents(batch)
    print(f"[入库] 第 {i//batch_size + 1} 批完成")

print("[完成] 向量库构建完毕，路径：./chroma_db")