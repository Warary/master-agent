import os

# 1. 强制使用国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 2. 全局开启离线模式，禁止联网检查
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("[提示] 正在加载本地嵌入模型...")

# 3. 把 local_files_only 放进 model_kwargs 里
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={
        "device": "cpu",
        "local_files_only": True  # 👈 关键修正：参数放入 model_kwargs
    },
    encode_kwargs={"normalize_embeddings": True}
)

# 加载已存在的向量库
vector_store = Chroma(
    collection_name="suanming_knowledge",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

# 测试检索
query = "属马的今年运势怎么样"

print(f"\n[查询] {query}")
results = vector_store.similarity_search(query, k=3)
print(f"[召回] 找到 {len(results)} 条相关文档\n")

for i, doc in enumerate(results):
    print(f"--- 第 {i+1} 条 ---")
    print(doc.page_content[:200])
    print()