from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os 
from dotenv import load_dotenv
from Demo2 import Master

load_dotenv()

app = FastAPI()

# 全局单例：后端只负责计算，不负责保存状态
master = Master(os.getenv("Zhipu_api_key"))

# 跨域配置（允许前端 Streamlit 调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求体模型：接收前端发来的状态
class ChatQuestion(BaseModel):
    question: str
    history: list = []
    last_bazi: str | None = None

@app.post("/chat")
async def chat(req: ChatQuestion):
    async def event_generator():
        # 调用无状态的 ask_stream
        for chunk in master.ask_stream(req.question, req.history, req.last_bazi):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")