"""
FastAPI 后端服务

提供 RESTful API，让 Streamlit 前端或其他客户端调用 RAG 能力。
实现了前后端分离架构：
  Streamlit / 其他前端 → FastAPI (REST API) → RAG Service
"""

import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag import RagService
from knowledge_base import KnowledgeBaseService
import config_data as config

# ── 创建 FastAPI 应用 ──

app = FastAPI(
    title="RAG 智能客服系统 API",
    version="1.0.0",
    description="基于 LangChain + ChromaDB + 通义千问 的 RAG 问答系统后端",
)

# 允许跨域（Streamlit 或其他前端可以调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 初始化服务（单例） ──

rag = RagService()
kb = KnowledgeBaseService()


# ── 请求/响应模型 ──

class ChatRequest(BaseModel):
    input: str


class ChatResponse(BaseModel):
    answer: str


# ── API 路由 ──

@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "rag-api"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    非流式问答

    发送用户问题，返回完整回答。
    使用示例:
        curl -X POST http://localhost:8000/api/chat \\
            -H "Content-Type: application/json" \\
            -d '{"input": "春天穿什么颜色"}'
    """
    try:
        result = rag.chain.invoke(
            {"input": req.input},
            config=config.session_config,
        )
        return ChatResponse(answer=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式问答（Server-Sent Events）

    返回流式响应，前端可以逐字显示。
    使用示例:
        curl -X POST http://localhost:8000/api/chat/stream \\
            -H "Content-Type: application/json" \\
            -d '{"input": "春天穿什么颜色"}'
    """
    async def event_stream():
        try:
            async for chunk in rag.chain.astream(
                {"input": req.input},
                config=config.session_config,
            ):
                if chunk:
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件到知识库

    支持 .txt 和 .pdf 文件，自动提取文本并向量化存储。
    使用示例:
        curl -X POST http://localhost:8000/api/upload \\
            -F "file=@/path/to/your/file.txt"
    """
    allowed_extensions = (".txt", ".pdf")
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型，仅支持 {', '.join(allowed_extensions)}",
        )

    try:
        file_bytes = await file.read()
        result = kb.upload_by_file(file_bytes, file.filename)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


# ── 直接运行时 ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
