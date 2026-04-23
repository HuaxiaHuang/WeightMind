"""
Sci_RAG FastAPI 应用入口
"""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes_chat import router as chat_router
from api.routes_upload import router as upload_router
from api.routes_papers import router as papers_router
from database.session import init_db
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化数据库"""
    logger.info("🚀 Sci_RAG 系统启动中...")
    await init_db()
    logger.info("✅ 数据库连接就绪")
    yield
    logger.info("🛑 Sci_RAG 系统关闭")


app = FastAPI(
    title="Sci_RAG — 科研论文深度研究系统",
    description="基于 LlamaIndex + Qdrant + BGE 的科研论文 RAG 系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允许 Streamlit 前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router, prefix="/chat", tags=["对话"])
app.include_router(upload_router, prefix="/upload", tags=["文件上传"])
app.include_router(papers_router, prefix="/papers", tags=["论文管理"])


@app.get("/health", tags=["系统"])
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
