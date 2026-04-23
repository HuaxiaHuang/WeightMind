"""
数据库初始化脚本
用法: python scripts/init_db.py
"""
import asyncio
import sys
from pathlib import Path

# 确保能找到项目根目录的模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine

from core.config import settings
from database.models import Base
from database.qdrant_client import qdrant_manager


async def init_postgres():
    """初始化 PostgreSQL 表结构"""
    logger.info("正在初始化 PostgreSQL 表...")
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    logger.success("✅ PostgreSQL 表创建完成")


async def init_qdrant():
    """测试 Qdrant 连接"""
    logger.info("正在检查 Qdrant 连接...")
    try:
        collections = await qdrant_manager.list_collections()
        logger.success(f"✅ Qdrant 连接成功，已有 {len(collections)} 个 Collection")
    except Exception as e:
        logger.error(f"❌ Qdrant 连接失败: {e}")
        logger.info("请确认 Qdrant 已启动: docker-compose up qdrant -d")


def init_directories():
    """创建必要的本地目录"""
    dirs = [
        settings.RAW_DATA_DIR,
        settings.TEMP_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"目录就绪: {d}")
    logger.success("✅ 本地目录初始化完成")


async def main():
    logger.info("=" * 50)
    logger.info("Sci_RAG 系统初始化")
    logger.info("=" * 50)

    init_directories()
    await init_postgres()
    await init_qdrant()

    logger.info("=" * 50)
    logger.success("🎉 初始化完成！现在可以启动系统了")
    logger.info("=" * 50)
    logger.info("\n启动步骤：")
    logger.info("1. 基础设施: docker-compose up -d")
    logger.info("2. FastAPI:   python main.py")
    logger.info("3. Celery:    celery -A worker.celery_app worker --loglevel=info -Q parsing,indexing")
    logger.info("4. 前端:      streamlit run frontend/app.py")


if __name__ == "__main__":
    asyncio.run(main())
