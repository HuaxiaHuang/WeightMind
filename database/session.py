"""
数据库连接池 + 初始化
"""
import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from loguru import logger

from core.config import settings
from database.models import Base

# 创建异步引擎
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # 连接前检测活性
)

# 会话工厂
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """初始化数据库表结构"""
    # ✨ 增加一层环境变量判断：如果明确是生产环境，则跳过自动建表
    # 假设你在云端 .env 文件中配置了 ENVIRONMENT=production
    env_mode = os.getenv("ENVIRONMENT", "development")
    
    if env_mode == "production":
        logger.info("🟢 生产环境：跳过自动 create_all，请使用 Alembic 管理数据库迁移")
        return

    # 🟡 开发环境：保留自动建表逻辑
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表初始化完成 (自动建表已执行)")

async def get_db():
    """FastAPI 依赖注入：获取数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
