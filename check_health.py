import os
import asyncio
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from qdrant_client import QdrantClient
import redis.asyncio as redis

# 加载 .env 文件中的环境变量
load_dotenv()

async def check_health():
    print("🔍 开始系统全面体检...\n" + "="*30)
    
    # 1. 检查 PostgreSQL 关系型数据库
    try:
        db_url = os.getenv("DATABASE_URL")
        engine = create_async_engine(db_url)
        async with engine.connect() as conn:
            print("✅ [PostgreSQL]: 连接成功！(关系型元数据库已就绪)")
        await engine.dispose()
    except Exception as e:
        print(f"❌ [PostgreSQL]: 失败! 报错信息: {e}")

    # 2. 检查 Qdrant 向量数据库
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        collections = client.get_collections()
        print("✅ [Qdrant]: 连接成功！(向量检索引擎已就绪)")
    except Exception as e:
        print(f"❌ [Qdrant]: 失败! 报错信息: {e}")

    # 3. 检查 Redis 缓存队列
    try:
        redis_url = os.getenv("REDIS_URL")
        r = redis.from_url(redis_url)
        await r.ping()
        print("✅ [Redis]: 连接成功！(异步任务缓存已就绪)")
        await r.close()
    except Exception as e:
        print(f"❌ [Redis]: 失败! 报错信息: {e}")
        
    print("="*30 + "\n🏁 体检结束！")

if __name__ == "__main__":
    # 运行异步检查程序
    asyncio.run(check_health())