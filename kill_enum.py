import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from core.config import settings
from database.models import Base

async def hard_reset_database():
    print("🔥 准备进行最深层次的数据库格式化...")
    # 必须使用 AUTOCOMMIT 才能删除底层类型
    engine = create_async_engine(settings.DATABASE_URL, isolation_level="AUTOCOMMIT")
    
    async with engine.begin() as conn:
        # 1. 先删掉所有的表
        await conn.run_sync(Base.metadata.drop_all)
        print("💥 表已摧毁。")
        
    async with engine.connect() as conn:
        # 2. 核心操作：强制删除底层的遗留枚举类型！
        try:
            await conn.execute(text("DROP TYPE IF EXISTS parse_tool CASCADE;"))
            print("💀 遗留的 'parse_tool' 门禁类型已被连根拔起！")
        except Exception as e:
            print(f"⚠️ 提示 (可忽略): {e}")
            
    async with engine.begin() as conn:
        # 3. 按照你最新（带有 String 字段）的 models.py 重新建表
        await conn.run_sync(Base.metadata.create_all)
        print("🌱 新表已完美重建！没有任何旧规则残留。")
        
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(hard_reset_database())