import asyncio
import requests
from database.session import engine
from database.models import Base

async def reset_pg():
    print("🧹 正在清空 PostgreSQL 数据库...")
    async with engine.begin() as conn:
        # 删除所有表
        await conn.run_sync(Base.metadata.drop_all)
        # 重新创建干净的表
        await conn.run_sync(Base.metadata.create_all)
    print("✅ PostgreSQL (SQLAlchemy) 清空并重建完成！")

def reset_qdrant():
    print("🧹 正在清空 Qdrant 向量数据库...")
    try:
        # 获取所有现存的 Collection
        resp = requests.get("http://localhost:6333/collections", timeout=5)
        if resp.status_code == 200:
            collections = resp.json().get("result", {}).get("collections", [])
            for c in collections:
                c_name = c["name"]
                # 逐个删除 Collection
                del_resp = requests.delete(f"http://localhost:6333/collections/{c_name}")
                if del_resp.status_code == 200:
                    print(f"  - 💥 已彻底删除集合: {c_name}")
        print("✅ Qdrant 向量库清空完成！")
    except Exception as e:
        print(f"❌ 清理 Qdrant 失败 (请确认 Qdrant 已启动): {e}")

if __name__ == "__main__":
    # 运行异步的 PG 清理任务
    asyncio.run(reset_pg())
    # 运行同步的 Qdrant 清理任务
    reset_qdrant()
    print("\n🎉 所有数据库均已完成『洗髓换血』！可以重新启动系统并上传新论文了。")