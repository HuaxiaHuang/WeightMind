"""
FastAPI 接口测试（使用 httpx 测试客户端，无需真实启动服务）
"""
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 模拟环境（测试时不真正连接数据库） ───────────────────────

@pytest.fixture(scope="session", autouse=True)
def mock_settings(monkeypatch=None):
    """使用 SQLite 内存数据库代替 PostgreSQL（测试用）"""
    import os
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    os.environ.setdefault("LLM_API_KEY", "test-key")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
    os.environ.setdefault("CELERY_BROKER_URL", "memory://")
    os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")


@pytest_asyncio.fixture
async def client():
    """创建测试客户端"""
    from main import app
    from database.session import init_db

    # 使用内存 SQLite 初始化表
    await init_db()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


# ── 健康检查 ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ── 论文列表（空库） ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_papers_empty(client):
    response = await client.get("/papers/")
    assert response.status_code == 200
    assert response.json() == []


# ── 上传接口（非 PDF 文件应返回 400） ──────────────────────────

@pytest.mark.asyncio
async def test_upload_invalid_file_type(client):
    response = await client.post(
        "/upload/paper",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


# ── 对话接口（空知识库时应返回合理响应） ───────────────────────

@pytest.mark.asyncio
async def test_chat_ask_empty_kb(client):
    """空知识库时，接口不应崩溃"""
    response = await client.post(
        "/chat/ask",
        json={"query": "什么是 Transformer？", "stream": False},
    )
    # 空知识库时可能无法回答，但不应 500
    assert response.status_code in (200, 503)


# ── 获取领域列表（空库时应返回空数组） ─────────────────────────

@pytest.mark.asyncio
async def test_get_domains_empty(client):
    response = await client.get("/upload/domains")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
