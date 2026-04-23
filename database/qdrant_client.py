"""
Qdrant 向量数据库客户端封装
支持按领域 Collection 动态路由
"""
from typing import Optional

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from core.config import settings

# ── Qdrant Collection 名称规范 ──────────────────────────────
# 领域名称统一用小写 + 下划线，与 PostgreSQL 一致
COLLECTION_PREFIX = "sci_rag_"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = 1024  # BGE-M3 输出维度


def domain_to_collection(domain: str) -> str:
    """领域名 → Qdrant Collection 名"""
    clean = domain.lower().replace(" ", "_").replace("-", "_")
    return f"{COLLECTION_PREFIX}{clean}"


def collection_to_domain(collection: str) -> str:
    """Qdrant Collection 名 → 领域名"""
    return collection.removeprefix(COLLECTION_PREFIX)


class QdrantManager:
    """Qdrant 操作封装，单例模式"""

    def __init__(self):
        self._client: Optional[AsyncQdrantClient] = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=30,
            )
        return self._client

    async def ensure_collection(self, domain: str) -> str:
        """确保该领域的 Collection 存在，不存在则创建"""
        collection_name = domain_to_collection(domain)
        
        existing = await self.client.collection_exists(collection_name)
        if not existing:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=DENSE_DIM,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams()
                },
            )
            logger.info(f"创建 Qdrant Collection: {collection_name}")
        
        return collection_name

    async def list_collections(self) -> list[str]:
        """列出所有科研 RAG 相关的 Collection"""
        result = await self.client.get_collections()
        return [
            c.name for c in result.collections
            if c.name.startswith(COLLECTION_PREFIX)
        ]

    async def upsert_points(
        self,
        domain: str,
        points: list[PointStruct],
    ) -> None:
        """批量写入向量点"""
        collection_name = await self.ensure_collection(domain)
        await self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        logger.debug(f"写入 {len(points)} 个向量点到 {collection_name}")

    async def hybrid_search(
        self,
        domain: str,
        dense_vector: list[float],
        sparse_vector: Optional[dict] = None,
        top_k: int = 10,
        paper_id_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        混合检索：稠密向量 + 稀疏向量
        返回格式: [{"id": ..., "score": ..., "payload": {...}}]
        """
        collection_name = domain_to_collection(domain)
        
        # 检查 Collection 是否存在
        if not await self.client.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} 不存在，跳过")
            return []

        # 构建过滤条件（可选：只在特定论文内搜索）
        query_filter = None
        if paper_id_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=paper_id_filter)
                )]
            )

        # 稠密向量检索
        dense_results = await self.client.search(
            collection_name=collection_name,
            query_vector=(DENSE_VECTOR_NAME, dense_vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        # 如果有稀疏向量，做 RRF 融合
        if sparse_vector:
            sparse_results = await self.client.search(
                collection_name=collection_name,
                query_vector=(
                    SPARSE_VECTOR_NAME,
                    SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"],
                    ),
                ),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            return self._rrf_fusion(dense_results, sparse_results, top_k)

        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload}
            for r in dense_results
        ]

    @staticmethod
    def _rrf_fusion(dense_results, sparse_results, top_k: int, k: int = 60) -> list[dict]:
        """
        Reciprocal Rank Fusion 融合稠密和稀疏检索结果
        """
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, r in enumerate(dense_results):
            pid = str(r.id)
            scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank + 1)
            payloads[pid] = r.payload

        for rank, r in enumerate(sparse_results):
            pid = str(r.id)
            scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank + 1)
            if pid not in payloads:
                payloads[pid] = r.payload

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [
            {"id": pid, "score": scores[pid], "payload": payloads[pid]}
            for pid in sorted_ids[:top_k]
        ]

    async def delete_paper_vectors(self, domain: str, paper_id: str) -> None:
        """删除某篇论文的所有向量"""
        collection_name = domain_to_collection(domain)
        await self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=paper_id)
                )]
            ),
        )


# 单例
qdrant_manager = QdrantManager()
