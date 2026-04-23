"""
BGE-M3 Embedding + Qdrant 向量化入库
支持稠密向量（Dense）+ 稀疏向量（Sparse）双路
"""
import uuid
from typing import Optional

from loguru import logger
from qdrant_client.models import PointStruct

from core.config import settings
from core.indexing.chunker import TextChunk
from database.qdrant_client import qdrant_manager


class BGEEmbedder:
    """BGE-M3 向量化，单例模式（模型加载耗时，只加载一次）"""

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                settings.EMBEDDING_MODEL_PATH,
                use_fp16=True,          # 半精度，节省显存
                device=settings.EMBEDDING_DEVICE,
            )
            logger.info(f"BGE-M3 模型加载完成: {settings.EMBEDDING_MODEL_PATH}")
        except ImportError:
            logger.error("FlagEmbedding 未安装: pip install FlagEmbedding")
            raise

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        return_sparse: bool = True,
    ) -> list[dict]:
        """
        批量编码文本
        
        Returns: list of {
            "dense": list[float],          # 稠密向量（1024维）
            "sparse": {"indices": [...], "values": [...]}  # 稀疏向量
        }
        """
        self._load_model()
        
        bs = batch_size or settings.EMBEDDING_BATCH_SIZE
        results = []
        
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            
            output = self._model.encode(
                batch,
                batch_size=bs,
                max_length=8192,           # BGE-M3 支持 8192 token
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )
            
            dense_vecs = output["dense_vecs"].tolist()
            
            for j, dense in enumerate(dense_vecs):
                item = {"dense": dense}
                
                if return_sparse and "lexical_weights" in output:
                    # 稀疏向量：词汇权重字典 → indices/values
                    lex_weights = output["lexical_weights"][j]
                    if lex_weights:
                        indices = [int(k) for k in lex_weights.keys()]
                        values = [float(v) for v in lex_weights.values()]
                    else:
                        indices, values = [], []
                    item["sparse"] = {"indices": indices, "values": values}
                
                results.append(item)
        
        return results


# 全局单例
embedder = BGEEmbedder()


async def index_chunks_to_qdrant(
    parent_chunks: list[TextChunk],
    child_chunks: list[TextChunk],
    domain: str,
) -> int:
    """
    将子块向量化并存入 Qdrant
    父块信息作为 payload 存储，用于父子块检索时的内容扩展
    
    Returns: 成功写入的向量数量
    """
    if not child_chunks:
        logger.warning("没有子块需要向量化")
        return 0
    
    # ── 1. 构建父块查找表 ────────────────────────────────────
    parent_map = {chunk.id: chunk for chunk in parent_chunks}
    
    # ── 2. 批量 Embedding ─────────────────────────────────────
    texts = [chunk.text for chunk in child_chunks]
    logger.info(f"开始向量化 {len(texts)} 个子块...")
    
    embeddings = embedder.encode(texts, return_sparse=True)
    
    # ── 3. 构建 Qdrant PointStruct ────────────────────────────
    points = []
    for chunk, emb in zip(child_chunks, embeddings):
        parent = parent_map.get(chunk.parent_id)
        
        payload = {
            "text": chunk.text,                          # 子块文本
            "parent_text": parent.text if parent else chunk.text,  # 父块文本（LLM 用）
            "paper_id": chunk.paper_id,
            "domain": domain,
            "chunk_type": chunk.chunk_type,
            "parent_id": chunk.parent_id,
            "position": chunk.position,
            **chunk.metadata,
        }
        
        point = PointStruct(
            id=chunk.id,
            vector={
                "dense": emb["dense"],
                **({"sparse": emb["sparse"]} if "sparse" in emb else {}),
            },
            payload=payload,
        )
        points.append(point)
    
    # ── 4. 批量写入 Qdrant ────────────────────────────────────
    batch_size = 100
    total_written = 0
    
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await qdrant_manager.upsert_points(domain, batch)
        total_written += len(batch)
        logger.debug(f"写入进度: {total_written}/{len(points)}")
    
    logger.info(f"向量化完成，共写入 {total_written} 个向量点")
    return total_written
