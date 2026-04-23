"""
检索管道：HyDE + 混合检索 + BGE重排 + 父子块扩展
"""
import json
import logging
from typing import Optional

from loguru import logger

from core.config import settings
from core.indexing.embedder import embedder
from core.llm.client import llm_client
from core.prompts import (
    DOMAIN_ROUTING_PROMPT,
    HYDE_PROMPT,
    QUERY_REWRITE_PROMPT,
    RAG_ANSWER_PROMPT,
    CHAT_WITH_HISTORY_PROMPT,
)
from database.qdrant_client import qdrant_manager


# ══════════════════════════════════════════════════════════════
#  查询重写
# ══════════════════════════════════════════════════════════════

async def rewrite_query(query: str) -> dict:
    """
    将用户问题重写为多个检索变体
    Returns: {"original": str, "rewrites": list[str], "keywords": list[str]}
    """
    try:
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        response = await llm_client.complete(prompt, json_mode=True)
        result = json.loads(response)
        return result
    except Exception as e:
        logger.warning(f"查询重写失败，使用原始查询: {e}")
        return {"original": query, "rewrites": [], "keywords": []}


# ══════════════════════════════════════════════════════════════
#  领域路由
# ══════════════════════════════════════════════════════════════

async def route_to_domain(query: str, available_domains: list[str]) -> dict:
    """
    判断问题属于哪个/哪些领域
    Returns: {"primary_domain": str, "secondary_domains": list, "search_all": bool}
    """
    if not available_domains:
        return {"primary_domain": None, "secondary_domains": [], "search_all": True}
    
    if len(available_domains) == 1:
        return {
            "primary_domain": available_domains[0],
            "secondary_domains": [],
            "search_all": False,
        }
    
    try:
        prompt = DOMAIN_ROUTING_PROMPT.format(
            query=query,
            available_domains="\n".join(f"- {d}" for d in available_domains),
        )
        response = await llm_client.complete(prompt, json_mode=True)
        return json.loads(response)
    except Exception as e:
        logger.warning(f"领域路由失败，搜索所有领域: {e}")
        return {"primary_domain": None, "secondary_domains": [], "search_all": True}


# ══════════════════════════════════════════════════════════════
#  HyDE — 假设文档嵌入
# ══════════════════════════════════════════════════════════════

async def generate_hyde_query(query: str) -> str:
    """
    生成假设性答案文档用于向量检索
    比原始问题的语义空间更接近真实答案
    """
    try:
        prompt = HYDE_PROMPT.format(query=query)
        hypothesis = await llm_client.complete(
            prompt,
            max_tokens=512,
            temperature=0.3,  # 稍高温度，让假设答案更多样
        )
        logger.debug(f"HyDE 假设答案: {hypothesis[:100]}...")
        return hypothesis
    except Exception as e:
        logger.warning(f"HyDE 生成失败，使用原始查询: {e}")
        return query


# ══════════════════════════════════════════════════════════════
#  BGE 重排
# ══════════════════════════════════════════════════════════════

class BGEReranker:
    """BGE 重排模型，对检索结果二次精排"""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                settings.RERANKER_MODEL_PATH,
                use_fp16=True,
                device=settings.EMBEDDING_DEVICE,
            )
            logger.info(f"BGE Reranker 加载完成: {settings.RERANKER_MODEL_PATH}")
        except ImportError:
            logger.error("FlagEmbedding 未安装: pip install FlagEmbedding")
            raise

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        对候选结果重排
        candidates: [{"id": ..., "score": ..., "payload": {...}}, ...]
        Returns: 重排后的 top_k 候选列表
        """
        if not candidates:
            return []
        
        top_k = top_k or settings.TOP_K_RERANK
        self._load()
        
        # 构建 (query, passage) 对
        pairs = [
            (query, c["payload"].get("text", ""))
            for c in candidates
        ]
        
        scores = self._model.compute_score(pairs, normalize=True)
        
        # 附加重排分数
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        
        # 按重排分数降序
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# 全局单例
reranker = BGEReranker()


# ══════════════════════════════════════════════════════════════
#  父子块扩展
# ══════════════════════════════════════════════════════════════

def expand_to_parent_chunks(candidates: list[dict]) -> list[dict]:
    """
    将命中的子块扩展为父块文本（提供更完整的上下文给 LLM）
    如果多个子块属于同一父块，合并为一个
    """
    seen_parents: dict[str, dict] = {}
    
    for candidate in candidates:
        payload = candidate["payload"]
        parent_id = payload.get("parent_id")
        parent_text = payload.get("parent_text", payload.get("text", ""))
        
        if parent_id:
            if parent_id not in seen_parents:
                seen_parents[parent_id] = {
                    "id": candidate["id"],
                    "score": candidate.get("rerank_score", candidate.get("score", 0)),
                    "payload": {
                        **payload,
                        "text": parent_text,   # 使用父块文本
                        "_is_parent_expanded": True,
                    },
                }
            else:
                # 同一父块多个子块命中，取最高分
                existing = seen_parents[parent_id]
                new_score = candidate.get("rerank_score", candidate.get("score", 0))
                if new_score > existing["score"]:
                    existing["score"] = new_score
        else:
            # 没有父块，直接使用
            seen_parents[candidate["id"]] = candidate
    
    return list(seen_parents.values())


# ══════════════════════════════════════════════════════════════
#  完整检索管道
# ══════════════════════════════════════════════════════════════

async def retrieve(
    query: str,
    available_domains: Optional[list[str]] = None,
    top_k_retrieve: Optional[int] = None,
    top_k_rerank: Optional[int] = None,
) -> tuple[list[dict], dict]:
    """
    完整 RAG 检索管道：
    查询重写 → 领域路由 → HyDE → 混合检索 → BGE重排 → 父子块扩展
    """
    top_k_retrieve = top_k_retrieve or settings.TOP_K_RETRIEVE
    top_k_rerank = top_k_rerank or settings.TOP_K_RERANK
    
    pipeline_info = {
        "original_query": query,
        "domains_searched": [],
        "total_candidates": 0,
        "after_rerank": 0,
    }
    
    # ── 1. 查询重写 ────────────────────────────────────────────
    query_data = await rewrite_query(query)
    all_queries = [query] + query_data.get("rewrites", [])
    
    # ── 2. 领域路由 ────────────────────────────────────────────
    domains = available_domains or []
    if not domains:
        collections = await qdrant_manager.list_collections()
        from database.qdrant_client import collection_to_domain
        domains = [collection_to_domain(c) for c in collections]
    
    routing = await route_to_domain(query, domains)
    
    if routing.get("search_all"):
        search_domains = domains
    else:
        search_domains = [routing["primary_domain"]] if routing.get("primary_domain") else []
        search_domains += routing.get("secondary_domains", [])
        search_domains = list(dict.fromkeys(search_domains))
    
    pipeline_info["domains_searched"] = search_domains
    
    if not search_domains:
        logger.warning("没有可搜索的领域")
        return [], pipeline_info
    
    # ── 3. HyDE 生成假设答案 ───────────────────────────────────
    hyde_text = await generate_hyde_query(query)
    
    # ── 4. 编码 HyDE 文本并极致防御格式错误 ─────────────────────
    hyde_embeddings = embedder.encode([hyde_text], return_sparse=True)
    
    raw_dense = hyde_embeddings[0].get("dense")
    # 强制将稠密向量转为标准的 Python float 列表，防止 Pydantic 校验崩溃
    if hasattr(raw_dense, "tolist"):
        dense_vec = raw_dense.tolist()
    else:
        dense_vec = [float(x) for x in raw_dense]
        
    sparse_vec = hyde_embeddings[0].get("sparse")
    
    # ── 5. 混合检索（带智能降级保护） ───────────────────────────
    all_candidates = []
    
    for domain in search_domains:
        try:
            # 正常尝试混合检索
            results = await qdrant_manager.hybrid_search(
                domain=domain,
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                top_k=top_k_retrieve,
            )
            all_candidates.extend(results)
        except Exception as e:
            logger.error(f"⚠️ 领域 {domain} 混合检索发生底层报错 (可能由于 Pydantic 校验稀疏向量引发): {e}")
            logger.info("🛡️ 触发智能降级保护：正在剥离稀疏向量，仅使用稠密向量进行纯语义检索...")
            try:
                # 降级：丢弃引发崩溃的 sparse_vector，只传稠密向量
                results = await qdrant_manager.hybrid_search(
                    domain=domain,
                    dense_vector=dense_vec,
                    sparse_vector=None,  # 强制清除惹祸的稀疏向量
                    top_k=top_k_retrieve,
                )
                all_candidates.extend(results)
                logger.info(f"✅ 领域 {domain} 降级检索成功！")
            except Exception as inner_e:
                logger.error(f"❌ 领域 {domain} 降级检索依然失败: {inner_e}")
    
    pipeline_info["total_candidates"] = len(all_candidates)
    
    if not all_candidates:
        return [], pipeline_info
    
    # ── 6. BGE 重排 ────────────────────────────────────────────
    reranked = reranker.rerank(query, all_candidates, top_k=top_k_rerank)
    pipeline_info["after_rerank"] = len(reranked)
    
    # ── 7. 父子块扩展 ──────────────────────────────────────────
    expanded = expand_to_parent_chunks(reranked)
    
    logger.info(
        f"检索完成: {len(all_candidates)} 候选 → {len(reranked)} 重排 → {len(expanded)} 父块扩展"
    )
    
    return expanded, pipeline_info


# ══════════════════════════════════════════════════════════════
#  RAG 生成 (附带 Token 截断防爆机制)
# ══════════════════════════════════════════════════════════════

def format_context(chunks: list[dict], max_chars: int = 6000) -> str:
    """
    将检索到的 chunks 格式化为 LLM 上下文
    增加 max_chars 截断保护，防止父块过大撑爆 LLM Token 限制
    (注: 中文环境下 6000 字符大约对应 3000~4000 Tokens)
    """
    context_parts = []
    current_length = 0
    
    for i, chunk in enumerate(chunks, 1):
        payload = chunk["payload"]
        paper_title = payload.get("title", "未知论文")
        text = payload.get("text", "")
        score = chunk.get("rerank_score", chunk.get("score", 0))
        section = payload.get("section", "正文或未标明章节")
        
        # 组装当前这篇文献的内容
        part = (
            f"【文献 {i}】\n"
            f"来源论文：《{paper_title}》\n"
            f"所属章节/位置：{section}\n"
            f"相关度：{score:.3f}\n"
            f"原文内容：\n{text}"
        )
        
        # 🧨 熔断机制：如果加上这篇文献超出了安全长度
        if current_length + len(part) > max_chars:
            remaining_space = max_chars - current_length
            # 如果剩余空间还够塞几百字，就截断塞进去
            if remaining_space > 300:
                truncated_part = part[:remaining_space] + "\n...[字数超限，该文献尾部已截断]..."
                context_parts.append(truncated_part)
            else:
                logger.warning(f"触发 Token 保护机制，丢弃了第 {i} 篇及以后的边缘相关文献。")
            
            # 空间已满，直接结束循环
            break
            
        context_parts.append(part)
        current_length += len(part)
    
    return "\n\n====================\n\n".join(context_parts)


async def generate_answer(
    query: str,
    chunks: list[dict],
    chat_history: Optional[list[dict]] = None,
) -> str:
    """
    基于检索结果生成最终回答（非流式）
    """
    context = format_context(chunks)
    
    if chat_history:
        history_str = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in chat_history[-4:]  # 最近 4 轮
        )
        prompt = CHAT_WITH_HISTORY_PROMPT.format(
            chat_history=history_str,
            query=query,
            context=context,
        )
    else:
        prompt = RAG_ANSWER_PROMPT.format(query=query, context=context)
    
    return await llm_client.complete(prompt, max_tokens=4096)


import logging # 如果文件顶部没有引入 logging 或 logger，加上这行

async def stream_answer(
    query: str,
    chunks: list[dict],
    chat_history: Optional[list[dict]] = None,
):
    """
    流式生成回答
    Yields: 文本片段
    """
    context = format_context(chunks)
    
    if chat_history:
        history_str = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in chat_history[-4:]
        )
        system_content = CHAT_WITH_HISTORY_PROMPT.format(
            chat_history=history_str,
            query=query,
            context=context,
        )
    else:
        system_content = RAG_ANSWER_PROMPT.format(query=query, context=context)
    
    messages = [{"role": "user", "content": system_content}]
    
    # ✨ 核心修复：加上 try...except 异常拦截
    try:
        async for token in llm_client.stream_complete(messages):
            yield token
    except Exception as e:
        # 1. 把真实死因打印在后端的黑色终端里，方便你排查
        logging.error(f"🚨 大模型流式生成异常: {str(e)}")
        
        # 2. 把报错包装成友好的提示，作为 token 发给前端，避免前端直接红屏崩溃
        error_msg = (
            f"\n\n> ⚠️ **惟明引擎警报：大模型生成被意外中断**\n"
            f">\n> **底层真实报错：** `{str(e)}`\n"
            f">\n> 诊断建议：通常是因为单次检索的【父片段】太长，导致输入大模型的 Token 数量超限。请尝试清空历史会话，或在配置中调小 `TOP_K_RETRIEVE` 的值。"
        )
        yield error_msg