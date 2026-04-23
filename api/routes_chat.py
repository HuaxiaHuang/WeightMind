"""
FastAPI 路由：对话接口（含流式输出与思考过程透出）
"""
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from loguru import logger

# 引入配置与底层模块
from core.config import settings
from core.indexing.embedder import embedder
from database.qdrant_client import qdrant_manager
from database.crud import (
    create_chat_session,
    add_chat_message,
    get_session_messages,
    get_all_domains,
)
from database.session import get_db

# 从管线中引入细分的执行组件
from core.retrieval.pipeline import (
    retrieve,  # 保留供非流式接口使用
    route_to_domain,
    generate_hyde_query,
    reranker,
    expand_to_parent_chunks,
    stream_answer,
    generate_answer
)

router = APIRouter()

# ── 请求/响应模型 ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # 不传则创建新会话
    stream: bool = True               # 是否流式输出


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict]
    domains_searched: list[str]


# ── 流式对话接口 ──────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    流式对话接口（Server-Sent Events）
    支持实时输出 RAG 思考管线的每一个步骤
    """
    # 获取/创建会话
    if request.session_id:
        try:
            session_id = uuid.UUID(request.session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的 session_id")
    else:
        session = await create_chat_session(db, title=request.query[:50])
        session_id = session.id
        await db.commit()
    
    # 获取历史消息
    history_msgs = await get_session_messages(db, session_id, limit=10)
    chat_history = [{"role": msg.role, "content": msg.content} for msg in history_msgs]
    
    # 获取可用领域
    domains = await get_all_domains(db)
    
    # 保存用户消息
    await add_chat_message(db, session_id=session_id, role="user", content=request.query)
    await db.commit()
    
    # 构建 SSE 生成器 (这是魔法发生的地方)
    async def event_generator():
        # 定义一个发送“思考过程”的快捷器
        def send_thought(msg: str):
            logger.info(msg)  # ✨ 新增这行：让后端终端同步打印思考过程
            return f"data: {json.dumps({'type': 'log', 'content': msg}, ensure_ascii=False)}\n\n"

        yield send_thought("🚀 收到指令，正在唤醒惟明 (WeightMind) 混合检索管线...")

        # ── Step 1: 领域路由分析 ──
        yield send_thought("🧭 正在调用大模型进行跨学科领域路由分析...")
        routing = await route_to_domain(request.query, domains)
        if routing.get("search_all"):
            search_domains = domains
        else:
            search_domains = [routing["primary_domain"]] if routing.get("primary_domain") else []
            search_domains += routing.get("secondary_domains", [])
            search_domains = list(dict.fromkeys(search_domains)) # 去重
            
        if not search_domains:
            search_domains = domains
            yield send_thought("⚠️ 未匹配到特定单一领域，启用全库穿透检索。")
        else:
            yield send_thought(f"🎯 领域锁定：将精准检索 [{', '.join(search_domains)}] 库。")

        # ── Step 2: HyDE 假设生成 ──
        yield send_thought("🧠 正在推演 HyDE (假设性文档嵌入) 以弥合语义鸿沟...")
        hyde_text = await generate_hyde_query(request.query)
        yield send_thought(f"💡 生成假设答案预览：<br><i>\"{hyde_text[:80]}...\"</i>")

        # ── Step 3: 向量编码 ──
        yield send_thought("🔢 正在激活 BGE-M3 模型，将假设内容转化为高维向量矩阵...")
        hyde_embeddings = embedder.encode([hyde_text], return_sparse=True)
        raw_dense = hyde_embeddings[0].get("dense")
        dense_vec = raw_dense.tolist() if hasattr(raw_dense, "tolist") else [float(x) for x in raw_dense]
        sparse_vec = hyde_embeddings[0].get("sparse")

        # ── Step 4: Qdrant 混合检索 ──
        yield send_thought("🔍 正在 Qdrant 中并行执行 Dense(稠密) + Sparse(稀疏) 混合语义检索...")
        all_candidates = []
        for domain in search_domains:
            try:
                results = await qdrant_manager.hybrid_search(
                    domain=domain, dense_vector=dense_vec, sparse_vector=sparse_vec, top_k=settings.TOP_K_RETRIEVE
                )
                all_candidates.extend(results)
            except Exception as e:
                yield send_thought(f"🛡️ 领域 {domain} 触发智能降级，正在执行纯稠密向量检索...")
                try:
                    results = await qdrant_manager.hybrid_search(
                        domain=domain, dense_vector=dense_vec, sparse_vector=None, top_k=settings.TOP_K_RETRIEVE
                    )
                    all_candidates.extend(results)
                except:
                    pass
        
        yield send_thought(f"📦 浅层初筛完成，跨领域共召回 {len(all_candidates)} 个可能相关的文档块。")

        # ── Step 5: BGE Reranker 重排 ──
        yield send_thought("⚖️ 正在调用底层 BGE-Reranker 对候选文档进行极致的交叉注意力重排...")
        reranked = reranker.rerank(request.query, all_candidates, top_k=settings.TOP_K_RERANK)
        yield send_thought(f"✅ 深度重排完成，已精准剥离噪音，保留 Top {len(reranked)} 个核心论点。")

        # ── Step 6: 父子块扩展 ──
        yield send_thought("🔗 正在执行父块溯源，为碎片化片段恢复完整的论文上下文语境...")
        expanded_chunks = expand_to_parent_chunks(reranked)

        # ── Step 7: 检索完毕，推流准备 ──
        yield send_thought("✨ 检索管线全部执行完毕！正在融合知识生成最终科研解答...")
        
        # 修复点 2.1：先发送 Session 等元数据包
        meta = {
            "type": "meta",
            "session_id": str(session_id),
            "domains_searched": search_domains,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        
        # 修复点 2.2：独立发送前端期待的 references 数据包，并补齐 id 字段
        references = {
            "type": "references",
            "content": [
                {
                    "id": i + 1,  # 前端 UI 需要这个序号
                    "title": c["payload"].get("title", "未知论文"),
                    "domain": c["payload"].get("domain", ""),
                    "score": round(c.get("rerank_score", c.get("score", 0)), 3),
                    "text_preview": c["payload"].get("text", "")[:150],
                    "paper_id": c["payload"].get("paper_id")
                }
                for i, c in enumerate(expanded_chunks[:5])
            ]
        }
        yield f"data: {json.dumps(references, ensure_ascii=False)}\n\n"
        
        # 流式输出大模型最终回答
        full_answer = ""
        try:
            async for token in stream_answer(request.query, expanded_chunks, chat_history):
                full_answer += token
                chunk_data = {"type": "token", "content": token}
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 捕获大模型报错，并在后端打印真实死因
            logger.error(f"🚨 调用大模型生成回答时崩溃: {str(e)}")
            error_msg = f"\n\n> ⚠️ 惟明引擎提示：大模型生成被意外中断，底层报错为：`{str(e)}`。可能是单次喂入的论文文本过长导致 Token 超限，请尝试换一个问题或清理历史会话。"
            full_answer += error_msg
            yield f"data: {json.dumps({'type': 'token', 'content': error_msg}, ensure_ascii=False)}\n\n"
        
        # 落库保存大模型的回答
        await add_chat_message(
            db,
            session_id=session_id,
            role="assistant",
            content=full_answer,
            retrieved_paper_ids=[c["payload"].get("paper_id") for c in expanded_chunks],
            qdrant_collections_searched=search_domains,
        )
        await db.commit()
        
        # 结束信号
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
    
    # 将流数据打包返回，关闭缓存避免延迟
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no", 
        },
    )


# ── 非流式对话接口（备用） ─────────────────────────────────────

@router.post("/ask", response_model=ChatResponse)
async def chat_ask(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """非流式对话（一次性返回完整答案）"""
    if request.session_id:
        session_id = uuid.UUID(request.session_id)
    else:
        session = await create_chat_session(db, title=request.query[:50])
        session_id = session.id
        await db.commit()
    
    domains = await get_all_domains(db)
    # 此处依然调用 pipeline.py 中的封装版本
    chunks, pipeline_info = await retrieve(request.query, available_domains=domains)
    
    history_msgs = await get_session_messages(db, session_id, limit=10)
    chat_history = [{"role": m.role, "content": m.content} for m in history_msgs]
    
    answer = await generate_answer(request.query, chunks, chat_history)
    
    await add_chat_message(db, session_id=session_id, role="user", content=request.query)
    await add_chat_message(db, session_id=session_id, role="assistant", content=answer)
    await db.commit()
    
    return ChatResponse(
        answer=answer,
        session_id=str(session_id),
        sources=[
            {
                "title": c["payload"].get("title", ""),
                "score": round(c.get("rerank_score", 0), 3),
            }
            for c in chunks[:5]
        ],
        domains_searched=pipeline_info.get("domains_searched", []),
    )