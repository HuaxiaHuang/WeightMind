"""
Celery 异步任务定义 - 终极云端双擎路由版 (严格类型转换版)
功能：PyMuPDF极速读首页 -> LLM初步提取 -> Crossref官方API精准校对 -> 终极去重 -> TextIn/LlamaParse 云端解析 -> 落盘与向量化
"""
import sys
import os
import hashlib
import asyncio
import json
import uuid
import re
import shutil
import requests
import httpx     
import fitz      
from datetime import datetime
from pathlib import Path
from llama_cloud import LlamaCloud  

# ── 核心数据库组件导入 (放在顶部确保全局可用) ──
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

# ── 路径预处理 ──────────────
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from loguru import logger
from worker.celery_app import celery_app
from core.config import settings

def calculate_file_hash(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def _fetch_crossref_sync(doi: str, title: str) -> dict:
    email = "18316743129@163.com" 
    headers = {"User-Agent": f"SciRAG-Agent/1.0 (mailto:{email})"}
    try:
        if doi and len(doi) > 5:
            url = f"https://api.crossref.org/works/{doi}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200: return resp.json().get("message", {})
        if title and len(title) > 5:
            url = "https://api.crossref.org/works"
            params = {"query.bibliographic": title, "rows": 1}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                items = resp.json().get("message", {}).get("items", [])
                if items: return items[0]
    except Exception as e:
        logger.warning(f"⚠️ Crossref API 请求失败: {e}")
    return {}

async def _parse_with_textin(pdf_path: Path) -> dict:
    app_id = getattr(settings, "TEXTIN_APP_ID", "")
    secret_code = getattr(settings, "TEXTIN_SECRET_CODE", "")
    url = "https://api.textin.com/ai/service/v1/pdf_to_markdown"
    headers = {"x-ti-app-id": app_id, "x-ti-secret-code": secret_code, "Content-Type": "application/octet-stream"}
    params = {"table_flavor": "md", "apply_document_tree": "1", "markdown_details": "1", "apply_image_analysis": "1", "formula_level": "0"}
    try:
        with open(pdf_path, "rb") as f: file_data = f.read()
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, params=params, content=file_data, timeout=180)
            resp.raise_for_status()
            md_text = resp.json().get("result", {}).get("markdown", "")
            return {"success": True, "text": md_text, "source": "TextIn"}
    except Exception as e:
        logger.error(f"❌ TextIn 失败: {e}")
        return {"success": False, "text": "", "error": str(e), "source": "TextIn"}

async def _parse_with_llamaparse(pdf_path: Path) -> dict:
    def _run_sync():
        api_key = getattr(settings, "LLAMA_CLOUD_API_KEY", "")
        os.environ["LLAMA_CLOUD_API_KEY"] = api_key
        client = LlamaCloud(base_url="https://api.cloud.eu.llamaindex.ai") 
        file = client.files.create(file=str(pdf_path), purpose="parse")
        result = client.parsing.parse(file_id=file.id, tier="agentic", expand=["markdown"])
        md = "\n\n".join([page.markdown for page in result.markdown.pages])
        return {"success": True, "text": md, "source": "LlamaParse"}
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _run_sync)
    except Exception as e:
        logger.error(f"❌ LlamaParse 失败: {e}")
        return {"success": False, "text": "", "error": str(e), "source": "LlamaParse"}

# ── 核心 Celery 任务 ──────────────
@celery_app.task(bind=True, name="worker.tasks.process_paper", max_retries=2, default_retry_delay=60)
def process_paper(self, paper_id: str, pdf_path: str):
    logger.info(f"🚀 任务启动: paper_id={paper_id}")
    task_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(task_loop)
    try:
        return task_loop.run_until_complete(_process_paper_async(self, paper_id, pdf_path))
    except Exception as exc:
        logger.error(f"❌ 任务重试触发: {exc}")
        task_loop.run_until_complete(_update_status(paper_id, "failed", str(exc)))
        raise self.retry(exc=exc)
    finally:
        task_loop.close()

async def _process_paper_async(task, paper_id: str, pdf_path: str):
    task_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    TaskSessionLocal = async_sessionmaker(task_engine, expire_on_commit=False)

    from database.crud import update_paper_status, create_parse_result
    from database.models import ParseStatus, Paper
    from core.llm.client import llm_client
    from core.indexing.chunker import create_parent_child_chunks
    from core.indexing.embedder import index_chunks_to_qdrant
    from sqlalchemy import select, or_

    pdf = Path(pdf_path)
    if not pdf.exists(): raise FileNotFoundError(f"文件不存在: {pdf_path}")
    current_loop = asyncio.get_running_loop()

    try:
        file_hash = await current_loop.run_in_executor(None, calculate_file_hash, pdf)
        async with TaskSessionLocal() as db:
            existing_paper = (await db.execute(select(Paper).where(Paper.file_hash == file_hash, Paper.status == "COMPLETED"))).scalars().first()
            if existing_paper:
                logger.info(f"✨ Hash 匹配，复用数据")
                return await _reuse_existing_data(paper_id, existing_paper, "Hash Match")

        _update_progress(task, 10, "正在提取元数据...")
        doc = await current_loop.run_in_executor(None, fitz.open, pdf)
        first_page_text = doc[0].get_text("text")[:2500] 
        doc.close()

        smart_meta_prompt = f"Extract paper metadata as JSON (title, authors, journal, year, doi, language) from: {first_page_text}"
        meta_json_str = await llm_client.complete(smart_meta_prompt, json_mode=True) or "{}"
        accurate_meta = json.loads(meta_json_str)
        
        # 🛡️ 类型强制清洗防线：对抗大模型胡说八道
        try:
            accurate_meta["year"] = int(accurate_meta.get("year")) if accurate_meta.get("year") else None
        except (ValueError, TypeError):
            accurate_meta["year"] = None
            
        if not isinstance(accurate_meta.get("authors"), list):
            accurate_meta["authors"] = []

        cr_data = await current_loop.run_in_executor(None, _fetch_crossref_sync, accurate_meta.get("doi"), accurate_meta.get("title"))
        if cr_data:
            if cr_data.get("title"): accurate_meta["title"] = cr_data["title"][0]
            if cr_data.get("DOI"): accurate_meta["doi"] = cr_data["DOI"]

        final_title = str(accurate_meta.get("title", "")).strip()
        final_doi = str(accurate_meta.get("doi", "")).strip()
        blacklist = {"", "none", "null", "unknown", "abstract"}

        await _update_status(paper_id, "parsing")
        PARSE_MODE = getattr(settings, "PARSE_MODE", "eco") 
        res = await _parse_with_textin(pdf)
        if not res["success"]: res = await _parse_with_llamaparse(pdf)
        merged_text, source_used = res["text"], res["source"]

        if not merged_text: raise ValueError("所有解析引擎均失效")

        _update_progress(task, 60, "领域划分与安全落盘...")
        domain_prompt = f"Please classify the scientific domain of this text and return the result in **json** format with a key named 'domain'. Text: {merged_text[:1000]}"
        domain_str = await llm_client.complete(domain_prompt, json_mode=True) or '{"domain": "other"}'
        domain = json.loads(domain_str).get("domain", "other").replace(" ", "_").lower()

        clean_title = re.sub(r'\s+', ' ', final_title if final_title.lower() not in blacklist else "Untitled").strip()
        safe_title = re.sub(r'[\\/*?:"<>|]', "", clean_title)[:50].strip() or "Paper"
        
        base_filename = f"{safe_title}_{paper_id[:8]}"
        domain_dir = settings.RAW_DATA_DIR / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        target_pdf, target_md = domain_dir / f"{base_filename}.pdf", domain_dir / f"{base_filename}.md"
        
        await current_loop.run_in_executor(None, shutil.copy2, pdf, target_pdf)
        def write_md():
            with open(target_md, "w", encoding="utf-8") as f: f.write(merged_text)
        await current_loop.run_in_executor(None, write_md)

        _update_progress(task, 85, "写入向量库...")
        p_chunks, c_chunks = create_parent_child_chunks(merged_text, paper_id, domain, accurate_meta)
        chunk_count = await index_chunks_to_qdrant(p_chunks, c_chunks, domain)

        async with TaskSessionLocal() as db:
            from database.models import ParseResult
            from sqlalchemy import delete
            await db.execute(delete(ParseResult).where(ParseResult.paper_id == uuid.UUID(paper_id)))
            await create_parse_result(db, paper_id=uuid.UUID(paper_id), tool_name=source_used, raw_text=merged_text[:50000], success=1)
            await update_paper_status(
                db, paper_id=uuid.UUID(paper_id), status=ParseStatus.COMPLETED,
                title=final_title, authors=accurate_meta.get("authors"), 
                journal_or_venue=accurate_meta.get("journal"), year=accurate_meta.get("year"),
                doi=final_doi if final_doi else None, domain=domain, 
                pdf_path=str(target_pdf), extracted_text_path=str(target_md),
                file_hash=file_hash, chunk_count=chunk_count, indexed_at=datetime.utcnow()
            )
            await db.commit()

        _update_progress(task, 100, f"✅ 完成 ({source_used})")
        return {"status": "success", "paper_id": paper_id}
    finally:
        await task_engine.dispose()

async def _reuse_existing_data(new_paper_id: str, source_paper, reason: str):
    task_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    TaskSessionLocal = async_sessionmaker(task_engine, expire_on_commit=False)
    from database.crud import update_paper_status
    from database.models import ParseStatus
    try:
        async with TaskSessionLocal() as db:
            await update_paper_status(
                db, paper_id=uuid.UUID(new_paper_id), status=ParseStatus.COMPLETED,
                title=source_paper.title, authors=source_paper.authors,
                journal_or_venue=source_paper.journal_or_venue, year=source_paper.year,
                domain=source_paper.domain, pdf_path=source_paper.pdf_path, 
                extracted_text_path=source_paper.extracted_text_path,
                chunk_count=source_paper.chunk_count, file_hash=source_paper.file_hash,
                indexed_at=datetime.utcnow()
            )
            await db.commit()
        return {"paper_id": new_paper_id, "reused": True}
    finally:
        await task_engine.dispose()

async def _update_status(paper_id: str, status_str: str, error: str = None):
    task_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    TaskSessionLocal = async_sessionmaker(task_engine, expire_on_commit=False)
    from database.crud import update_paper_status
    from database.models import ParseStatus
    status_map = {"parsing": ParseStatus.PARSING, "failed": ParseStatus.FAILED}
    try:
        async with TaskSessionLocal() as db:
            await update_paper_status(db, paper_id=uuid.UUID(paper_id), status=status_map.get(status_str, ParseStatus.FAILED), error_message=error)
            await db.commit()
    finally:
        await task_engine.dispose()

def _update_progress(task, percent: int, message: str):
    task.update_state(state="PROGRESS", meta={"percent": percent, "message": message})