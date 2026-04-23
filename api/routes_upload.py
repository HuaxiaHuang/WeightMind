"""
FastAPI 路由：文件上传 + 任务状态查询
"""
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from database.crud import create_paper, get_paper, list_papers, get_all_domains
from database.models import ParseStatus
from database.session import get_db
from worker.tasks import process_paper

router = APIRouter()


# ── 响应模型 ────────────────────────────────────────────────

class UploadResponse(BaseModel):
    paper_id: str
    task_id: str
    filename: str
    message: str


class TaskStatusResponse(BaseModel):
    paper_id: str
    status: str
    progress_percent: Optional[int] = None
    progress_message: Optional[str] = None
    domain: Optional[str] = None
    title: Optional[str] = None
    error_message: Optional[str] = None


class PaperSummary(BaseModel):
    id: str
    title: Optional[str]
    domain: str
    status: str
    keywords: Optional[list[str]]
    journal_or_venue: Optional[str]
    year: Optional[int]
    chunk_count: int
    created_at: str


# ── 上传接口 ────────────────────────────────────────────────

@router.post("/paper", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_paper(
    file: UploadFile = File(..., description="PDF 论文文件"),
    db: AsyncSession = Depends(get_db),
):
    """
    上传 PDF 论文
    - 文件保存到临时目录
    - 创建论文记录（状态: PENDING）
    - 触发 Celery 异步处理任务
    - 立即返回 task_id（不等待处理完成）
    """
    # 校验文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="只支持 PDF 文件",
        )
    
    # 校验文件大小
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件大小超出限制（最大 {settings.MAX_UPLOAD_SIZE_MB} MB）",
        )
    
    # 保存到临时目录
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = settings.TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
    temp_path.write_bytes(content)
    
    # 创建 Paper 数据库记录
    paper = await create_paper(
        db,
        original_filename=file.filename,
        pdf_path=str(temp_path),
        domain="pending",  # 待分类
        status=ParseStatus.PENDING,
    )
    
    # 触发 Celery 任务
    task = process_paper.delay(str(paper.id), str(temp_path))
    
    # 回写 Celery task_id
    paper.celery_task_id = task.id
    await db.flush()
    
    return UploadResponse(
        paper_id=str(paper.id),
        task_id=task.id,
        filename=file.filename,
        message="论文已上传，正在后台处理...",
    )


# ── 任务状态查询 ─────────────────────────────────────────────

@router.get("/status/{paper_id}", response_model=TaskStatusResponse)
async def get_task_status(
    paper_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    查询论文处理状态
    前端可每隔 2 秒轮询此接口更新进度条
    """
    try:
        pid = uuid.UUID(paper_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的 paper_id 格式")
    
    paper = await get_paper(db, pid)
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    
    # 从 Celery 获取进度信息
    progress_percent = None
    progress_message = None
    
    if paper.celery_task_id:
        from worker.celery_app import celery_app
        task_result = celery_app.AsyncResult(paper.celery_task_id)
        
        if task_result.state == "PROGRESS":
            meta = task_result.info or {}
            progress_percent = meta.get("percent")
            progress_message = meta.get("message")
        elif task_result.state == "SUCCESS":
            progress_percent = 100
            progress_message = "处理完成"
    
    return TaskStatusResponse(
        paper_id=str(paper.id),
        status=paper.status.value,
        progress_percent=progress_percent,
        progress_message=progress_message,
        domain=paper.domain if paper.domain != "pending" else None,
        title=paper.title,
        error_message=paper.error_message,
    )


# ── 获取所有已有领域 ──────────────────────────────────────────

@router.get("/domains", response_model=list[str])
async def get_domains(db: AsyncSession = Depends(get_db)):
    """获取已处理完成的所有研究领域"""
    return await get_all_domains(db)
