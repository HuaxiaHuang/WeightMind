"""
FastAPI 路由：论文元数据 CRUD
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.crud import get_paper, list_papers
from database.session import get_db

router = APIRouter()


class PaperDetail(BaseModel):
    id: str
    title: Optional[str]
    authors: Optional[list]
    abstract_summary_zh: Optional[str]
    journal_or_venue: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    domain: str
    subdomain: Optional[str]
    keywords: Optional[list]
    methodology: Optional[str]
    status: str
    chunk_count: int
    parse_quality_score: Optional[float]
    classification_confidence: Optional[float]
    created_at: str
    indexed_at: Optional[str]
    original_filename: str


@router.get("/", response_model=list[PaperDetail])
async def list_all_papers(
    domain: Optional[str] = Query(None, description="按领域过滤"),
    status: Optional[str] = Query(None, description="按状态过滤"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
):
    """获取论文列表"""
    from database.models import ParseStatus
    
    status_enum = None
    if status:
        try:
            status_enum = ParseStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效状态: {status}")
    
    papers = await list_papers(db, domain=domain, status=status_enum, limit=limit, offset=offset)
    
    return [
        PaperDetail(
            id=str(p.id),
            title=p.title,
            authors=p.authors,
            abstract_summary_zh=p.abstract_summary_zh,
            journal_or_venue=p.journal_or_venue,
            year=p.year,
            doi=p.doi,
            domain=p.domain,
            subdomain=p.subdomain,
            keywords=p.keywords,
            methodology=p.methodology,
            status=p.status.value,
            chunk_count=p.chunk_count or 0,
            parse_quality_score=p.parse_quality_score,
            classification_confidence=p.classification_confidence,
            created_at=p.created_at.isoformat(),
            indexed_at=p.indexed_at.isoformat() if p.indexed_at else None,
            original_filename=p.original_filename,
        )
        for p in papers
    ]


@router.get("/{paper_id}", response_model=PaperDetail)
async def get_paper_detail(
    paper_id: str,
    db: AsyncSession = Depends(get_db),
):
    """获取单篇论文详情"""
    try:
        pid = uuid.UUID(paper_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的 paper_id")
    
    paper = await get_paper(db, pid)
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    
    return PaperDetail(
        id=str(paper.id),
        title=paper.title,
        authors=paper.authors,
        abstract_summary_zh=paper.abstract_summary_zh,
        journal_or_venue=paper.journal_or_venue,
        year=paper.year,
        doi=paper.doi,
        domain=paper.domain,
        subdomain=paper.subdomain,
        keywords=paper.keywords,
        methodology=paper.methodology,
        status=paper.status.value,
        chunk_count=paper.chunk_count or 0,
        parse_quality_score=paper.parse_quality_score,
        classification_confidence=paper.classification_confidence,
        created_at=paper.created_at.isoformat(),
        indexed_at=paper.indexed_at.isoformat() if paper.indexed_at else None,
        original_filename=paper.original_filename,
    )
