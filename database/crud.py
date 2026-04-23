"""
数据库 CRUD 操作封装
"""
import uuid
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Paper, ParseResult, ChatSession, ChatMessage, ParseStatus


# ══════════════════════════════════════════════════════════════
#  Paper CRUD
# ══════════════════════════════════════════════════════════════

async def create_paper(db: AsyncSession, **kwargs) -> Paper:
    paper = Paper(**kwargs)
    db.add(paper)
    await db.flush()
    await db.refresh(paper)
    return paper


async def get_paper(db: AsyncSession, paper_id: uuid.UUID) -> Optional[Paper]:
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    return result.scalar_one_or_none()


async def get_paper_by_celery_task(db: AsyncSession, task_id: str) -> Optional[Paper]:
    result = await db.execute(
        select(Paper).where(Paper.celery_task_id == task_id)
    )
    return result.scalar_one_or_none()


async def list_papers(
    db: AsyncSession,
    domain: Optional[str] = None,
    status: Optional[ParseStatus] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Paper]:
    query = select(Paper).order_by(Paper.created_at.desc()).limit(limit).offset(offset)
    if domain:
        query = query.where(Paper.domain == domain)
    if status:
        query = query.where(Paper.status == status)
    result = await db.execute(query)
    return list(result.scalars().all())


async def update_paper_status(
    db: AsyncSession,
    paper_id: uuid.UUID,
    status: ParseStatus,
    error_message: Optional[str] = None,
    **kwargs,
) -> None:
    values = {"status": status, **kwargs}
    if error_message:
        values["error_message"] = error_message
    await db.execute(
        update(Paper).where(Paper.id == paper_id).values(**values)
    )


async def get_all_domains(db: AsyncSession) -> list[str]:
    """获取所有已有领域"""
    result = await db.execute(
        select(Paper.domain).distinct().where(Paper.status == ParseStatus.COMPLETED)
    )
    return [row[0] for row in result.fetchall()]


# ══════════════════════════════════════════════════════════════
#  ParseResult CRUD
# ══════════════════════════════════════════════════════════════

async def create_parse_result(db: AsyncSession, **kwargs) -> ParseResult:
    pr = ParseResult(**kwargs)
    db.add(pr)
    await db.flush()
    return pr


# ══════════════════════════════════════════════════════════════
#  ChatSession CRUD
# ══════════════════════════════════════════════════════════════

async def create_chat_session(db: AsyncSession, title: Optional[str] = None) -> ChatSession:
    session = ChatSession(title=title)
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


async def add_chat_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: str,
    content: str,
    **kwargs,
) -> ChatMessage:
    msg = ChatMessage(session_id=session_id, role=role, content=content, **kwargs)
    db.add(msg)
    await db.flush()
    return msg


async def get_session_messages(
    db: AsyncSession, session_id: uuid.UUID, limit: int = 20
) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
    )
    return list(result.scalars().all())
