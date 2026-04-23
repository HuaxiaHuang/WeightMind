"""
SQLAlchemy ORM 模型定义
论文元数据的完整数据库结构
"""
import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class ParseStatus(str, PyEnum):
    PENDING = "pending"          # 等待处理
    PARSING = "parsing"          # 解析中
    CLASSIFYING = "classifying"  # 分类中
    INDEXING = "indexing"        # 向量化中
    COMPLETED = "completed"      # 完成
    FAILED = "failed"            # 失败


class Paper(Base):
    """论文主表 — 核心元数据"""
    __tablename__ = "papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # ── 基础信息 ────────────────────────────────────────────
    title = Column(String(1000), nullable=True, comment="论文标题")
    authors = Column(JSON, nullable=True, comment="作者列表 ['author1', 'author2']")
    abstract = Column(Text, nullable=True, comment="英文摘要")
    abstract_summary_zh = Column(Text, nullable=True, comment="中文摘要（LLM生成）")
    file_hash = Column(String(64), index=True, nullable=True)
    
    # ── 发表信息 ────────────────────────────────────────────
    journal_or_venue = Column(String(500), nullable=True, comment="期刊或会议名称")
    year = Column(Integer, nullable=True, comment="发表年份")
    doi = Column(String(200), nullable=True, unique=True, comment="DOI")
    url = Column(String(1000), nullable=True, comment="论文链接")
    institution = Column(JSON, nullable=True, comment="机构列表")
    
    # ── 领域分类 ────────────────────────────────────────────
    domain = Column(String(100), nullable=False, index=True, comment="一级领域")
    subdomain = Column(String(200), nullable=True, comment="细分领域")
    keywords = Column(JSON, nullable=True, comment="关键词列表")
    methodology = Column(String(500), nullable=True, comment="主要方法论")
    
    # ── 文件路径 ────────────────────────────────────────────
    original_filename = Column(String(500), nullable=False, comment="原始文件名")
    pdf_path = Column(String(1000), nullable=True, comment="PDF存储路径")
    extracted_text_path = Column(String(1000), nullable=True, comment="提取文本路径")
    
    # ── 解析质量 ────────────────────────────────────────────
    parse_quality_score = Column(Float, nullable=True, comment="解析质量分数 0-1")
    classification_confidence = Column(Float, nullable=True, comment="领域分类置信度")
    
    # ── 向量化状态 ──────────────────────────────────────────
    status = Column(
        Enum(ParseStatus), 
        default=ParseStatus.PENDING, 
        nullable=False, 
        index=True,
        comment="处理状态"
    )
    celery_task_id = Column(String(200), nullable=True, comment="Celery任务ID")
    error_message = Column(Text, nullable=True, comment="失败原因")
    
    # ── 向量库引用 ──────────────────────────────────────────
    qdrant_collection = Column(String(200), nullable=True, comment="对应的Qdrant Collection名")
    chunk_count = Column(Integer, default=0, comment="向量化的chunk数量")
    
    # ── 时间戳 ──────────────────────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True, comment="完成向量化的时间")
    
    # ── 关联 ────────────────────────────────────────────────
    parse_results = relationship("ParseResult", back_populates="paper", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Paper id={self.id} title={self.title!r} domain={self.domain}>"


class ParseResult(Base):
    """三路解析结果表 — 存储每个解析工具的原始输出"""
    __tablename__ = "parse_results"
    __table_args__ = (
        UniqueConstraint("paper_id", "tool_name", name="uq_paper_tool"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False, index=True)
    
    tool_name = Column(
        String(50),
        nullable=False,
        comment="解析工具名称 (例如 TextIn, LlamaParse)"
    )
    raw_text = Column(Text, nullable=True, comment="原始解析文本")
    text_path = Column(String(1000), nullable=True, comment="文本文件路径（文本过长时）")
    char_count = Column(Integer, default=0, comment="字符数")
    parse_time_seconds = Column(Float, nullable=True, comment="解析耗时")
    success = Column(Integer, default=1, comment="是否成功 1/0")
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    paper = relationship("Paper", back_populates="parse_results")


class ChatSession(Base):
    """对话会话表"""
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=True, comment="会话标题（自动提取）")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """对话消息表"""
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    
    role = Column(Enum("user", "assistant", name="message_role"), nullable=False)
    content = Column(Text, nullable=False)
    
    # RAG 相关信息
    retrieved_paper_ids = Column(JSON, nullable=True, comment="检索到的论文ID列表")
    qdrant_collections_searched = Column(JSON, nullable=True, comment="搜索了哪些Collection")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("ChatSession", back_populates="messages")
