"""
带结构感知的父子块分割器
子块（128 token）用于向量检索精准定位
父块（512 token）用于给 LLM 提供完整上下文
增加特性：Markdown 章节雷达，实现细粒度溯源 [1, 引言, 第3段]
"""
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from core.config import settings


@dataclass
class TextChunk:
    """文本块数据结构"""
    id: str
    text: str
    paper_id: str
    domain: str
    chunk_type: str          # "parent" | "child"
    parent_id: Optional[str] # 子块指向父块 ID
    position: int            # 在文档中的顺序位置
    metadata: dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """粗略估算 token 数（1 token ≈ 4 字符）"""
        return len(self.text) // 4


def _split_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    按 token 数量分割文本
    用于将过长的单个段落切分为合规的子块
    """
    char_size = chunk_size * 3   # 保守估算
    char_overlap = overlap * 3
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + char_size, text_len)
        
        # 尽量在句子边界截断
        if end < text_len:
            for boundary in ["\n\n", "\n", "。", ". ", "! ", "? "]:
                pos = text.rfind(boundary, start + char_size // 2, end)
                if pos > 0:
                    end = pos + len(boundary)
                    break
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= text_len:
            break
                
        start = max(start + 1, end - char_overlap)
        if start >= text_len:
            break
    
    return chunks


def create_parent_child_chunks(
    text: str,
    paper_id: str,
    domain: str,
    paper_metadata: Optional[dict] = None,
    parent_chunk_size: Optional[int] = None,
    child_chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> tuple[list[TextChunk], list[TextChunk]]:
    """
    创建带有章节结构坐标的父子块
    """
    parent_size_chars = (parent_chunk_size or settings.PARENT_CHUNK_SIZE) * 3
    child_size = child_chunk_size or settings.CHILD_CHUNK_SIZE
    chunk_overlap = overlap or settings.CHUNK_OVERLAP
    
    base_metadata = paper_metadata or {}
    
    parent_chunks: list[TextChunk] = []
    child_chunks: list[TextChunk] = []
    
    # ── 核心改造：按段落拆分，扫描结构 ───────────────────────────
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_section = "正文或未标明章节"
    para_count_in_section = 0
    
    current_parent_id = str(uuid.uuid4())
    current_parent_texts = []
    current_parent_char_count = 0
    
    p_idx = 0
    child_position = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 🎯 魔法雷达：探测 Markdown 标题 (支持 "# 标题" 或 "1. 标题" 或 "1.2 标题")
        header_match = re.match(r'^(#{1,6}\s+|\d+(\.\d+)*\s+)(.+)', para)
        if header_match and len(para) < 200:  # 标题通常不会太长
            current_section = header_match.group(3).strip()
            para_count_in_section = 0
            
            # 将标题也放入父块上下文，帮助大模型理解
            current_parent_texts.append(para)
            current_parent_char_count += len(para)
            continue
            
        para_count_in_section += 1
        
        # 📍 构建精确的位置坐标系
        section_label = f"{current_section} - 第{para_count_in_section}段"
        
        # ── 针对当前段落进行子块切分（防止单段过长） ──
        child_texts = _split_by_tokens(para, child_size, chunk_overlap)
        
        for c_text in child_texts:
            child_id = str(uuid.uuid4())
            child_chunk = TextChunk(
                id=child_id,
                text=c_text,
                paper_id=paper_id,
                domain=domain,
                chunk_type="child",
                parent_id=current_parent_id,
                position=child_position,
                metadata={
                    **base_metadata,
                    "parent_chunk_id": current_parent_id,
                    "section": section_label,  # 🎯 核心注入：把坐标永远刻在 metadata 里！
                },
            )
            child_chunks.append(child_chunk)
            child_position += 1
            
        # ── 累加到父块中 ──
        current_parent_texts.append(para)
        current_parent_char_count += len(para)
        
        # ── 如果当前积累的段落达到了父块的容量上限，封装并开启下一个父块 ──
        if current_parent_char_count >= parent_size_chars:
            parent_text = "\n\n".join(current_parent_texts)
            parent_chunk = TextChunk(
                id=current_parent_id,
                text=parent_text,
                paper_id=paper_id,
                domain=domain,
                chunk_type="parent",
                parent_id=None,
                position=p_idx,
                metadata={
                    **base_metadata,
                    "chunk_index": p_idx,
                    "section": f"{current_section} (部分汇总)", # 父块的范围坐标
                },
            )
            parent_chunks.append(parent_chunk)
            
            # 重置状态，准备迎接下一个父块
            current_parent_id = str(uuid.uuid4())
            current_parent_texts = []
            current_parent_char_count = 0
            p_idx += 1

    # ── 处理末尾遗留的段落 ──
    if current_parent_texts:
        parent_text = "\n\n".join(current_parent_texts)
        parent_chunk = TextChunk(
            id=current_parent_id,
            text=parent_text,
            paper_id=paper_id,
            domain=domain,
            chunk_type="parent",
            parent_id=None,
            position=p_idx,
            metadata={
                **base_metadata,
                "chunk_index": p_idx,
                "section": f"{current_section} (末尾)",
            },
        )
        parent_chunks.append(parent_chunk)

    logger.info(
        f"结构感知切块完成：{len(parent_chunks)} 父块，{len(child_chunks)} 子块 "
        f"(paper_id={paper_id})"
    )
    
    return parent_chunks, child_chunks