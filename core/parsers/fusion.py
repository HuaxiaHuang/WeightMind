"""
三路解析结果比对与融合
策略：LLM 仲裁 + 规则兜底
"""
import json
import re
from pathlib import Path
from typing import Optional

from loguru import logger

from core.llm.client import llm_client
from core.prompts import PARSE_FUSION_PROMPT


def _truncate(text: str, max_chars: int = 3000) -> str:
    """截断文本用于 LLM 比对（节省 Token）"""
    if len(text) <= max_chars:
        return text
    # 取开头和中间部分，更有代表性
    half = max_chars // 2
    return text[:half] + "\n\n[...中间内容已截断...]\n\n" + text[-half:]


def _simple_quality_score(text: str) -> float:
    """
    简单启发式质量评分
    评估标准：字符数、段落结构、非乱码比例
    """
    if not text or len(text) < 100:
        return 0.0
    
    score = 0.0
    
    # 字符数（越多通常越完整）
    char_score = min(len(text) / 5000, 1.0) * 0.3
    score += char_score
    
    # 段落结构（换行数量）
    paragraph_score = min(text.count("\n\n") / 20, 1.0) * 0.2
    score += paragraph_score
    
    # 英文+数字比例（论文通常以英文为主）
    alpha_ratio = sum(1 for c in text if c.isalnum() or c.isspace()) / len(text)
    score += alpha_ratio * 0.3
    
    # 是否包含常见学术结构
    has_abstract = bool(re.search(r"abstract|摘要", text, re.I))
    has_intro = bool(re.search(r"introduction|introduction|1\.", text, re.I))
    score += (0.1 if has_abstract else 0) + (0.1 if has_intro else 0)
    
    return min(score, 1.0)


async def fuse_parse_results(
    nougat_result: dict,
    marker_result: dict,
    grobid_result: dict,
    use_llm: bool = True,
) -> dict:
    """
    融合三路解析结果
    
    Args:
        nougat_result / marker_result / grobid_result:
            {"success": bool, "text": str, "parse_time": float, ...}
        use_llm: 是否使用 LLM 仲裁（关闭可节省 Token）
    
    Returns:
        {
            "merged_text": str,
            "quality_score": float,
            "successful_tools": list[str],
            "primary_tool": str,
            "issues": list[str],
        }
    """
    # ── 1. 收集成功的解析结果 ──────────────────────────────
    results = {
        "nougat": nougat_result,
        "marker": marker_result,
        "grobid": grobid_result,
    }
    
    successful = {
        name: r for name, r in results.items()
        if r.get("success") and r.get("text", "").strip()
    }
    
    logger.info(f"解析成功工具: {list(successful.keys())}")
    
    if not successful:
        return {
            "merged_text": "",
            "quality_score": 0.0,
            "successful_tools": [],
            "primary_tool": "none",
            "issues": ["所有解析工具均失败"],
        }
    
    # ── 2. 只有一个成功时直接返回 ─────────────────────────
    if len(successful) == 1:
        name, r = next(iter(successful.items()))
        return {
            "merged_text": r["text"],
            "quality_score": _simple_quality_score(r["text"]),
            "successful_tools": [name],
            "primary_tool": name,
            "issues": [f"只有 {name} 解析成功"],
        }
    
    # ── 3. 计算各工具的启发式质量分 ────────────────────────
    quality_scores = {
        name: _simple_quality_score(r["text"])
        for name, r in successful.items()
    }
    logger.debug(f"各工具质量分: {quality_scores}")
    
    # 选择质量最高的作为主版本（LLM 兜底）
    primary_tool = max(quality_scores, key=lambda k: quality_scores[k])
    primary_text = successful[primary_tool]["text"]
    
    # ── 4. LLM 仲裁融合 ────────────────────────────────────
    if use_llm and len(successful) >= 2:
        try:
            texts = {name: r["text"] for name, r in successful.items()}
            
            prompt = PARSE_FUSION_PROMPT.format(
                text_a=_truncate(texts.get("nougat", "（Nougat 未运行）")),
                text_b=_truncate(texts.get("marker", "（Marker 未运行）")),
                text_c=_truncate(texts.get("grobid", "（Grobid 未运行）")),
            )
            
            response = await llm_client.complete(prompt, json_mode=True)
            fusion_data = json.loads(response)
            
            merged_text = fusion_data.get("merged_text", primary_text)
            quality_score = fusion_data.get("quality_score", quality_scores[primary_tool])
            issues = fusion_data.get("issues_found", [])
            
            # 如果 LLM 融合结果比主版本短太多，说明可能截断了，用主版本
            if len(merged_text) < len(primary_text) * 0.5:
                logger.warning("LLM 融合文本过短，回退到最优单一解析结果")
                merged_text = primary_text
                quality_score = quality_scores[primary_tool]
                issues.append("LLM 融合文本异常，已回退到最优单一结果")
            
            return {
                "merged_text": merged_text,
                "quality_score": quality_score,
                "successful_tools": list(successful.keys()),
                "primary_tool": primary_tool,
                "issues": issues,
            }
            
        except Exception as e:
            logger.warning(f"LLM 融合失败，使用规则融合: {e}")
    
    # ── 5. 规则融合兜底：选质量最高的单一结果 ───────────────
    return {
        "merged_text": primary_text,
        "quality_score": quality_scores[primary_tool],
        "successful_tools": list(successful.keys()),
        "primary_tool": primary_tool,
        "issues": [],
    }
