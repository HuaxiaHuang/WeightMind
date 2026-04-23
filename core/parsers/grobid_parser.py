"""
Grobid 解析器
通过 HTTP 调用 Grobid Docker 服务
擅长学术论文结构化提取（章节、引用、作者等）
"""
import time
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import requests
from loguru import logger

from core.config import settings


class GrobidParser:
    """
    通过 Grobid REST API 解析 PDF
    需要先启动 Grobid Docker 服务：
    docker run -p 8070:8070 lfoppiano/grobid:0.8.0
    """

    def __init__(self, grobid_url: Optional[str] = None):
        self.base_url = grobid_url or settings.GROBID_URL
        self.timeout = 120

    def _is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/isalive", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def parse(self, pdf_path: Path) -> dict:
        """
        调用 Grobid 解析 PDF 为 TEI XML，然后提取纯文本
        Returns: {"success": bool, "text": str, "parse_time": float, "metadata": dict, "error": str|None}
        """
        start = time.time()
        
        if not pdf_path.exists():
            return {"success": False, "text": "", "parse_time": 0, "error": "文件不存在"}

        if not self._is_available():
            logger.warning(f"Grobid 服务不可用: {self.base_url}")
            return {
                "success": False,
                "text": "",
                "parse_time": 0,
                "error": f"Grobid 服务未启动 ({self.base_url})。请运行 docker-compose up grobid",
            }

        try:
            with open(pdf_path, "rb") as f:
                response = requests.post(
                    f"{self.base_url}/api/processFulltextDocument",
                    files={"input": f},
                    data={
                        "consolidateHeader": "1",       # 合并头部信息
                        "consolidateCitations": "0",    # 不合并引用（节省时间）
                        "includeRawAffiliations": "1",
                        "segmentSentences": "1",
                    },
                    timeout=self.timeout,
                )
            
            parse_time = time.time() - start
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "text": "",
                    "parse_time": parse_time,
                    "error": f"Grobid HTTP {response.status_code}: {response.text[:200]}",
                }
            
            # 解析 TEI XML
            tei_xml = response.text
            text, metadata = self._parse_tei(tei_xml)
            
            logger.info(f"Grobid 解析完成，{len(text)} 字符，耗时 {parse_time:.1f}s")
            
            return {
                "success": True,
                "text": text,
                "parse_time": parse_time,
                "metadata": metadata,
                "error": None,
            }
            
        except requests.Timeout:
            return {
                "success": False,
                "text": "",
                "parse_time": time.time() - start,
                "error": "Grobid 解析超时",
            }
        except Exception as e:
            logger.error(f"Grobid 解析异常: {e}")
            return {
                "success": False,
                "text": "",
                "parse_time": time.time() - start,
                "error": str(e),
            }

    @staticmethod
    def _parse_tei(tei_xml: str) -> tuple[str, dict]:
        """
        从 TEI XML 中提取纯文本和元数据
        """
        try:
            # 定义命名空间
            ns = {
                "tei": "http://www.tei-c.org/ns/1.0",
                "xml": "http://www.w3.org/XML/1998/namespace",
            }
            
            root = ET.fromstring(tei_xml)
            
            # ── 提取元数据 ──────────────────────────────────
            metadata = {}
            
            # 标题
            title_el = root.find(".//tei:titleStmt/tei:title", ns)
            if title_el is not None:
                metadata["title"] = title_el.text
            
            # 作者
            authors = []
            for author in root.findall(".//tei:biblStruct//tei:author", ns):
                forename = author.findtext(".//tei:forename", default="", namespaces=ns)
                surname = author.findtext(".//tei:surname", default="", namespaces=ns)
                if forename or surname:
                    authors.append(f"{forename} {surname}".strip())
            metadata["authors"] = authors
            
            # 摘要
            abstract_el = root.find(".//tei:abstract", ns)
            if abstract_el is not None:
                metadata["abstract"] = " ".join(abstract_el.itertext()).strip()
            
            # ── 提取正文 ────────────────────────────────────
            body = root.find(".//tei:body", ns)
            text_parts = []
            
            if body is not None:
                for div in body.findall(".//tei:div", ns):
                    # 章节标题
                    head = div.find("tei:head", ns)
                    if head is not None and head.text:
                        text_parts.append(f"\n## {head.text}\n")
                    
                    # 段落
                    for p in div.findall("tei:p", ns):
                        para_text = " ".join(p.itertext()).strip()
                        if para_text:
                            text_parts.append(para_text)
            
            # 如果没有正文，用摘要
            if not text_parts and "abstract" in metadata:
                text_parts.append(metadata["abstract"])
            
            full_text = "\n\n".join(text_parts)
            return full_text, metadata
            
        except ET.ParseError as e:
            logger.warning(f"TEI XML 解析错误: {e}")
            # 回退：直接提取所有文本
            try:
                root = ET.fromstring(tei_xml)
                text = " ".join(root.itertext())
                return text, {}
            except Exception:
                return "", {}
