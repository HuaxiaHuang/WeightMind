"""
Nougat 解析器
擅长处理科学文献中的数学公式和特殊符号
需要 GPU 效果最佳，CPU 也可运行但较慢
"""
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from loguru import logger


class NougatParser:
    """
    调用 Nougat CLI 解析 PDF
    安装: pip install nougat-ocr
    """

    def __init__(self, model: str = "0.1.0-small"):
        self.model = model
        self._check_available()

    def _check_available(self) -> bool:
        try:
            result = subprocess.run(
                ["nougat", "--help"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Nougat 未安装或不可用，将跳过 Nougat 解析")
            return False

    def parse(self, pdf_path: Path, output_dir: Optional[Path] = None) -> dict:
        """
        解析 PDF 文件
        Returns: {"success": bool, "text": str, "parse_time": float, "error": str|None}
        """
        start = time.time()
        
        if not pdf_path.exists():
            return {"success": False, "text": "", "parse_time": 0, "error": "文件不存在"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) if output_dir is None else output_dir
            
            try:
                cmd = [
                    "nougat",
                    str(pdf_path),
                    "--out", str(out_dir),
                    "--model", self.model,
                    "--no-skipping",  # 不跳过难以识别的页面
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 分钟超时
                )
                
                parse_time = time.time() - start
                
                if result.returncode != 0:
                    logger.error(f"Nougat 解析失败: {result.stderr}")
                    return {
                        "success": False,
                        "text": "",
                        "parse_time": parse_time,
                        "error": result.stderr[:500],
                    }
                
                # 查找输出的 .mmd 文件（Nougat 输出格式）
                mmd_files = list(out_dir.glob("*.mmd"))
                if not mmd_files:
                    return {
                        "success": False,
                        "text": "",
                        "parse_time": parse_time,
                        "error": "Nougat 未生成输出文件",
                    }
                
                text = mmd_files[0].read_text(encoding="utf-8", errors="replace")
                logger.info(f"Nougat 解析完成，{len(text)} 字符，耗时 {parse_time:.1f}s")
                
                return {
                    "success": True,
                    "text": text,
                    "parse_time": parse_time,
                    "error": None,
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "text": "",
                    "parse_time": time.time() - start,
                    "error": "Nougat 解析超时（>5分钟）",
                }
            except Exception as e:
                return {
                    "success": False,
                    "text": "",
                    "parse_time": time.time() - start,
                    "error": str(e),
                }
