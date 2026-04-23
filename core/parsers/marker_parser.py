"""
Marker 解析器
速度快，布局保持好，输出 Markdown 格式
"""
import time
from pathlib import Path

from loguru import logger


class MarkerParser:
    """
    使用 marker-pdf 库解析 PDF
    安装: pip install marker-pdf
    """

    def __init__(self):
        self._converter = None

    def _get_converter(self):
        """延迟加载 Marker，避免启动时占用内存"""
        if self._converter is None:
            try:
                from marker.converters.pdf import PdfConverter
                from marker.models import create_model_dict

                self._converter = PdfConverter(
                    artifact_dict=create_model_dict()
                )
            except Exception as e:
                # 🧨 终极照妖镜：捕获所有错误并打印完整的报错调用栈
                import traceback
                logger.error("❌ ===============================================")
                logger.error("❌ Marker 加载发生了底层崩溃，真凶如下：")
                logger.error(f"❌ 简略错误: {repr(e)}")
                logger.error(f"❌ 完整调用栈:\n{traceback.format_exc()}")
                logger.error("❌ ===============================================")
                
                logger.warning("marker-pdf 加载失败，本次解析跳过该模型...")
                return None
        return self._converter

    def parse(self, pdf_path: Path) -> dict:
        """
        解析 PDF 文件
        Returns: {"success": bool, "text": str, "parse_time": float, "error": str|None}
        """
        start = time.time()
        
        if not pdf_path.exists():
            return {"success": False, "text": "", "parse_time": 0, "error": "文件不存在"}

        converter = self._get_converter()
        if converter is None:
            return {
                "success": False,
                "text": "",
                "parse_time": 0,
                "error": "Marker 未安装",
            }

        try:
            rendered = converter(str(pdf_path))
            
            # rendered.markdown 是主要文本内容
            text = rendered.markdown
            parse_time = time.time() - start
            
            logger.info(f"Marker 解析完成，{len(text)} 字符，耗时 {parse_time:.1f}s")
            
            return {
                "success": True,
                "text": text,
                "parse_time": parse_time,
                "error": None,
                "metadata": rendered.metadata if hasattr(rendered, "metadata") else {},
            }
            
        except Exception as e:
            logger.error(f"Marker 解析异常: {e}")
            return {
                "success": False,
                "text": "",
                "parse_time": time.time() - start,
                "error": str(e),
            }
