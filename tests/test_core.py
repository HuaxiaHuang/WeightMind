"""
解析器单元测试
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 解析融合测试（纯逻辑，不需要 LLM/GPU） ─────────────────

class TestParserFusion:
    def test_fusion_all_success(self):
        """三路都成功时应返回融合文本"""
        from core.parsers.fusion import fuse_parse_results

        nougat_r = {"success": True, "text": "Nougat text " * 100, "parse_time": 1.0}
        marker_r = {"success": True, "text": "Marker text " * 200, "parse_time": 0.5}
        grobid_r = {"success": True, "text": "Grobid text " * 150, "parse_time": 2.0}

        result = asyncio.run(
            fuse_parse_results(nougat_r, marker_r, grobid_r, use_llm=False)
        )

        assert result["merged_text"], "融合文本不应为空"
        assert result["quality_score"] > 0, "质量分应大于0"
        assert len(result["successful_tools"]) == 3
        assert result["primary_tool"] in ("nougat", "marker", "grobid")

    def test_fusion_one_success(self):
        """只有一个工具成功时，直接返回该工具结果"""
        from core.parsers.fusion import fuse_parse_results

        nougat_r = {"success": False, "text": "", "parse_time": 0}
        marker_r = {"success": True, "text": "Marker text content " * 100, "parse_time": 0.5}
        grobid_r = {"success": False, "text": "", "parse_time": 0}

        result = asyncio.run(
            fuse_parse_results(nougat_r, marker_r, grobid_r, use_llm=False)
        )

        assert result["merged_text"] == marker_r["text"]
        assert result["primary_tool"] == "marker"
        assert result["successful_tools"] == ["marker"]

    def test_fusion_all_fail(self):
        """全部失败时返回空文本"""
        from core.parsers.fusion import fuse_parse_results

        empty = {"success": False, "text": "", "parse_time": 0}
        result = asyncio.run(
            fuse_parse_results(empty, empty, empty, use_llm=False)
        )

        assert result["merged_text"] == ""
        assert result["quality_score"] == 0.0
        assert result["successful_tools"] == []


# ── Chunker 测试 ──────────────────────────────────────────────

class TestChunker:
    SAMPLE_TEXT = (
        "Abstract\nThis paper introduces a new approach to machine learning. "
        "We propose a novel architecture that combines attention mechanisms "
        "with convolutional layers.\n\n"
        "Introduction\nDeep learning has transformed many fields. "
        "In this work, we build upon previous research.\n\n"
    ) * 20

    def test_parent_child_structure(self):
        """父子块结构正确性"""
        from core.indexing.chunker import create_parent_child_chunks

        parents, children = create_parent_child_chunks(
            text=self.SAMPLE_TEXT,
            paper_id="test-001",
            domain="computer_science",
        )

        assert len(parents) > 0, "应该生成父块"
        assert len(children) > 0, "应该生成子块"
        assert len(children) >= len(parents), "子块数应 >= 父块数"

    def test_parent_child_linkage(self):
        """每个子块都应该指向一个有效的父块"""
        from core.indexing.chunker import create_parent_child_chunks

        parents, children = create_parent_child_chunks(
            text=self.SAMPLE_TEXT,
            paper_id="test-001",
            domain="computer_science",
        )

        parent_ids = {p.id for p in parents}
        for child in children:
            assert child.parent_id in parent_ids, (
                f"子块 {child.id} 的 parent_id {child.parent_id} 在父块中不存在"
            )

    def test_chunk_types(self):
        """父块类型为 parent，子块类型为 child"""
        from core.indexing.chunker import create_parent_child_chunks

        parents, children = create_parent_child_chunks(
            text=self.SAMPLE_TEXT,
            paper_id="test-001",
            domain="cs",
        )

        assert all(p.chunk_type == "parent" for p in parents)
        assert all(c.chunk_type == "child" for c in children)

    def test_child_smaller_than_parent(self):
        """子块文本应该不超过父块文本长度"""
        from core.indexing.chunker import create_parent_child_chunks

        parents, children = create_parent_child_chunks(
            text=self.SAMPLE_TEXT,
            paper_id="test-001",
            domain="cs",
        )

        parent_map = {p.id: p for p in parents}
        for child in children:
            parent = parent_map[child.parent_id]
            assert len(child.text) <= len(parent.text) + 100  # 允许少量误差


# ── LLM Client 测试（Mock） ───────────────────────────────────

class TestLLMClient:
    def test_extract_json_from_code_block(self):
        """从 ```json ... ``` 代码块中提取 JSON"""
        from core.llm.client import LLMClient

        text = '```json\n{"key": "value", "num": 42}\n```'
        result = LLMClient._extract_json(text)
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["num"] == 42

    def test_extract_json_direct(self):
        """直接是 JSON 字符串时"""
        from core.llm.client import LLMClient

        text = '{"domain": "computer_science", "keywords": ["AI", "ML"]}'
        result = LLMClient._extract_json(text)
        import json
        parsed = json.loads(result)
        assert parsed["domain"] == "computer_science"

    def test_extract_json_embedded(self):
        """JSON 嵌入在文字中"""
        from core.llm.client import LLMClient

        text = '这是一个响应。\n{"result": "ok"}\n结束。'
        result = LLMClient._extract_json(text)
        import json
        parsed = json.loads(result)
        assert parsed["result"] == "ok"


# ── Qdrant 工具函数测试 ───────────────────────────────────────

class TestQdrantUtils:
    def test_domain_to_collection(self):
        """领域名 -> Collection 名转换"""
        from database.qdrant_client import domain_to_collection

        assert domain_to_collection("computer_science") == "sci_rag_computer_science"
        assert domain_to_collection("Natural Language Processing") == "sci_rag_natural_language_processing"
        assert domain_to_collection("bio-med") == "sci_rag_bio_med"

    def test_collection_to_domain(self):
        """Collection 名 -> 领域名转换"""
        from database.qdrant_client import collection_to_domain

        assert collection_to_domain("sci_rag_computer_science") == "computer_science"
        assert collection_to_domain("sci_rag_biology") == "biology"

    def test_rrf_fusion(self):
        """RRF 融合排序测试"""
        from database.qdrant_client import QdrantManager

        class MockResult:
            def __init__(self, id, score, payload):
                self.id = id
                self.score = score
                self.payload = payload

        dense = [MockResult("a", 0.9, {"text": "dense top"}), MockResult("b", 0.8, {"text": "second"})]
        sparse = [MockResult("b", 0.95, {"text": "sparse top"}), MockResult("c", 0.7, {"text": "third"})]

        fused = QdrantManager._rrf_fusion(dense, sparse, top_k=3)

        # b 在两个结果中都出现，RRF 分数应该最高
        assert fused[0]["id"] == "b", "在两个列表中都出现的结果应排第一"
        assert len(fused) <= 3


# ── 配置测试 ──────────────────────────────────────────────────

class TestConfig:
    def test_settings_load(self):
        """配置应能正常加载"""
        from core.config import settings

        assert settings.APP_PORT == 8000
        assert settings.CHILD_CHUNK_SIZE < settings.PARENT_CHUNK_SIZE
        assert 0 < settings.DENSE_WEIGHT <= 1
        assert 0 < settings.SPARSE_WEIGHT <= 1

    def test_data_paths_are_absolute(self):
        """数据路径应该是绝对路径"""
        from core.config import settings
        from pathlib import Path

        assert Path(settings.RAW_DATA_DIR).is_absolute()
        assert Path(settings.TEMP_DIR).is_absolute()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
