"""
端到端管道测试脚本
用于验证整个处理流程是否正常工作（无需完整启动服务）
用法: python scripts/test_pipeline.py <pdf_path>
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


async def test_parsers(pdf_path: Path):
    """测试三路解析器"""
    logger.info("\n" + "=" * 40)
    logger.info("🔍 测试解析器")
    logger.info("=" * 40)

    from core.parsers.nougat_parser import NougatParser
    from core.parsers.marker_parser import MarkerParser
    from core.parsers.grobid_parser import GrobidParser

    results = {}

    # Marker（通常最快，优先测试）
    logger.info("测试 Marker...")
    marker = MarkerParser()
    t0 = time.time()
    marker_r = marker.parse(pdf_path)
    logger.info(
        f"  Marker: {'✅' if marker_r['success'] else '❌'} "
        f"| {len(marker_r.get('text',''))} 字符 "
        f"| {time.time()-t0:.1f}s"
    )
    results["marker"] = marker_r

    # Grobid
    logger.info("测试 Grobid...")
    grobid = GrobidParser()
    t0 = time.time()
    grobid_r = grobid.parse(pdf_path)
    logger.info(
        f"  Grobid: {'✅' if grobid_r['success'] else '❌'} "
        f"| {len(grobid_r.get('text',''))} 字符 "
        f"| {time.time()-t0:.1f}s"
        + (f" | 错误: {grobid_r.get('error','')}" if not grobid_r['success'] else "")
    )
    results["grobid"] = grobid_r

    # Nougat（最慢，最后测试）
    logger.info("测试 Nougat（可能较慢）...")
    nougat = NougatParser()
    t0 = time.time()
    nougat_r = nougat.parse(pdf_path)
    logger.info(
        f"  Nougat: {'✅' if nougat_r['success'] else '❌'} "
        f"| {len(nougat_r.get('text',''))} 字符 "
        f"| {time.time()-t0:.1f}s"
        + (f" | 错误: {nougat_r.get('error','')}" if not nougat_r['success'] else "")
    )
    results["nougat"] = nougat_r

    return results


async def test_fusion(parse_results: dict):
    """测试三路融合"""
    logger.info("\n" + "=" * 40)
    logger.info("🔗 测试解析结果融合")
    logger.info("=" * 40)

    from core.parsers.fusion import fuse_parse_results

    fusion = await fuse_parse_results(
        parse_results.get("nougat", {"success": False, "text": ""}),
        parse_results.get("marker", {"success": False, "text": ""}),
        parse_results.get("grobid", {"success": False, "text": ""}),
        use_llm=False,  # 测试时不调用 LLM，只用规则
    )

    logger.info(f"  融合质量分: {fusion['quality_score']:.3f}")
    logger.info(f"  成功工具: {fusion['successful_tools']}")
    logger.info(f"  主工具: {fusion['primary_tool']}")
    logger.info(f"  文本长度: {len(fusion['merged_text'])} 字符")
    logger.info(f"  文本预览: {fusion['merged_text'][:200]}...")

    return fusion


async def test_llm_classification(text: str):
    """测试 LLM 分类（需要配置 API Key）"""
    logger.info("\n" + "=" * 40)
    logger.info("🧠 测试 LLM 领域分类")
    logger.info("=" * 40)

    from core.llm.client import llm_client
    from core.prompts import PAPER_CLASSIFICATION_PROMPT

    try:
        sample = text[:2000]
        result_str = await llm_client.complete(
            PAPER_CLASSIFICATION_PROMPT.format(text=sample),
            json_mode=True,
        )
        result = json.loads(result_str)
        logger.info(f"  领域: {result.get('domain')}")
        logger.info(f"  细分: {result.get('subdomain')}")
        logger.info(f"  关键词: {result.get('keywords', [])[:5]}")
        logger.info(f"  摘要: {result.get('abstract_summary', '')[:100]}...")
        return result
    except Exception as e:
        logger.error(f"  LLM 分类失败: {e}")
        logger.warning("  请检查 .env 中的 LLM_API_KEY 和 LLM_BASE_URL 配置")
        return None


async def test_chunking(text: str):
    """测试父子块分割"""
    logger.info("\n" + "=" * 40)
    logger.info("✂️  测试父子块分割")
    logger.info("=" * 40)

    from core.indexing.chunker import create_parent_child_chunks

    parents, children = create_parent_child_chunks(
        text=text,
        paper_id="test-paper-001",
        domain="computer_science",
    )

    logger.info(f"  父块数量: {len(parents)}")
    logger.info(f"  子块数量: {len(children)}")
    if parents:
        logger.info(f"  父块示例（前200字符）: {parents[0].text[:200]}...")
    if children:
        logger.info(f"  子块示例（前100字符）: {children[0].text[:100]}...")
        logger.info(f"  子块->父块 ID: {children[0].parent_id}")

    return parents, children


async def test_embedding(children):
    """测试 BGE Embedding（需要下载模型）"""
    logger.info("\n" + "=" * 40)
    logger.info("🧮 测试 BGE-M3 Embedding")
    logger.info("=" * 40)

    try:
        from core.indexing.embedder import embedder

        sample_texts = [c.text for c in children[:3]]
        logger.info(f"  对 {len(sample_texts)} 个子块做 Embedding...")
        t0 = time.time()

        embeddings = embedder.encode(sample_texts, return_sparse=True)
        elapsed = time.time() - t0

        logger.info(f"  耗时: {elapsed:.2f}s")
        logger.info(f"  稠密向量维度: {len(embeddings[0]['dense'])}")
        logger.info(f"  稀疏向量非零元素: {len(embeddings[0].get('sparse', {}).get('indices', []))}")
        logger.success("  ✅ Embedding 测试通过")
        return True
    except Exception as e:
        logger.error(f"  Embedding 失败: {e}")
        logger.warning("  请安装: pip install FlagEmbedding torch")
        return False


def print_summary(results: dict):
    logger.info("\n" + "=" * 50)
    logger.info("📊 测试汇总")
    logger.info("=" * 50)
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        logger.info(f"  {icon} {name}")


async def main():
    if len(sys.argv) < 2:
        # 如果没有提供 PDF，生成一个简单测试文本
        logger.warning("未提供 PDF 路径，使用模拟文本测试")
        test_text = """
        Abstract
        This paper presents a novel approach to natural language processing using transformer-based 
        architectures. We propose a new attention mechanism that significantly improves performance 
        on benchmark datasets.

        Introduction
        Large language models have revolutionized the field of artificial intelligence. In this work,
        we introduce BERT-Enhanced, a model that builds upon the original BERT architecture.

        Methodology
        Our approach uses a multi-head attention mechanism with 12 layers and 768 hidden dimensions.
        We train on a corpus of 100 billion tokens using masked language modeling.

        Results
        Our model achieves state-of-the-art results on GLUE, SuperGLUE, and SQuAD benchmarks.
        """ * 5

        all_results = {}
        parents, children = await test_chunking(test_text)
        all_results["分块测试"] = len(parents) > 0 and len(children) > 0

        classification = await test_llm_classification(test_text)
        all_results["LLM分类"] = classification is not None

        embed_ok = await test_embedding(children)
        all_results["BGE Embedding"] = embed_ok

        print_summary(all_results)
        return

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        logger.error(f"文件不存在: {pdf_path}")
        sys.exit(1)

    logger.info(f"测试 PDF: {pdf_path}")
    all_results = {}

    # 1. 解析测试
    parse_results = await test_parsers(pdf_path)
    any_success = any(r["success"] for r in parse_results.values())
    all_results["三路解析"] = any_success

    if any_success:
        # 2. 融合测试
        fusion = await test_fusion(parse_results)
        all_results["结果融合"] = bool(fusion["merged_text"])

        if fusion["merged_text"]:
            # 3. 分类测试
            classification = await test_llm_classification(fusion["merged_text"])
            all_results["LLM分类"] = classification is not None

            # 4. 分块测试
            parents, children = await test_chunking(fusion["merged_text"])
            all_results["父子块分割"] = len(parents) > 0

            # 5. Embedding 测试
            if children:
                embed_ok = await test_embedding(children)
                all_results["BGE Embedding"] = embed_ok

    print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
