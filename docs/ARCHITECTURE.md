# Sci_RAG 系统架构文档

> 科研论文深度研究 RAG 系统 — 架构决策记录与设计说明

---

## 1. 系统概述

Sci_RAG 是一个面向科研人员的论文知识库系统，核心能力：

- **知识摄入管道**：多工具解析 PDF → 自动领域分类 → 结构化入库
- **深度检索生成**：HyDE + 混合检索 + 重排 + 父子块 → LLM 生成回答

```
用户上传 PDF
      │
      ▼
┌─────────────────────────────────────────────────┐
│              Celery 异步任务链                    │
│  Parse(×3) → 比对融合 → LLM摘要 → 领域判定      │
│      → 保存物理文件 → Chunk → Embed → 入Qdrant  │
│      → 元数据写 PostgreSQL                        │
└─────────────────────────────────────────────────┘
                                                    
用户提问
      │
      ▼
┌─────────────────────────────────────────────────┐
│              RAG 检索生成管道                     │
│  查询重写 → 领域路由 → HyDE伪答案               │
│      → Qdrant混合检索 → BGE重排                 │
│      → 父子块扩展 → LLM生成最终回答             │
└─────────────────────────────────────────────────┘
```

---

## 2. 架构决策记录（ADR）

### ADR-001: 采用 FastAPI + Celery 分离请求与重型任务

**状态**: 已接受

**上下文**  
PDF解析（尤其是Nougat、Grobid）耗时30秒~5分钟，直接在HTTP请求中处理会导致前端超时，用户体验极差。

**决策**  
文件上传接口仅接收文件并创建任务记录，立即返回 `task_id`；实际解析、向量化全部交给 Celery Worker 异步执行；前端通过 `/task/{id}/status` 轮询进度。

**后果**  
✅ 前端响应始终< 200ms  
✅ Worker 可横向扩展  
⚠️ 引入 Redis 依赖  
⚠️ 需要额外处理任务失败重试逻辑

---

### ADR-002: 向量库按领域划分 Qdrant Collection

**状态**: 已接受

**上下文**  
你的问题："向量库是否要分开？" — 这是个真实的权衡。

| 方案 | 优势 | 劣势 |
|------|------|------|
| 单一 Collection + 领域 Payload 过滤 | 跨领域查询更容易 | 随规模增长召回噪音增大 |
| 按领域划分多 Collection | 领域内精准召回，隔离性好 | 查询时需要先做领域路由 |

**决策**  
采用**按领域划分 Collection**，同时在查询时用 LLM 先做领域路由。理由：科研场景中，论文领域差异巨大（NLP vs 材料学 vs 医学），混在一起会严重稀释语义相关性。

**后果**  
✅ 领域内召回精度高  
✅ 向量空间语义聚集  
⚠️ 跨领域问题需要 multi-collection 查询（已在 retrieval 模块实现）

---

### ADR-003: 三路解析 + LLM 仲裁融合

**状态**: 已接受

**上下文**  
任何单一 PDF 解析工具都有盲区：Nougat 擅长公式但慢；Marker 速度快但对复杂布局有时出错；Grobid 擅长结构化但需要 Docker。

**决策**  
三路并行解析，用 LLM 对比三份输出的**差异区域**做仲裁融合，而非简单投票。差异区域标记可信度分数存入元数据。

**后果**  
✅ 解析质量大幅提升，尤其是公式和表格  
✅ 可信度分数给下游 RAG 提供额外信号  
⚠️ 解析时间增加（但在 Celery 后台，不影响用户体验）  
⚠️ Grobid 需要 Docker，初次配置成本略高

---

### ADR-004: HyDE（假设文档嵌入）策略

**状态**: 已接受  

**上下文**  
你的问题5："是否需要先让大模型进行初步预答，然后再进行RAG？" — 这正是 HyDE 的思想。

**决策**  
采用 HyDE：先让 LLM 基于问题生成一个**假设性答案**（不调用真实知识库，纯靠 LLM 参数记忆），将这个假设答案做 Embedding，再用这个向量去检索真实文档。假设答案的语义空间比原始问题更接近真实答案，召回率显著提升。

**流程**:  
```
用户问题 → LLM生成假设答案 → Embed(假设答案) → 
混合检索(Dense+Sparse) → BGE重排 → 父子块扩展 → 
LLM最终生成（基于真实上下文）
```

**后果**  
✅ 召回率比直接查询高 15-25%（论文实测）  
✅ 对专业术语描述不准确的问题特别有效  
⚠️ 每次查询额外一次 LLM 调用（但是小模型调用，成本低）

---

### ADR-005: 父子块检索（Auto-Merging Retrieval）

**状态**: 已接受

**上下文**  
Chunking 是 RAG 效果的核心瓶颈：小块精准但上下文不足；大块上下文完整但引入噪音。

**决策**  
双层分块：  
- **子块（Child Chunk）**：128 token，用于向量检索（精准定位）  
- **父块（Parent Chunk）**：512 token，用于喂给 LLM（完整上下文）

命中子块后，自动返回其父块给 LLM，兼顾精准度和上下文完整性。

---

## 3. 技术栈总览

| 层次 | 技术选型 | 版本建议 |
|------|----------|----------|
| API 框架 | FastAPI | ≥ 0.111 |
| 异步任务 | Celery + Redis | Celery 5.x |
| RAG 编排 | LlamaIndex | ≥ 0.10 |
| 关系数据库 | PostgreSQL + SQLAlchemy | PG 16 |
| 向量数据库 | Qdrant | ≥ 1.9 |
| Embedding | BAAI/bge-m3 | — |
| Reranker | BAAI/bge-reranker-v2-m3 | — |
| PDF 解析 | Nougat + Marker + Grobid | — |
| 前端 | Streamlit | ≥ 1.34 |
| 容器化 | Docker Compose | — |

---

## 4. 数据流图（C4 Component Level）

```
┌──────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                    │
│   ┌──────────────────┐    ┌────────────────────────────┐ │
│   │  Chat Page       │    │  Knowledge Base Manager    │ │
│   │  - 对话框        │    │  - 拖拽上传 PDF            │ │
│   │  - 流式输出      │    │  - 任务进度条              │ │
│   │  - 引用来源展示  │    │  - 论文元数据列表          │ │
│   └────────┬─────────┘    └────────────┬───────────────┘ │
└────────────┼─────────────────────────── ┼────────────────┘
             │ HTTP/SSE                   │ HTTP
             ▼                            ▼
┌──────────────────────────────────────────────────────────┐
│                     FastAPI Backend                       │
│   /chat/stream          /upload          /task/{id}/status│
│   /papers               /papers/{id}     /domains         │
└──────────┬──────────────────┬─────────────────┬──────────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
    │ RAG Pipeline│   │ Celery Worker│   │ PostgreSQL  │
    │ (LlamaIndex)│   │ 异步解析任务 │   │ 元数据 ORM  │
    └──────┬──────┘   └──────┬───────┘   └─────────────┘
           │                  │
           ▼                  ▼
    ┌─────────────────────────────┐
    │         Qdrant              │
    │  Collection: cs_ai          │
    │  Collection: biology        │
    │  Collection: medicine       │
    │  Collection: ...            │
    └─────────────────────────────┘
```

---

## 5. 目录结构说明

```
Sci_RAG/
├── api/                    # FastAPI 路由层
│   ├── __init__.py
│   ├── routes_chat.py      # 对话 + 流式输出接口
│   ├── routes_upload.py    # 文件上传 + 任务状态查询
│   └── routes_papers.py    # 论文元数据 CRUD
│
├── core/                   # 核心业务逻辑层
│   ├── parsers/            
│   │   ├── __init__.py
│   │   ├── nougat_parser.py    # Nougat 解析器
│   │   ├── marker_parser.py    # Marker 解析器
│   │   ├── grobid_parser.py    # Grobid 解析器
│   │   └── fusion.py           # 三路结果比对融合
│   ├── indexing/           
│   │   ├── __init__.py
│   │   ├── chunker.py          # 父子块分割
│   │   └── embedder.py         # BGE-M3 向量化 + 入Qdrant
│   ├── retrieval/          
│   │   ├── __init__.py
│   │   ├── query_rewriter.py   # 查询重写
│   │   ├── domain_router.py    # 领域路由
│   │   ├── hyde.py             # HyDE 假设文档生成
│   │   ├── hybrid_search.py    # 混合检索(Dense+Sparse)
│   │   ├── reranker.py         # BGE 重排
│   │   └── parent_retriever.py # 父子块扩展
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py           # LLM 客户端统一封装
│   └── prompts.py              # 所有 Prompt 模板集中管理
│
├── database/               # 数据访问层
│   ├── __init__.py
│   ├── models.py           # SQLAlchemy ORM 模型
│   ├── crud.py             # 数据库操作封装
│   ├── session.py          # 数据库连接池管理
│   └── qdrant_client.py    # Qdrant 客户端封装
│
├── worker/                 # 异步任务层
│   ├── __init__.py
│   ├── celery_app.py       # Celery 应用配置
│   └── tasks.py            # 任务定义：parse → classify → index
│
├── frontend/               # Streamlit 前端
│   ├── app.py              # 主入口，底部 Tab 导航
│   ├── pages/
│   │   ├── chat.py         # 对话页
│   │   └── knowledge_base.py # 知识库管理页
│   └── components/
│       └── file_uploader.py   # 文件拖拽上传组件
│
├── data/                   # 本地数据存储（gitignore）
│   ├── raw/                # 原始PDF + 解析文本（按领域）
│   │   └── {domain}/
│   │       ├── {paper_id}.pdf
│   │       └── {paper_id}_extracted.md
│   └── temp/               # 上传临时文件
│
├── tests/                  # 单元测试
│   ├── test_parsers.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── scripts/                # 运维脚本
│   ├── init_db.py          # 初始化数据库表
│   └── test_pipeline.py    # 端到端测试脚本
│
├── docs/                   # 文档
│   └── ARCHITECTURE.md     # 本文件
│
├── .env.example            # 环境变量模板
├── .env                    # 实际配置（gitignore）
├── .gitignore
├── docker-compose.yml      # 一键启动基础设施
├── requirements.txt        # Python 依赖
├── main.py                 # FastAPI 应用入口
└── README.md               # 快速上手指南
```
