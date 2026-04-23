# 🔬 Sci_RAG — 科研论文深度研究系统

> 基于 **RAG + LLM** 的科研论文智能问答系统。上传 PDF → 三路解析 → 自动分类 → 向量入库 → 深度检索问答。

---

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| **三路解析** | Nougat + Marker + Grobid 并行解析，LLM 仲裁融合，提升准确率 |
| **自动领域分类** | LLM 判断论文所属领域，按领域建立独立向量库 |
| **父子块检索** | 子块精准定位 + 父块完整上下文，兼顾召回率和生成质量 |
| **HyDE 策略** | 先生成假设答案再检索，大幅提升语义匹配准确率 |
| **混合检索** | BGE-M3 稠密 + 稀疏双路向量，RRF 融合排序 |
| **BGE 重排** | bge-reranker-v2-m3 二次精排，Top-K 准确率显著提升 |
| **流式输出** | SSE 流式输出，逐字显示回答，支持对话历史 |
| **异步处理** | Celery 后台任务，上传后立即返回，前端进度条轮询 |

---

## 🗂️ 目录结构

```
Sci_RAG/
├── api/                    # FastAPI 路由（对话、上传、论文管理）
├── core/
│   ├── parsers/            # Nougat / Marker / Grobid + 融合
│   ├── indexing/           # 父子块分割 + BGE Embedding
│   ├── retrieval/          # HyDE + 混合检索 + 重排 + 领域路由
│   ├── llm/                # LLM 客户端统一封装
│   └── prompts.py          # 所有 Prompt 模板
├── database/               # PostgreSQL ORM + Qdrant 客户端
├── worker/                 # Celery 异步任务
├── frontend/               # Streamlit 前端
│   ├── app.py              # 主入口（底部 Tab 导航）
│   └── pages/
│       ├── chat.py         # 对话页
│       └── knowledge_base.py  # 知识库管理页
├── data/
│   ├── raw/{domain}/       # 原始 PDF + 提取文本（按领域分类）
│   └── temp/               # 上传临时文件
├── scripts/
│   ├── init_db.py          # 数据库初始化
│   ├── test_pipeline.py    # 端到端管道测试
│   └── start.py            # 一键启动脚本
├── tests/                  # 单元测试
├── docker-compose.yml      # PostgreSQL + Qdrant + Redis + Grobid
├── .env.example            # 环境变量模板
└── requirements.txt        # Python 依赖
```

---

## 🚀 快速上手

### 第一步：环境准备

**系统要求**
- Python 3.10+
- Docker Desktop（用于启动数据库服务）
- 至少 8GB RAM（运行 BGE 模型）
- NVIDIA GPU（可选，大幅提升 Embedding 速度）

**安装 Python 依赖**

```bash
cd E:\WorkBuddy\Sci_RAG

# 推荐使用虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows PowerShell

pip install -r requirements.txt
```

> **注意**：Nougat 体积较大且需要 GPU，可先注释掉 requirements.txt 中的 `nougat-ocr` 行。Marker + Grobid 两路已足够使用。

---

### 第二步：配置环境变量

```bash
# 复制模板
copy .env.example .env
```

打开 `.env`，**必须修改**以下配置：

```dotenv
# ── 最重要：配置你的 LLM ──
LLM_API_KEY=sk-your-actual-api-key     # OpenAI / DeepSeek / 硅基流动等
LLM_BASE_URL=https://api.openai.com/v1 # 如用 DeepSeek: https://api.deepseek.com/v1
LLM_MODEL=gpt-4o-mini                  # 推荐性价比模型

# ── 设备配置 ──
EMBEDDING_DEVICE=cpu     # 有 GPU 改为 cuda
```

**推荐的国内 LLM 接入方案**（更快更便宜）：

| 服务 | BASE_URL | 推荐模型 |
|------|----------|---------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-72B-Instruct` |
| 月之暗面 | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |

---

### 第三步：启动基础设施

```bash
# 启动 PostgreSQL + Qdrant + Redis (+ Grobid)
docker-compose up -d

# 验证服务状态
docker-compose ps
```

预期输出（STATUS 均为 Up）：
```
sci_rag_postgres   Up   0.0.0.0:5432->5432/tcp
sci_rag_qdrant     Up   0.0.0.0:6333->6333/tcp
sci_rag_redis      Up   0.0.0.0:6379->6379/tcp
sci_rag_grobid     Up   0.0.0.0:8070->8070/tcp
```

---

### 第四步：初始化数据库

```bash
python scripts/init_db.py
```

---

### 第五步：启动所有服务（4个终端）

**终端 1 — FastAPI 后端**
```bash
python main.py
# → http://localhost:8000
# → API 文档: http://localhost:8000/docs
```

**终端 2 — Celery Worker（文件处理后台）**
```bash
celery -A worker.celery_app worker --loglevel=info -Q parsing,indexing --concurrency=2
```

**终端 3 — Streamlit 前端**
```bash
streamlit run frontend/app.py
# → http://localhost:8501
```

或者使用一键启动脚本（Windows，会自动打开多个窗口）：
```bash
python scripts/start.py all
```

---

## 📖 使用指南

### 上传论文

1. 打开浏览器访问 `http://localhost:8501`
2. 点击底部 **"📚 知识库管理"** 标签
3. 拖拽 PDF 文件到上传区域（支持多文件）
4. 点击 **"🚀 开始上传并处理"**
5. 观察进度条（实时显示处理阶段）

**处理阶段说明：**
```
上传 → 三路解析 (30%) → LLM融合+分类 (45%) → 
保存文件 (60%) → 向量化 (70%~95%) → 完成 (100%)
```

处理完成后，可在论文列表中看到：
- 自动提取的标题、作者、关键词
- 所属研究领域
- 中文摘要（LLM 生成）
- 向量块数量
- 解析质量分数

### 智能问答

1. 点击 **"💬 对话问答"** 标签
2. 在输入框中输入问题，支持：
   - 中英文混合提问
   - 具体技术问题：`"这些论文中用了哪些 Attention 机制的改进？"`
   - 对比分析：`"比较不同论文的实验结果"`
   - 文献综述：`"关于 Transformer 优化的主要方向有哪些？"`
3. 右侧面板显示检索来源（可追溯到具体论文和段落）

---

## 🏗️ 系统架构说明

### 数据处理管道

```
PDF 上传
    │
    ├──→ Nougat（擅长数学公式）
    ├──→ Marker（擅长布局保持）    ──→ LLM 仲裁融合
    └──→ Grobid（擅长结构化）
                │
                ↓
    LLM 领域分类 + 元数据提取
                │
                ↓
    保存到 data/raw/{domain}/
    写入 PostgreSQL 元数据表
                │
                ↓
    父子块分割（子块128t / 父块512t）
                │
                ↓
    BGE-M3 Embedding（稠密+稀疏双路）
                │
                ↓
    写入 Qdrant Collection（按领域分库）
```

### 检索生成管道

```
用户问题
    │
    ├──→ 查询重写（生成3个变体）
    ├──→ 领域路由（LLM判断搜哪个Collection）
    └──→ HyDE（生成假设答案，用于检索）
                │
                ↓
    BGE-M3 Embedding（假设答案）
                │
                ↓
    Qdrant 混合检索（Dense + Sparse，RRF融合）
                │
                ↓
    BGE Reranker 重排（Top-10 → Top-5）
                │
                ↓
    父子块扩展（子块 → 父块完整上下文）
                │
                ↓
    LLM 生成最终回答（流式输出）
```

---

## ⚙️ 高级配置

### 向量参数调优（`.env`）

```dotenv
CHILD_CHUNK_SIZE=128    # 子块大小（token）。越小越精准但上下文少
PARENT_CHUNK_SIZE=512   # 父块大小（token）。越大上下文越完整
CHUNK_OVERLAP=20        # 块间重叠，避免边界切断语义
TOP_K_RETRIEVE=10       # 初始检索返回数量
TOP_K_RERANK=5          # 重排后保留数量
DENSE_WEIGHT=0.7        # 稠密向量权重（语义相似度）
SPARSE_WEIGHT=0.3       # 稀疏向量权重（关键词匹配）
```

**调优建议：**
- 数学公式多的论文：增大 `CHILD_CHUNK_SIZE`（256~512）
- 问题较精确时：增大 `DENSE_WEIGHT`（0.8~0.9）
- 需要精确关键词匹配时：增大 `SPARSE_WEIGHT`（0.4~0.5）

### 使用本地 Ollama（离线模式）

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=ollama  # 随意填写
```

```bash
# 安装 Ollama 后拉取模型
ollama pull qwen2.5:7b
```

---

## 🧪 运行测试

```bash
# 核心逻辑单元测试（无需启动任何服务）
pytest tests/test_core.py -v

# 端到端管道测试（需要配置 LLM）
python scripts/test_pipeline.py path/to/your/paper.pdf

# API 接口测试（需要安装 aiosqlite）
pip install aiosqlite
pytest tests/test_api.py -v
```

---

## 🐛 常见问题

**Q: Grobid 解析失败，提示连接拒绝**
```bash
# 确认 Grobid 容器已启动
docker-compose up grobid -d
# 等待约 30 秒启动完成
curl http://localhost:8070/api/isalive
```

**Q: BGE 模型下载很慢**
```bash
# 使用 HuggingFace 镜像
$env:HF_ENDPOINT="https://hf-mirror.com"
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

**Q: Celery Worker 无法连接 Redis**
```bash
# 检查 Redis 是否在运行
docker-compose ps redis
# 测试 Redis 连接
redis-cli ping  # 返回 PONG 则正常
```

**Q: 上传 PDF 后一直卡在 "等待中"**
- 确认 Celery Worker 终端已启动
- 检查 Worker 终端是否有报错信息
- 确认 Redis 正常运行

**Q: LLM 调用失败**
- 检查 `.env` 中 `LLM_API_KEY` 是否正确
- 检查 `LLM_BASE_URL` 是否与你的 API 服务商匹配
- 网络问题可尝试使用国内服务商（DeepSeek / 硅基流动）

---

## 📦 依赖说明

| 包 | 用途 | 是否必须 |
|----|------|---------|
| `fastapi` | Web API 框架 | ✅ |
| `celery` | 异步任务队列 | ✅ |
| `qdrant-client` | 向量数据库客户端 | ✅ |
| `sqlalchemy` | ORM | ✅ |
| `FlagEmbedding` | BGE 向量模型 | ✅ |
| `streamlit` | 前端 UI | ✅ |
| `marker-pdf` | PDF 解析（Marker） | 推荐 |
| `nougat-ocr` | PDF 解析（Nougat，公式） | 可选 |
| `openai` | LLM 客户端 | ✅ |
| `torch` | 深度学习（BGE 依赖） | ✅ |

---

## 📄 License

MIT License — 自由使用，欢迎贡献改进。
