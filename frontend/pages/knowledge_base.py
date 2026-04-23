"""
知识库管理页面：文件上传 + 论文列表 + 处理进度
"""
import time
from typing import Optional

import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def _check_api() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _upload_paper(file) -> Optional[dict]:
    try:
        r = requests.post(
            f"{API_BASE}/upload/paper",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=30,
        )
        if r.status_code == 202:
            return r.json()
        else:
            st.error(f"上传失败: {r.json().get('detail', r.text)}")
            return None
    except requests.ConnectionError:
        st.error("⚠️ 无法连接到后端服务")
        return None


def _get_task_status(paper_id: str) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}/upload/status/{paper_id}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _get_papers(domain: Optional[str] = None) -> list:
    try:
        params = {}
        if domain and domain != "全部":
            params["domain"] = domain
        r = requests.get(f"{API_BASE}/papers/", params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def _get_domains() -> list[str]:
    try:
        r = requests.get(f"{API_BASE}/upload/domains", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


STATUS_LABELS = {
    "pending":     ("⏳ 等待中", "badge-processing"),
    "parsing":     ("🔍 解析中", "badge-processing"),
    "classifying": ("🧠 分类中", "badge-processing"),
    "indexing":    ("📦 向量化中", "badge-processing"),
    "completed":   ("✅ 已完成", "badge-completed"),
    "failed":      ("❌ 失败", "badge-failed"),
}

DOMAIN_LABELS = {
    "computer_science": "💻 计算机科学",
    "biology":          "🧬 生物学",
    "medicine":         "🏥 医学",
    "physics":          "⚛️ 物理学",
    "chemistry":        "🧪 化学",
    "materials_science":"🔩 材料科学",
    "environmental_science": "🌿 环境科学",
    "economics":        "📈 经济学",
    "psychology":       "🧠 心理学",
    "other":            "📄 其他",
}


def render_knowledge_base_page():
    # ── session state 初始化 ──────────────────────────────────
    if "uploading_tasks" not in st.session_state:
        st.session_state.uploading_tasks = {}  # paper_id -> task_info

    # ── API 状态检查 ──────────────────────────────────────────
    api_ok = _check_api()
    if not api_ok:
        st.warning(
            "⚠️ 后端服务未运行。请先执行：`python main.py`",
            icon="⚠️",
        )

    # ══════════════════════════════════════════════════════════
    #  区域 1：文件上传
    # ══════════════════════════════════════════════════════════
    st.markdown("### 📤 上传论文")

    upload_col, info_col = st.columns([2, 1])

    with upload_col:
        uploaded_files = st.file_uploader(
            "拖拽 PDF 文件到此处，或点击选择文件",
            type=["pdf"],
            accept_multiple_files=True,
            disabled=not api_ok,
            help="支持同时上传多篇论文，每篇最大 100 MB",
        )

        if uploaded_files:
            if st.button("🚀 开始上传并处理", type="primary", use_container_width=True):
                for f in uploaded_files:
                    with st.spinner(f"正在上传 {f.name}..."):
                        result = _upload_paper(f)
                        if result:
                            st.session_state.uploading_tasks[result["paper_id"]] = {
                                "paper_id": result["paper_id"],
                                "filename": result["filename"],
                                "task_id": result["task_id"],
                                "status": "pending",
                            }
                            st.success(f"✅ {f.name} 上传成功，后台处理中...")

    with info_col:
        st.markdown(
            """
            <div style="background:#f0f7ff; border-radius:10px; padding:16px; font-size:0.88rem;">
            <b>📋 处理流程</b><br><br>
            1️⃣ <b>三路解析</b><br>Nougat + Marker + Grobid<br><br>
            2️⃣ <b>LLM融合</b><br>比对提取结果，领域分类<br><br>
            3️⃣ <b>向量化</b><br>BGE-M3 父子块 Embedding<br><br>
            4️⃣ <b>入库</b><br>Qdrant 按领域 Collection 存储
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════
    #  区域 2：处理进度监控
    # ══════════════════════════════════════════════════════════
    active_tasks = {
        pid: info
        for pid, info in st.session_state.uploading_tasks.items()
        if info["status"] not in ("completed", "failed")
    }

    if active_tasks:
        st.markdown("---")
        st.markdown("### ⚙️ 处理进度")

        need_rerun = False
        for paper_id, task_info in list(active_tasks.items()):
            status_data = _get_task_status(paper_id)

            if status_data:
                current_status = status_data.get("status", "pending")
                progress = status_data.get("progress_percent") or 0
                message = status_data.get("progress_message") or "处理中..."
                title = status_data.get("title") or task_info["filename"]
                domain = status_data.get("domain", "")

                st.session_state.uploading_tasks[paper_id]["status"] = current_status

                label, _ = STATUS_LABELS.get(current_status, ("处理中", "badge-processing"))

                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{title[:60]}**")
                    
                    # 1. 如果失败了，展示醒目的红色终端报错
                    if current_status == "failed":
                        error_msg = status_data.get('error_message', '未知网络超时或后台崩溃')
                        log_html = f"""
                        <div class="terminal-log terminal-error">
                        [FATAL ERROR] 论文处理中断！<br>
                        > 状态：FAILED<br>
                        > 详细捕获：{error_msg}<br>
                        > 建议：请检查后台 Celery 窗口日志，或开启全局代理直连后重试。
                        </div>
                        """
                        st.markdown(log_html, unsafe_allow_html=True)
                        
                    # 2. 如果完成了，展示清爽的成功提示
                    elif current_status == "completed":
                        st.success(f"✅ 处理完成  |  最终判定领域: {DOMAIN_LABELS.get(domain, domain)}")
                        
                    # 3. 如果正在处理中，展示极客风格的动态日志终端
                    else:
                        safe_msg = message.replace('\n', '<br>')
                        log_html = f"""
                        <div class="terminal-log">
                        > [SYSTEM] 任务节点切换至：{current_status.upper()}<br>
                        > <span class="terminal-info">当前执行指令：{safe_msg}</span><br>
                        > <span style="color:#666;">后台全速运转中，请勿关闭页面...</span>
                        </div>
                        """
                        # 进度条保留，作为直观的进度指示，剥离原本自带的闪烁文字
                        st.progress(progress / 100) 
                        # 插入终端UI
                        st.markdown(log_html, unsafe_allow_html=True)
                        need_rerun = True
                        
                with col_b:
                    st.caption(f"Task ID: `{paper_id[:8]}...`")

        # 有任务在进行时自动刷新
        if need_rerun:
            time.sleep(2)
            st.rerun()

    # ══════════════════════════════════════════════════════════
    #  区域 3：论文列表
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📚 知识库论文列表")

    # 筛选控件
    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])

    with filter_col1:
        domains = _get_domains()
        domain_options = ["全部"] + [
            DOMAIN_LABELS.get(d, d) for d in domains
        ]
        domain_raw = ["全部"] + domains
        selected_domain_label = st.selectbox("按领域筛选", domain_options)
        selected_domain = domain_raw[domain_options.index(selected_domain_label)]

    with filter_col2:
        status_filter = st.selectbox(
            "按状态筛选",
            ["全部", "已完成", "处理中", "失败"],
        )
        status_map = {
            "全部": None, "已完成": "completed",
            "处理中": "parsing", "失败": "failed",
        }

    with filter_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 刷新列表", use_container_width=True):
            st.rerun()

    # 获取并显示论文列表
    domain_param = None if selected_domain == "全部" else selected_domain
    papers = _get_papers(domain=domain_param)

    # 状态过滤（客户端）
    status_val = status_map.get(status_filter)
    if status_val:
        papers = [p for p in papers if p["status"] == status_val]

    if not papers:
        st.info("暂无论文，请先上传 PDF 文件")
    else:
        st.caption(f"共 {len(papers)} 篇论文")

        for paper in papers:
            status_label, status_class = STATUS_LABELS.get(
                paper["status"], ("未知", "badge-processing")
            )
            domain_label = DOMAIN_LABELS.get(paper.get("domain", ""), paper.get("domain", ""))

            with st.expander(
                f"{paper.get('title') or paper['original_filename']}  "
                f"— {domain_label}",
                expanded=False,
            ):
                detail_col1, detail_col2 = st.columns([2, 1])

                with detail_col1:
                    if paper.get("abstract_summary_zh"):
                        st.markdown(f"**摘要**: {paper['abstract_summary_zh']}")

                    if paper.get("keywords"):
                        kw_tags = "  ".join(
                            f"`{kw}`" for kw in paper["keywords"][:8]
                        )
                        st.markdown(f"**关键词**: {kw_tags}")

                    if paper.get("methodology"):
                        st.markdown(f"**方法论**: {paper['methodology']}")

                with detail_col2:
                    st.markdown(f"**状态**: {status_label}")
                    if paper.get("journal_or_venue"):
                        st.markdown(f"**期刊/会议**: {paper['journal_or_venue']}")
                    if paper.get("year"):
                        st.markdown(f"**年份**: {paper['year']}")
                    if paper.get("authors"):
                        authors_str = ", ".join(paper["authors"][:3])
                        if len(paper["authors"]) > 3:
                            authors_str += " 等"
                        st.markdown(f"**作者**: {authors_str}")
                    st.markdown(f"**向量块**: {paper.get('chunk_count', 0)} 个")
                    if paper.get("parse_quality_score"):
                        score = paper["parse_quality_score"]
                        st.markdown(f"**解析质量**: {score:.2f}")
                    st.caption(f"ID: `{paper['id'][:8]}...`")
                    st.caption(f"上传: {paper['created_at'][:10]}")
