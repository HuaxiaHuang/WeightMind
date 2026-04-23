"""
Sci_RAG Streamlit 前端主入口 - 终极云端生产版
架构：Streamlit Frontend -> HTTP SSE -> FastAPI Backend
"""
import sys
import os
import uuid
import json
import streamlit as st
import requests
from dotenv import load_dotenv

# =====================================================================
# ✅ 强制加载环境变量，确保底层配置不被本地系统干扰
# =====================================================================
load_dotenv()

# =====================================================================
# ☢️ 核弹级网络隔离配置：彻底切断 Python 与 Windows 系统代理/梯子的联系
# =====================================================================
http_client = requests.Session()
http_client.trust_env = False  # 绝对不读取任何系统代理环境变量！
http_client.proxies = {
    "http": None,
    "https": None,
}

API_URL = "http://127.0.0.1:8000" 
CHAT_ENDPOINT = f"{API_URL}/chat/stream"
UPLOAD_ENDPOINT = f"{API_URL}/upload/paper" 
# =====================================================================

st.set_page_config(
    page_title="WeightMind | 科研级 RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 深度 CSS 定制 ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {display: none !important;} 
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stApp { background-color: #1E1E1E; color: #ECECEC; }
    [data-testid="stSidebar"] { background-color: #171717; border-right: 1px solid #2F2F2F; }
    .sidebar-logo { font-size: 1.8rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px; margin-bottom: 0.2rem; }
    .sidebar-subtitle { color: #888888; font-size: 0.8rem; margin-bottom: 2rem; }
    [data-testid="stSidebar"] div.stButton > button { width: 100%; background-color: transparent; border: 1px solid transparent; color: #ECECEC; border-radius: 8px; text-align: left; transition: all 0.2s; }
    [data-testid="stSidebar"] div.stButton > button:hover { background-color: #2A2A2A; }
    
    .source-card {
        background: #252526;
        border: 1px solid #3E3E42;
        border-left: 4px solid #10A37F;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.85rem;
        color: #D4D4D4;
        line-height: 1.6;
    }
    .source-title { color: #10A37F; font-weight: 600; margin-bottom: 6px; font-size: 0.9rem; }
    .source-meta { font-size: 0.75rem; color: #858585; margin-bottom: 8px; font-style: italic; }
    
    [data-testid="stChatInput"] { border-radius: 20px !important; border: 1px solid #3E3E42 !important; background-color: #2D2D2D !important; }
    .welcome-text { text-align: center; font-size: 2.2rem; font-weight: 600; color: #ECECEC; margin-top: 20vh; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 状态机管理 ─────────────────────────────────────────────────
if "current_view" not in st.session_state:
    st.session_state.current_view = "chat"
if "sessions" not in st.session_state:
    default_id = str(uuid.uuid4())
    st.session_state.sessions = {default_id: {"title": "新会话", "messages": []}}
    st.session_state.current_session_id = default_id

# ── 左侧边栏 ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">WeightMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">惟明 · 专注科学文献的深度推理</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ 新建会话", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.sessions[new_id] = {"title": "新会话", "messages": []}
            st.session_state.current_session_id = new_id
            st.session_state.current_view = "chat"
            st.rerun()
    with col2:
        if st.button("📁 知识库管理", use_container_width=True):
            st.session_state.current_view = "upload"
            st.rerun()

    st.markdown("<hr style='margin: 1rem 0; border-color: #333;'>", unsafe_allow_html=True)
    st.caption("💬 历史会话")
    for session_id, session_data in reversed(st.session_state.sessions.items()):
        is_active = (session_id == st.session_state.current_session_id and st.session_state.current_view == "chat")
        btn_label = f"▶ {session_data['title']}" if is_active else f"💬 {session_data['title']}"
        if st.button(btn_label, key=session_id):
            st.session_state.current_session_id = session_id
            st.session_state.current_view = "chat"
            st.rerun()

# ── 主工作区路由 ───────────────────────────────────────────────
if st.session_state.current_view == "upload":
    st.markdown("<br><br><h2 style='text-align:center;'>上传文献并解析入库</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("支持多 PDF 批量上传（将触发 Celery 异步解析与 BGE-M3 向量化）", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("开始上传并构建索引", type="primary"):
            with st.spinner(f"🚀 正在将 {len(uploaded_files)} 篇文档依次推送至队列..."):
                success_count = 0
                for file in uploaded_files:
                    try:
                        # ✨ 核心：每次只封装 1 个文件，发送 1 次请求，触发 Celery 队列
                        files_payload = {"file": (file.name, file.getvalue(), "application/pdf")}
                        response = http_client.post(UPLOAD_ENDPOINT, files=files_payload, timeout=120)
                        response.raise_for_status() 
                        success_count += 1
                    except Exception as e:
                        st.error(f"🚨 文件 {file.name} 提交失败：`{repr(e)}`")
                
                if success_count > 0:
                    st.success(f"🎉 成功向队列提交 {success_count} 篇文档！快去看看 Celery 终端是不是收到了一长串任务！")

    # ✨ 新增：展示云端已有的文献知识库（解决“前端失忆”问题）
    st.markdown("<br><hr style='border-color: #333;'>", unsafe_allow_html=True)
    st.markdown("### 📚 云端知识库文献列表")
    
    try:
        # 向后端请求已上传的论文列表
        papers_resp = http_client.get(f"{API_URL}/papers/", timeout=10)
        if papers_resp.status_code == 200:
            papers = papers_resp.json()
            if papers:
                for p in papers:
                    # 根据状态显示不同的 Emoji
                    status_emoji = "✅" if p['status'] == "completed" else "⏳"
                    # 显示文件名、领域和状态
                    display_name = p['title'] if p['title'] else p['original_filename']
                    st.markdown(f"{status_emoji} **{display_name}** <span style='color:#888; font-size:0.8rem;'>(领域: {p['domain']} | 状态: {p['status']})</span>", unsafe_allow_html=True)
            else:
                st.info("知识库空空如也，快去上传第一篇文献吧！")
    except Exception as e:
        st.warning("暂时无法连接到后端获取历史文献列表，请检查 FastAPI 服务是否启动。")
                    
elif st.session_state.current_view == "chat":
    current_session = st.session_state.sessions[st.session_state.current_session_id]

    if not current_session["messages"]:
        st.markdown('<div class="welcome-text">开始提问，探索你的论文知识库</div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#888;'>自动查询重写 · HyDE 假设生成 · 混合语义检索 · 精准溯源</p>", unsafe_allow_html=True)
    else:
        for msg in current_session["messages"]:
            avatar = "🧑‍💻" if msg["role"] == "user" else "🔬"
            with st.chat_message(msg["role"], avatar=avatar):
                # ✨ 如果有思考日志，用折叠框渲染出来
                if msg.get("logs"):
                    with st.status("⚙️ 惟明引擎 思考与检索路径...", expanded=False):
                        st.markdown(msg["logs"])
                
                # ✨ 渲染精美的引用卡片
                if msg.get("references"):
                    with st.expander("📎 点击查看溯源文献及片段"):
                        st.markdown(msg["references"], unsafe_allow_html=True)
                
                st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("例如：基于文献探讨，赤潮爆发的关键驱动因素是什么？"):
        current_session["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
            
        if len(current_session["messages"]) == 1:
            current_session["title"] = prompt[:12] + "..."

        with st.chat_message("assistant", avatar="🔬"):
            
            status_placeholder = st.status("⚙️ 惟明引擎 思考与检索路径...", expanded=True)
            message_placeholder = st.empty()
            
            full_answer = ""
            references_html = ""
            logs_history = ""  # ✨ 用于永久保存思考日志的变量
            has_error = False 
            
            try:
                payload = {"query": prompt}
                if len(current_session["messages"]) > 1:
                    payload["session_id"] = st.session_state.current_session_id
                
                # print(f"👉 准备向后端发送请求，地址: {CHAT_ENDPOINT}, payload: {payload}")

                response = http_client.post(
                    CHAT_ENDPOINT, 
                    json=payload, 
                    stream=True, 
                    timeout=300
                )
                response.raise_for_status() 
                
                stream_generator = response.iter_lines()

                for line in stream_generator:
                    if line:
                        decoded_line = line.decode('utf-8').removeprefix("data: ")
                        if not decoded_line:
                            continue
                            
                        data = json.loads(decoded_line)
                        msg_type = data.get("type")
                        
                        if msg_type == "log":
                            log_text = f"➤ {data['content']}"
                            status_placeholder.write(log_text)
                            logs_history += log_text + "\n\n"  # ✨ 累加日志
                            
                        elif msg_type == "meta":
                            real_session_id = data.get("session_id")
                            if real_session_id and real_session_id != st.session_state.current_session_id:
                                old_id = st.session_state.current_session_id
                                st.session_state.sessions[real_session_id] = st.session_state.sessions.pop(old_id)
                                st.session_state.current_session_id = real_session_id
                                
                        elif msg_type == "references":
                            refs = data.get("content", [])
                            status_placeholder.write(f"✅ 检索管线执行完毕！成功匹配到 {len(refs)} 个高相关度父文档块。")
                            status_placeholder.update(label="检索与思考完成", state="complete", expanded=False)
                            
                            for ref in refs:
                                # ✨ 获取 score，如果没有则默认为 0
                                score_val = ref.get('score', 0)
                                
                                references_html += f"""
                                <div class="source-card">
                                    <div class="source-title">📄 参考来源 [{ref.get('id', '*')}]：{ref.get('title', '未知')} <span style="color:#E2B93B; font-size:0.8rem;">(相关度: {score_val:.3f})</span></div>
                                    <div class="source-meta">来源：{ref.get('domain', 'Unknown')}</div>
                                    {ref.get('text_preview', '')[:200]}...
                                </div>
                                """
                            st.markdown("### 📎 溯源文献")
                            st.markdown(references_html, unsafe_allow_html=True)
                            st.markdown("---")
                            
                        elif msg_type == "token": 
                            full_answer += data["content"]
                            message_placeholder.markdown(full_answer + "▌")
                                        
            except requests.exceptions.ConnectionError as e:
                has_error = True
                status_placeholder.update(label="🔴 物理网络连接断开", state="error", expanded=True)
                full_answer = f"🚨 **无法连接到 FastAPI 后端**\n\n**真实底层报错代码如下：**\n\n`{repr(e)}`"
                message_placeholder.error(full_answer)
                
            except requests.exceptions.HTTPError as e:
                has_error = True
                status_placeholder.update(label="🔴 后端服务报错", state="error", expanded=True)
                try:
                    error_detail = response.json().get('detail', response.text)
                except:
                    error_detail = response.text
                full_answer = f"🚨 **后端 API 崩溃**：\n\n状态码：{response.status_code}\n\n```python\n{error_detail}\n```"
                message_placeholder.error(full_answer)
                
            except Exception as e:
                has_error = True
                status_placeholder.update(label="🔴 检索过程中发生异常", state="error", expanded=True)
                full_answer = f"🚨 **系统异常/响应超时**：\n\n`{repr(e)}`"
                message_placeholder.error(full_answer)
            
            if not has_error:
                message_placeholder.markdown(full_answer)
            
            # ✨ 核心修复：把日志和文献格式彻底存入 session_state，实现历史记录持久化渲染
            current_session["messages"].append({
                "role": "assistant", 
                "content": full_answer,
                "references": references_html if (references_html and not has_error) else None,
                "logs": logs_history if logs_history else None  
            })
            
        st.rerun()