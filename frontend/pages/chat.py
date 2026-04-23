"""
对话页面：流式 RAG 问答
"""
import json
import time
from typing import Optional

import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def _init_session():
    """初始化 Streamlit session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "last_domains" not in st.session_state:
        st.session_state.last_domains = []


def _render_message(role: str, content: str):
    """渲染单条消息"""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)


def _stream_chat(query: str, session_id: Optional[str]) -> tuple[str, list, list, str]:
    """
    调用流式接口，返回 (完整回答, 来源列表, 领域列表, session_id)
    """
    url = f"{API_BASE}/chat/stream"
    payload = {"query": query, "session_id": session_id, "stream": True}
    
    full_answer = ""
    sources = []
    domains = []
    new_session_id = session_id
    
    # 🌟 魔法核心：使用 st.status 实现自动折叠的思考框
    answer_placeholder = st.empty()  # 负责渲染最终黑字答案
    new_session_id = session_id
    
    # 建立一个状态框，初始状态为展开 (expanded=True)
    with st.status("⚙️ 惟明引擎 思考与检索路径...", expanded=True) as status_box:
        status_box.write("➤ 🚀 已接收问题，正在唤醒 RAG 检索管线...")
        
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    data_str = line[6:].decode("utf-8")
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    
                    msg_type = data.get("type")
                    
                    # 💡 接收思考步骤，直接写入 status_box
                    if msg_type == "thought" or msg_type == "step":
                        status_box.write(f"➤ {data.get('content', '')}")
                    
                    # 收到 meta，说明检索结束，准备生成答案
                    elif msg_type == "meta":
                        new_session_id = data.get("session_id", session_id)
                        sources = data.get("sources", [])
                        domains = data.get("domains_searched", [])
                        
                        status_box.write(f"➤ 📦 检索完成！成功在 [{', '.join(domains)}] 匹配到 {len(sources)} 个高相关度文档块。")
                        status_box.write("➤ ✨ 正在将知识融合，生成最终回答...")
                        
                        # 核心动作：检索思考结束，将状态框打勾并自动折叠！
                        status_box.update(label="⚙️ 检索与思考完成 (点击展开查看明细)", state="complete", expanded=False)
                    
                    # 打印大模型生成的字符
                    elif msg_type == "token":
                        full_answer += data.get("content", "")
                        answer_placeholder.markdown(full_answer + "▌")  
                    
                    # 结束
                    elif msg_type == "done":
                        answer_placeholder.markdown(full_answer)
                        break
    
    return full_answer, sources, domains, new_session_id


def render_chat_page():
    _init_session()
    
    col_main, col_sidebar = st.columns([3, 1])
    
    with col_main:
        # ── 显示历史消息 ──────────────────────────────────────
        for msg in st.session_state.messages:
            _render_message(msg["role"], msg["content"])
        
        # ── 欢迎提示（空对话时显示） ───────────────────────────
        if not st.session_state.messages:
            st.markdown(
                """
                <div style="text-align:center; padding: 3rem; color: #9e9e9e;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                    <div style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">
                        开始提问，探索你的论文知识库
                    </div>
                    <div style="font-size: 0.9rem;">
                        支持中英文提问 · 自动领域路由 · 引用来源可追溯
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # ── 输入框 ────────────────────────────────────────────
        query = st.chat_input("输入你的问题，例如：赤潮主要受什么影响？")
        
        if query:
            # 显示用户消息
            _render_message("user", query)
            st.session_state.messages.append({"role": "user", "content": query})
            
            # 流式调用
            try:
                with st.chat_message("assistant", avatar="🤖"):
                    answer, sources, domains, new_sid = _stream_chat(
                        query, st.session_state.session_id
                    )
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.session_id = new_sid
                st.session_state.last_sources = sources
                st.session_state.last_domains = domains
                
                st.rerun()
                
            except requests.exceptions.ReadTimeout:
                st.error("⚠️ 请求超时 (等待超过300秒)。可能是底层 BGE 重排模型正在初始化，请稍后重试。")
            except requests.ConnectionError:
                st.error("⚠️ 无法连接到后端服务，请确认 FastAPI 已启动（python main.py）")
            except Exception as e:
                st.error(f"⚠️ 请求出错: {str(e)}")
    
    with col_sidebar:
        st.markdown("### 📎 检索信息")
        
        # 显示搜索的领域
        if st.session_state.last_domains:
            st.markdown("**搜索领域：**")
            for d in st.session_state.last_domains:
                st.markdown(f"- `{d}`")
        
        # 显示引用来源
        if st.session_state.last_sources:
            st.markdown("**引用来源：**")
            for i, src in enumerate(st.session_state.last_sources, 1):
                with st.expander(f"[{i}] {src.get('title', '未知论文')[:30]}..."):
                    st.markdown(f"**领域**: `{src.get('domain', 'N/A')}`")
                    st.markdown(f"**相关度**: `{src.get('score', 0):.3f}`")
                    st.markdown(f"**摘要**: {src.get('text_preview', '')}...")
        else:
            st.markdown(
                '<p style="color:#9e9e9e; font-size:0.85rem;">提问后会显示引用来源</p>',
                unsafe_allow_html=True,
            )
        
        st.markdown("---")
        
        # 清空对话
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.last_sources = []
            st.session_state.last_domains = []
            st.rerun()
        
        # 新建会话
        if st.button("➕ 新建会话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()