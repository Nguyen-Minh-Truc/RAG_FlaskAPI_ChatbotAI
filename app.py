"""
RAG & Co-RAG Explorer — Streamlit UI
Connects to the Flask REST API defined in routes.py
"""

import time
import html
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Kham pha RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
def get_api_base() -> str:
    # Keep startup resilient without requiring .streamlit/secrets.toml.
    return os.getenv("API_BASE") or "http://127.0.0.1:5000"


def discover_api_base() -> str:
    """Try common local API addresses and return the first healthy one."""
    env_base = os.getenv("API_BASE")
    candidates = []
    if env_base:
        candidates.append(env_base)

    candidates.extend([
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ])

    seen = set()
    for base in candidates:
        clean_base = base.rstrip("/")
        if clean_base in seen:
            continue
        seen.add(clean_base)
        try:
            r = requests.get(f"{clean_base}/api/health", timeout=1.5)
            if r.status_code == 200:
                return clean_base
        except Exception:
            continue

    return (env_base or "http://127.0.0.1:5000").rstrip("/")


API_BASE = get_api_base()

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

/* ── Root palette ────────────────────────── */
:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --border:    #23283a;
    --muted:     #3a4060;
    --text:      #e2e6f0;
    --subtext:   #7a82a0;
    --accent1:   #4f8ef7;   /* blue  – RAG  */
    --accent2:   #34d39e;   /* teal  – CoRAG */
    --accent3:   #f76b8a;   /* rose  – delete/warn */
    --accent1bg: rgba(79,142,247,.10);
    --accent2bg: rgba(52,211,158,.10);
    --radius:    12px;
    --mono:      'DM Mono', monospace;
    --sans:      'Sora', sans-serif;
}

/* ── Global resets ───────────────────────── */
html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: var(--sans);
}

/* ── Hide Streamlit chrome ───────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 4px; }

/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--sans) !important; }

/* ── Sidebar section title ───────────────── */
.sidebar-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--subtext);
    margin: 1.4rem 0 .5rem;
    padding-bottom: .35rem;
    border-bottom: 1px solid var(--border);
}

/* ── Upload zone ─────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg) !important;
    border: 1.5px dashed var(--muted) !important;
    border-radius: var(--radius) !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent1) !important;
}

/* ── Buttons ─────────────────────────────── */
.stButton > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all .18s ease !important;
    height: 38px !important;
}
.stButton > button:hover {
    border-color: var(--accent1) !important;
    color: var(--accent1) !important;
    box-shadow: 0 0 0 3px var(--accent1bg) !important;
}

/* Primary button (use st.button(type="primary")) */
.stButton > button[kind="primary"] {
    background: var(--accent1) !important;
    color: #fff !important;
    border-color: var(--accent1) !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #6fa3ff !important;
    color: #fff !important;
}

/* ── Text inputs ─────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent1) !important;
    box-shadow: 0 0 0 3px var(--accent1bg) !important;
}

/* ── Selectbox ───────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Slider ──────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background: var(--accent2) !important;
    border-color: var(--accent2) !important;
}

/* ── Page title bar ──────────────────────── */
.page-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 28px 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.page-header .logo-badge {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.page-header h1 {
    margin: 0; padding: 0;
    font-size: 22px; font-weight: 700;
    letter-spacing: -.02em; color: var(--text);
}
.page-header span {
    font-size: 13px; color: var(--subtext); font-weight: 400;
}

/* ── Status pill ─────────────────────────── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px;
}
.status-ok  { background: rgba(52,211,158,.15); color: var(--accent2); }
.status-err { background: rgba(247,107,138,.15); color: var(--accent3); }
.status-off { background: rgba(122,130,160,.15); color: var(--subtext); }

/* ── Answer cards ────────────────────────── */
.answer-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px;
    line-height: 1.7;
    font-size: 14px;
    position: relative;
    overflow: hidden;
}
.answer-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
}
.answer-card.rag::before  { background: var(--accent1); }
.answer-card.corag::before { background: var(--accent2); }

.card-label {
    font-size: 10px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.card-label.rag   { color: var(--accent1); }
.card-label.corag { color: var(--accent2); }

/* ── Chat history ────────────────────────── */
.chat-turn {
    margin-bottom: 18px;
    animation: fadeIn .3s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }

.chat-q {
    display: flex; gap: 10px; align-items: flex-start;
    margin-bottom: 8px;
}
.chat-q .avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--muted); display: flex; align-items: center;
    justify-content: center; font-size: 13px; flex-shrink: 0; margin-top:2px;
}
.chat-q .bubble {
    background: var(--muted);
    border-radius: 0 10px 10px 10px;
    padding: 9px 14px; font-size: 13.5px; line-height: 1.55;
    max-width: 90%;
}
.chat-a {
    display: flex; gap: 10px; align-items: flex-start;
    flex-direction: row-reverse;
}
.chat-a .avatar {
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; flex-shrink: 0; margin-top:2px;
}
.chat-a .bubble {
    background: var(--accent1bg);
    border: 1px solid var(--accent1);
    border-radius: 10px 0 10px 10px;
    padding: 9px 14px; font-size: 13.5px; line-height: 1.55;
    max-width: 90%;
}

/* ── CoRAG trace accordion ───────────────── */
.trace-container {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
}
.trace-step {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 8px 0; border-bottom: 1px solid var(--border);
    font-size: 12.5px;
}
.trace-step:last-child { border-bottom: none; }
.trace-step .step-num {
    background: var(--accent2bg);
    color: var(--accent2);
    border-radius: 50%;
    width: 22px; height: 22px; min-width: 22px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; font-family: var(--mono);
}
.trace-step .step-body { flex:1; }
.trace-step .step-label {
    font-size: 10px; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--subtext); margin-bottom: 3px;
}

/* ── Conversation list ───────────────────── */
.conv-item {
    padding: 9px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-bottom: 6px;
    cursor: pointer;
    transition: all .15s;
    background: var(--bg);
    font-size: 13px;
}
.conv-item:hover { border-color: var(--accent1); background: var(--accent1bg); }
.conv-item.active { border-color: var(--accent1); background: var(--accent1bg); }
.conv-item .conv-title { font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conv-item .conv-meta  { font-size: 11px; color: var(--subtext); margin-top: 2px; }

/* ── Metric chips ────────────────────────── */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
.metric-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    min-width: 100px;
    text-align: center;
}
.metric-chip .mval { font-size: 22px; font-weight: 700; font-family: var(--mono); color: var(--text); }
.metric-chip .mlbl { font-size: 10px; color: var(--subtext); text-transform: uppercase; letter-spacing: .08em; margin-top:2px; }

/* ── Divider ─────────────────────────────── */
.hline { border:none; border-top: 1px solid var(--border); margin: 18px 0; }

/* ── Info box ────────────────────────────── */
.info-box {
    background: var(--accent1bg);
    border: 1px solid var(--accent1);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--text);
    margin-bottom: 14px;
}
.warn-box {
    background: rgba(247,107,138,.08);
    border: 1px solid var(--accent3);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--accent3);
}

/* ── Expander ────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--sans) !important;
    font-size: 13px !important;
    color: var(--text) !important;
}

/* ── Spinner override ────────────────────── */
[data-testid="stSpinner"] { color: var(--accent1) !important; }

/* ── Tab row ─────────────────────────────── */
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--subtext) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: var(--accent1) !important;
}

/* ── Scrollable chat box ─────────────────── */
.chat-scroll {
    max-height: 440px;
    overflow-y: auto;
    padding-right: 4px;
}

/* ── No-data placeholder ─────────────────── */
.empty-state {
    text-align: center;
    padding: 48px 20px;
    color: var(--subtext);
}
.empty-state .icon { font-size: 40px; margin-bottom: 12px; }
.empty-state p { font-size: 14px; margin:0; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {question, rag_answer, corag_answer}
if "api_status" not in st.session_state:
    st.session_state.api_status = None   # True/False/None
if "api_base" not in st.session_state:
    st.session_state.api_base = discover_api_base()

if "question_box_version" not in st.session_state:
    st.session_state.question_box_version = 0

# ─────────────────────────────────────────────
#  HELPER — API CALLS
# ─────────────────────────────────────────────
def api(method: str, path: str, timeout: int = 60, **kwargs):
    base = st.session_state.get("api_base") or API_BASE
    url = f"{base}{path}"
    try:
        r = getattr(requests, method)(url, timeout=timeout, **kwargs)
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        return {"error": "Khong the ket noi toi may chu API."}, 503
    except requests.exceptions.ReadTimeout:
        return {
            "error": (
                "Yeu cau qua thoi gian cho. Model local dang phan hoi cham, "
                "vui long thu lai hoac giam do dai cau hoi/tai lieu."
            )
        }, 504
    except Exception as exc:
        return {"error": str(exc)}, 500


def esc(value):
    return html.escape(str(value))


def check_health():
    data, code = api("get", "/api/health", timeout=5)
    st.session_state.api_status = (code == 200)
    return st.session_state.api_status


def list_conversations():
    data, code = api("get", "/api/conversations", timeout=10)
    if code == 200:
        return data.get("data", [])
    return []


def get_conversation(cid: str):
    data, code = api("get", f"/api/conversations/{cid}", timeout=10)
    if code == 200:
        return data.get("data", {})
    return {}


def delete_conversation(cid: str):
    _, code = api("delete", f"/api/conversations/{cid}", timeout=10)
    return code == 200


def delete_all_conversations():
    _, code = api("delete", "/api/conversations", timeout=10)
    return code == 200


def upload_file(file_bytes, filename: str):
    data, code = api(
        "post", "/api/upload",
        timeout=180,
        files={"file": (filename, file_bytes, "application/octet-stream")}
    )
    return data, code


def ask_question(question: str, conv_id: str, corag_rounds: int):
    data, code = api(
        "post", "/api/ask",
        timeout=300,
        json={"question": question, "conversation_id": conv_id, "corag_rounds": corag_rounds},
    )
    return data, code


DEFAULT_CORAG_ROUNDS = 1


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # ── Logo / app name ──────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;padding:8px 0 6px;">
            <div style="
                width:36px;height:36px;border-radius:10px;flex-shrink:0;
                background:linear-gradient(135deg,#4f8ef7,#34d39e);
                display:flex;align-items:center;justify-content:center;font-size:18px;">
                🔍
            </div>
            <div>
                <div style="font-weight:700;font-size:15px;letter-spacing:-.02em;">Kham pha RAG</div>
                <div style="font-size:11px;color:var(--subtext);">Tri tue tai lieu</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── API status ───────────────────────────
    st.markdown('<div class="sidebar-section">🔌 Trang thai may chu</div>', unsafe_allow_html=True)
    current_api_base = st.text_input(
        "Dia chi API",
        value=st.session_state.api_base,
        key="api_base_input",
        help="Vi du: http://127.0.0.1:5000",
    ).strip()
    if current_api_base and current_api_base.rstrip("/") != st.session_state.api_base:
        st.session_state.api_base = current_api_base.rstrip("/")
        st.session_state.api_status = None

    col_s, col_b = st.columns([3, 2])
    with col_s:
        if st.session_state.api_status is True:
            st.markdown('<span class="status-pill status-ok">● Dang hoat dong</span>', unsafe_allow_html=True)
        elif st.session_state.api_status is False:
            st.markdown('<span class="status-pill status-err">● Mat ket noi</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill status-off">● Chua kiem tra</span>', unsafe_allow_html=True)
    with col_b:
        if st.button("Kiem tra", key="btn_health", use_container_width=True):
            ok = check_health()
            if ok:
                st.toast(f"✅ API hoat dong on dinh: {st.session_state.api_base}", icon="✅")
            else:
                st.toast(f"❌ Khong the ket noi API: {st.session_state.api_base}", icon="❌")

    # ── Upload ───────────────────────────────
    st.markdown('<div class="sidebar-section">📂 Tai tai lieu</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Tha file vao day",
        type=["pdf", "doc", "docx"],
        label_visibility="collapsed",
    )
    do_upload = st.button("⬆ Nap tai lieu", key="btn_upload", use_container_width=True, type="primary")

    if do_upload:
        if uploaded is None:
            st.warning("Vui long chon file truoc.")
        else:
            with st.spinner("Dang nap du lieu..."):
                data, code = upload_file(uploaded.read(), uploaded.name)
            if code == 200:
                info = data.get("data", {})
                st.session_state.active_conv_id = info.get("conversation_id")
                st.session_state.chat_history = []
                st.toast(
                    f"✅ Da lap chi muc {info.get('chunks', '?')} doan tu {info.get('documents', '?')} trang.",
                    icon="📄",
                )
                st.rerun()
            else:
                detail = (data.get("error") or {}).get("detail") if isinstance(data.get("error"), dict) else None
                err = detail or data.get("message") or data.get("error", "Loi khong xac dinh")
                st.error(f"Tai file that bai: {err}")

    # ── Conversations ────────────────────────
    st.markdown('<div class="sidebar-section">💬 Hoi thoai</div>', unsafe_allow_html=True)
    convs = list_conversations()

    if not convs:
        st.markdown(
            "<div style='font-size:12px;color:var(--subtext);padding:4px 0;'>Chua co hoi thoai nao.</div>",
            unsafe_allow_html=True,
        )
    else:
        # Auto-select first
        if st.session_state.active_conv_id is None:
            st.session_state.active_conv_id = str(convs[0].get("conversation_id"))

        for conv in convs:
            cid   = str(conv.get("conversation_id", ""))
            title = conv.get("filename") or conv.get("title") or f"Hoi thoai {cid[:6]}"
            turns = conv.get("turn_count", conv.get("turns", 0))
            is_active = (cid == st.session_state.active_conv_id)
            badge = "active" if is_active else ""

            cols = st.columns([5, 1])
            with cols[0]:
                if st.button(
                    f"📄 {title[:28]}{'…' if len(title)>28 else ''}\n_{turns} luot hoi dap_",
                    key=f"conv_{cid}",
                    use_container_width=True,
                ):
                    st.session_state.active_conv_id = cid
                    detail = get_conversation(cid)
                    st.session_state.chat_history = [
                        {
                            "question": t.get("question", ""),
                            "rag_answer": t.get("answer", ""),
                            "corag_answer": t.get("corag_answer", ""),
                        }
                        for t in detail.get("turns", [])
                    ]
                    st.rerun()
            with cols[1]:
                if st.button("✕", key=f"del_{cid}", use_container_width=True):
                    if delete_conversation(cid):
                        if st.session_state.active_conv_id == cid:
                            st.session_state.active_conv_id = None
                            st.session_state.chat_history = []
                        st.toast("Da xoa hoi thoai.", icon="🗑")
                        st.rerun()

        st.markdown("<hr class='hline'>", unsafe_allow_html=True)
        if st.button("🗑 Xoa tat ca hoi thoai", key="del_all", use_container_width=True):
            if delete_all_conversations():
                st.session_state.active_conv_id = None
                st.session_state.chat_history = []
                st.toast("Da xoa toan bo hoi thoai.", icon="🗑")
                st.rerun()


# ─────────────────────────────────────────────
#  MAIN CONTENT AREA
# ─────────────────────────────────────────────

# ── Page header ──────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <div class="logo-badge">🔍</div>
        <div>
            <h1>Kham pha RAG</h1>
            <span>Retrieval-Augmented Generation · Nen tang hoi dap tai lieu</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs: Chat | History ──────────────────────
tab_chat, tab_history = st.tabs(["💬 Tro chuyen", "📜 Lich su hoi thoai"])

# ════════════════════════════════════════
#  TAB 1 — CHAT
# ════════════════════════════════════════
with tab_chat:
    convs_fresh = list_conversations()

    # No document uploaded yet
    if not convs_fresh and st.session_state.active_conv_id is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">📂</div>
                    <p style="font-weight:600;font-size:16px;color:var(--text);margin-bottom:6px;">Chua co tai lieu</p>
                    <p>Hay tai PDF hoac DOCX o thanh ben trai de bat dau.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ── Active conversation info ──────────────
        if st.session_state.active_conv_id:
            active_conv = next(
                (c for c in convs_fresh if str(c.get("conversation_id")) == st.session_state.active_conv_id),
                None,
            )
            if active_conv:
                fname = active_conv.get("filename") or active_conv.get("title", "Tai lieu")
                turns = active_conv.get("turn_count", len(st.session_state.chat_history))
                safe_fname = esc(fname)
                safe_conv = esc(st.session_state.active_conv_id)
                st.markdown(
                    f"""
                    <div class="info-box">
                        📄 <strong>{safe_fname}</strong>
                        &nbsp;·&nbsp; Ma hoi thoai <code>{safe_conv[:8]}…</code>
                        &nbsp;·&nbsp; {turns} luot hoi dap
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Chat history display ──────────────────
        if st.session_state.chat_history:
            st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
            for item in st.session_state.chat_history:
                safe_question = esc(item.get("question", ""))
                rag_answer = item.get("rag_answer")
                corag_answer = item.get("corag_answer")
                safe_rag = esc(rag_answer) if rag_answer else '<em style="color:var(--subtext)">—</em>'
                safe_corag = esc(corag_answer) if corag_answer else '<em style="color:var(--subtext)">—</em>'
                st.markdown(
                    f"""
                    <div class="chat-turn">
                        <div class="chat-q">
                            <div class="avatar">👤</div>
                            <div class="bubble">{safe_question}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("📬 Xem cau tra loi", expanded=(item == st.session_state.chat_history[-1])):
                    c1, c2 = st.columns(2, gap="medium")
                    with c1:
                        st.markdown(
                            f"""
                            <div class="answer-card rag">
                                <div class="card-label rag">🔵 RAG tieu chuan</div>
                                {safe_rag}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f"""
                            <div class="answer-card corag">
                                <div class="card-label corag">🟢 Co-RAG (lap)</div>
                                {safe_corag}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="icon">💭</div>
                    <p>Hay dat cau hoi dau tien ben duoi!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Question input ────────────────────────
        st.markdown("<hr class='hline'>", unsafe_allow_html=True)
        q_col, btn_col = st.columns([8, 2], gap="small")
        with q_col:
            question_input_key = f"question_input_{st.session_state.question_box_version}"
            question = st.text_input(
                "question",
                placeholder="Hoi bat ky dieu gi ve tai lieu cua ban...",
                label_visibility="collapsed",
                key=question_input_key,
            )
        with btn_col:
            send = st.button("Gui ➤", key="btn_send", use_container_width=True, type="primary")

        if send:
            if not question.strip():
                st.warning("Vui long nhap cau hoi.")
            elif not st.session_state.active_conv_id:
                st.error("Không co hoi thoai dang hoat dong. Hay tai tai lieu truoc.")
            else:
                with st.spinner("Đang suy nghi..."):
                    resp, code = ask_question(
                        question.strip(),
                        st.session_state.active_conv_id,
                        DEFAULT_CORAG_ROUNDS,
                    )

                if code == 200:
                    d = resp.get("data", {})
                    st.session_state.chat_history.append({
                        "question":     question.strip(),
                        "rag_answer":   d.get("rag_answer", ""),
                        "corag_answer": d.get("corag_answer", ""),
                    })
                    st.session_state.question_box_version += 1
                    st.rerun()
                else:
                    msg = resp.get("message") or resp.get("error", "Loi khong xac dinh")
                    st.error(f"Loi {code}: {msg}")


# ════════════════════════════════════════
#  TAB 2 — CONVERSATION HISTORY (full view)
# ════════════════════════════════════════
with tab_history:
    convs_list = list_conversations()

    if not convs_list:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">📭</div>
                <p>Khong tim thay hoi thoai nao.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ── Metrics ──────────────────────────────
        total_turns = sum(c.get("turn_count", c.get("turns", 0)) for c in convs_list)
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-chip">
                    <div class="mval">{len(convs_list)}</div>
                    <div class="mlbl">Hoi thoai</div>
                </div>
                <div class="metric-chip">
                    <div class="mval">{total_turns}</div>
                    <div class="mlbl">Tong luot hoi dap</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Selector ─────────────────────────────
        conv_pairs = []
        for c in convs_list:
            cid = str(c.get("conversation_id", ""))
            base = c.get("filename") or c.get("title") or "Conversation"
            conv_pairs.append((f"{base} • {cid[:8]}", cid))

        conv_options = {label: cid for label, cid in conv_pairs}
        chosen_label = st.selectbox(
            "Chon hoi thoai",
            options=list(conv_options.keys()),
            label_visibility="collapsed",
        )
        chosen_id = conv_options[chosen_label]

        detail = get_conversation(chosen_id)
        turns  = detail.get("turns", [])

        if not turns:
            st.markdown(
                '<div class="info-box">Hoi thoai nay chua co luot hoi dap nao.</div>',
                unsafe_allow_html=True,
            )
        else:
            for idx, turn in enumerate(turns):
                with st.expander(
                    f"Luot {idx+1} — {turn.get('question','')[:60]}{'…' if len(turn.get('question',''))>60 else ''}",
                    expanded=(idx == len(turns) - 1),
                ):
                    q = turn.get("question", "")
                    a = turn.get("answer", "")
                    ca = turn.get("corag_answer", "")
                    tid = turn.get("turn_id", "")
                    safe_tid = esc(tid)
                    safe_q = esc(q)
                    safe_a = esc(a)
                    safe_ca = esc(ca) if ca else '<em style="color:var(--subtext)">—</em>'
                    st.markdown(
                        f"""
                        <div style="margin-bottom:10px;">
                            <span style="font-size:10px;color:var(--subtext);text-transform:uppercase;
                                letter-spacing:.08em;">Ma luot</span><br>
                            <code style="font-size:11px;">{safe_tid}</code>
                        </div>
                        <div style="margin-bottom:10px;">
                            <span style="font-size:10px;color:var(--subtext);text-transform:uppercase;
                                letter-spacing:.08em;">Cau hoi</span><br>
                            <div style="font-size:14px;font-weight:600;margin-top:4px;">{safe_q}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div class="answer-card rag">
                            <div class="card-label rag">🔵 Tra loi RAG</div>
                            {safe_a}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div class="answer-card corag" style="margin-top:10px;">
                            <div class="card-label corag">🟢 Tra loi Co-RAG</div>
                            {safe_ca}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )