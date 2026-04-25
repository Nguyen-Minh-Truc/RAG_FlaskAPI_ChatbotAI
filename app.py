"""
RAG & Co-RAG Explorer — Streamlit UI
Connects to the Flask REST API defined in routes.py
"""

import time
import html
import os
import re
from datetime import datetime
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

mark {
    background: rgba(247, 214, 62, 0.35);
    color: var(--text);
    padding: 0 2px;
    border-radius: 3px;
}
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
if "api_base" not in st.session_state:
    st.session_state.api_base = discover_api_base()

if "question_box_version" not in st.session_state:
    st.session_state.question_box_version = 0
if "use_hybrid_search" not in st.session_state:
    st.session_state.use_hybrid_search = True
if "ask_mode" not in st.session_state:
    st.session_state.ask_mode = "Compare"
if "corag_rounds" not in st.session_state:
    st.session_state.corag_rounds = 3
if "available_documents" not in st.session_state:
    st.session_state.available_documents = []
if "selected_sources" not in st.session_state:
    st.session_state.selected_sources = []
if "selected_file_types" not in st.session_state:
    st.session_state.selected_file_types = []
if "filter_upload_date_from" not in st.session_state:
    st.session_state.filter_upload_date_from = None
if "filter_upload_date_to" not in st.session_state:
    st.session_state.filter_upload_date_to = None

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


_HIGHLIGHT_STOPWORDS = {
    "la", "va", "cua", "cho", "trong", "mot", "nhung", "cac", "the", "nay", "kia",
    "toi", "ban", "anh", "chi", "em", "hay", "voi", "ve", "tu", "tai", "duoc", "khong",
}


def _extract_keywords(question: str, max_terms: int = 8) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_\-\u00C0-\u1EF9]+", (question or "").lower())
    filtered: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if len(term) < 3 or term in _HIGHLIGHT_STOPWORDS or term in seen:
            continue
        filtered.append(term)
        seen.add(term)
        if len(filtered) >= max_terms:
            break

    return filtered


def _highlight_text(text: str, keywords: list[str]) -> str:
    if not text:
        return ""

    if not keywords:
        return esc(text)

    pattern = re.compile("(" + "|".join(re.escape(term) for term in keywords) + ")", re.IGNORECASE)
    parts = pattern.split(text)

    highlighted_parts: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            highlighted_parts.append(f"<mark>{esc(part)}</mark>")
        else:
            highlighted_parts.append(esc(part))

    return "".join(highlighted_parts)


def render_context_sources(context_chunks: list[dict], question: str, key_prefix: str):
    """Render source origins first; show raw context only when user requests it."""
    if not context_chunks:
        st.caption("Khong co context duoc luu cho luot hoi dap nay.")
        return

    st.markdown("#### Nguon goc thong tin")
    for idx, chunk in enumerate(context_chunks, start=1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source") or "unknown"
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        score = float(chunk.get("score", 0.0))

        st.markdown(
            f"""
            <div class="info-box" style="margin-top:8px;">
                <strong>[{idx}] {esc(source)}</strong><br>
                Trang: <strong>{esc(page if page is not None else '?')}</strong> · Vi tri chunk: <strong>{esc(chunk_index if chunk_index is not None else '?')}</strong> · score: <strong>{score:.4f}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    show_raw_context = st.toggle(
        "Xem context goc",
        value=False,
        key=f"{key_prefix}_show_raw_context",
        help="Bat neu ban muon xem toan bo doan context da truy hoi.",
    )

    if not show_raw_context:
        return

    keywords = _extract_keywords(question)

    tab_labels: list[str] = []
    for idx, chunk in enumerate(context_chunks, start=1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source") or "unknown"
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        score = float(chunk.get("score", 0.0))

        page_label = f"page {page}" if page is not None else "page ?"
        pos_label = f"chunk {chunk_index}" if chunk_index is not None else "chunk ?"
        tab_labels.append(f"[{idx}] {source} | {page_label} | {pos_label} | score={score:.4f}")

    tabs = st.tabs(tab_labels)

    for tab, chunk in zip(tabs, context_chunks):
        with tab:
            metadata = chunk.get("metadata", {}) or {}
            source = metadata.get("source") or "unknown"
            page = metadata.get("page")
            chunk_index = metadata.get("chunk_index")
            score = float(chunk.get("score", 0.0))
            text = chunk.get("text", "")

            st.markdown(
                f"""
                <div style="margin-bottom:8px;color:#b8bfd8;font-size:12px;">
                    Nguon: {esc(source)} | Trang: {esc(page)} | Vi tri chunk: {esc(chunk_index)} | score={score:.4f}
                </div>
                <div style="white-space:pre-wrap;line-height:1.65;">
                    {_highlight_text(text, keywords)}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_source_summary(source_summary: list[dict], title: str = "Tai lieu dong gop"):
    """Render document-level contribution summary."""
    if not source_summary:
        return

    st.markdown(f"#### {title}")
    for item in source_summary:
        source = item.get("source", "unknown")
        file_type = item.get("file_type") or "unknown"
        chunk_count = item.get("chunk_count", 0)
        contribution = item.get("contribution_pct", 0)
        avg_score = item.get("avg_score", 0)
        pages = item.get("pages", [])
        upload_date = item.get("upload_date") or "unknown"

        pages_text = ", ".join(str(page) for page in pages[:8]) if pages else "-"
        st.markdown(
            f"""
            <div class="info-box" style="margin-top:8px;">
                <strong>{esc(source)}</strong> ({esc(file_type)})<br>
                Chunk dong gop: <strong>{esc(chunk_count)}</strong> · Ti le: <strong>{esc(contribution)}%</strong> · Avg score: <strong>{esc(avg_score)}</strong><br>
                Trang lien quan: {esc(pages_text)}<br>
                Upload: {esc(upload_date)}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _extract_filter_options(uploaded_documents: list[dict]) -> tuple[list[str], list[str]]:
    """Build unique metadata filter options from uploaded document metadata."""
    sources: set[str] = set()
    file_types: set[str] = set()

    for item in uploaded_documents:
        source = str(item.get("source") or "").strip()
        file_type = str(item.get("file_type") or "").strip().lower()
        if source:
            sources.add(source)
        if file_type:
            file_types.add(file_type)

    return sorted(sources), sorted(file_types)


def _to_iso_day_start(date_text: str) -> str:
    return f"{date_text}T00:00:00+00:00"


def _to_iso_day_end(date_text: str) -> str:
    return f"{date_text}T23:59:59+00:00"


def _date_for_widget(date_text: str | None):
    """Convert YYYY-MM-DD string from session state to date object for date_input."""
    if not date_text:
        return None

    try:
        return datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        return None


def build_active_metadata_filters() -> dict:
    """Build metadata filter payload from sidebar controls."""
    return {
        "sources": st.session_state.selected_sources,
        "file_types": st.session_state.selected_file_types,
        "document_ids": [],
        "upload_date_from": _to_iso_day_start(st.session_state.filter_upload_date_from)
        if st.session_state.filter_upload_date_from
        else None,
        "upload_date_to": _to_iso_day_end(st.session_state.filter_upload_date_to)
        if st.session_state.filter_upload_date_to
        else None,
    }


def validate_filter_dates() -> str | None:
    """Validate date filter fields and return error message when invalid."""
    for field_name, field_value in (
        ("from", st.session_state.filter_upload_date_from),
        ("to", st.session_state.filter_upload_date_to),
    ):
        if not field_value:
            continue
        try:
            datetime.strptime(field_value, "%Y-%m-%d")
        except ValueError:
            return f"Upload date {field_name} khong dung dinh dang YYYY-MM-DD."

    return None


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


def get_conversation_documents(cid: str):
    data, code = api("get", f"/api/conversations/{cid}/documents", timeout=10)
    if code == 200:
        return data.get("data", {}).get("documents", [])
    return []


def delete_conversation(cid: str):
    _, code = api("delete", f"/api/conversations/{cid}", timeout=10)
    return code == 200


def delete_all_conversations():
    _, code = api("delete", "/api/conversations", timeout=10)
    return code == 200


def upload_file(uploaded_files, chunk_size: int, chunk_overlap: int, conversation_id: str | None = None):
    files_payload = []
    for uploaded_file in uploaded_files:
        files_payload.append(
            (
                "files",
                (
                    uploaded_file.name,
                    uploaded_file.read(),
                    uploaded_file.type or "application/octet-stream",
                ),
            )
        )

    form_data = {
        "chunk_size": str(chunk_size),
        "chunk_overlap": str(chunk_overlap),
    }
    if conversation_id:
        form_data["conversation_id"] = conversation_id

    data, code = api(
        "post", "/api/upload",
        timeout=300,
        files=files_payload,
        data=form_data,
    )
    return data, code


def ask_question(
    question: str,
    conv_id: str,
    corag_rounds: int,
    use_hybrid_search: bool,
    ask_mode: str,
    metadata_filters: dict,
):
    endpoint = {
        "RAG": "/api/rag-ask",
        "Co-RAG": "/api/corag-ask",
        "Compare": "/api/ask",
    }.get(ask_mode, "/api/ask")

    data, code = api(
        "post", endpoint,
        timeout=300,
        json={
            "question": question,
            "conversation_id": conv_id,
            "corag_rounds": corag_rounds,
            "use_hybrid_search": use_hybrid_search,
            "metadata_filters": metadata_filters,
        },
    )
    return data, code


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

    # Force hybrid search on for all requests.
    st.session_state.use_hybrid_search = True
 
    st.markdown('<div class="sidebar-section">🧠 Che do hoi dap</div>', unsafe_allow_html=True)
    st.session_state.corag_rounds = 3
    st.session_state.ask_mode = st.selectbox(
        "Che do hoi dap",
        options=["Compare", "RAG", "Co-RAG"],
        index=["Compare", "RAG", "Co-RAG"].index(st.session_state.ask_mode),
        help="Compare: tra ve ca RAG va Co-RAG, RAG/Co-RAG: chi chay 1 luong.",
    )

    # ── Upload ───────────────────────────────
    st.markdown('<div class="sidebar-section">📂 Tai tai lieu</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Tha file vao day",
        type=["pdf", "doc", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    append_to_current = st.checkbox(
        "Append vao hoi thoai hien tai",
        value=False,
        help="Neu bat, tai lieu moi se duoc them vao hoi thoai dang chon thay vi tao hoi thoai moi.",
    )
    chunk_size_option = st.selectbox(
        "Chunk size",
        options=[500, 1000, 1500, 2000],
        index=1,
        help="Do dai moi chunk khi chia van ban.",
    )
    chunk_overlap_option = st.selectbox(
        "Chunk overlap",
        options=[50, 100, 200],
        index=1,
        help="So ky tu overlap giua hai chunk lien tiep.",
    )
    do_upload = st.button("⬆ Nap tai lieu", key="btn_upload", use_container_width=True, type="primary")

    if do_upload:
        if not uploaded_files:
            st.warning("Vui long chon it nhat mot file truoc.")
        elif append_to_current and not st.session_state.active_conv_id:
            st.warning("Khong co hoi thoai dang chon de append.")
        else:
            with st.spinner("Dang nap du lieu..."):
                data, code = upload_file(
                    uploaded_files,
                    chunk_size=chunk_size_option,
                    chunk_overlap=chunk_overlap_option,
                    conversation_id=st.session_state.active_conv_id if append_to_current else None,
                )
            if code == 200:
                info = data.get("data", {})
                st.session_state.active_conv_id = info.get("conversation_id")
                st.session_state.chat_history = []
                st.session_state.available_documents = info.get("uploaded_documents", [])
                st.toast(
                    (
                        "✅ Da lap chi muc "
                        f"{info.get('chunks', '?')} doan tu {info.get('document_count', '?')} tai lieu "
                        f"(chunk_size={info.get('chunk_size', '?')}, overlap={info.get('chunk_overlap', '?')})."
                    ),
                    icon="📄",
                )
                parse_errors = info.get("parse_errors", [])
                if parse_errors:
                    st.warning(f"Co {len(parse_errors)} file khong parse duoc. Kiem tra lai dinh dang/noi dung file.")
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

        if st.session_state.active_conv_id and not st.session_state.available_documents:
            st.session_state.available_documents = get_conversation_documents(st.session_state.active_conv_id)

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
                            "mode": t.get("mode", "Compare"),
                            "context": t.get("context", []),
                            "source_summary": t.get("source_summary", []),
                            "metadata_filters": t.get("metadata_filters", {}),
                        }
                        for t in detail.get("turns", [])
                    ]
                    st.session_state.available_documents = get_conversation_documents(cid)
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
                st.session_state.available_documents = []
                st.toast("Da xoa toan bo hoi thoai.", icon="🗑")
                st.rerun()

    st.markdown('<div class="sidebar-section">🧩 Metadata filter</div>', unsafe_allow_html=True)
    source_options, file_type_options = _extract_filter_options(st.session_state.available_documents)
    active_filter_count = (
        len(st.session_state.selected_sources)
        + len(st.session_state.selected_file_types)
        + (1 if st.session_state.filter_upload_date_from else 0)
        + (1 if st.session_state.filter_upload_date_to else 0)
    )

    with st.popover(f"Mo bo loc metadata ({active_filter_count})", use_container_width=True):
        st.session_state.selected_sources = st.multiselect(
            "Loc theo ten file",
            options=source_options,
            default=[value for value in st.session_state.selected_sources if value in source_options],
            help="Chi truy hoi chunk tu cac file duoc chon.",
        )
        st.session_state.selected_file_types = st.multiselect(
            "Loc theo loai file",
            options=file_type_options,
            default=[value for value in st.session_state.selected_file_types if value in file_type_options],
            help="Vi du: pdf, doc, docx.",
        )
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            upload_date_from = st.date_input(
                "Upload date from",
                value=_date_for_widget(st.session_state.filter_upload_date_from),
                format="YYYY-MM-DD",
                help="De trong neu khong loc theo ngay bat dau.",
            )
        with date_col2:
            upload_date_to = st.date_input(
                "Upload date to",
                value=_date_for_widget(st.session_state.filter_upload_date_to),
                format="YYYY-MM-DD",
                help="De trong neu khong loc theo ngay ket thuc.",
            )

        st.session_state.filter_upload_date_from = (
            upload_date_from.strftime("%Y-%m-%d") if upload_date_from else None
        )
        st.session_state.filter_upload_date_to = (
            upload_date_to.strftime("%Y-%m-%d") if upload_date_to else None
        )

        if st.button("Xoa bo loc", use_container_width=True):
            st.session_state.selected_sources = []
            st.session_state.selected_file_types = []
            st.session_state.filter_upload_date_from = None
            st.session_state.filter_upload_date_to = None
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
            for history_index, item in enumerate(st.session_state.chat_history, start=1):
                safe_question = esc(item.get("question", ""))
                rag_answer = item.get("rag_answer")
                corag_answer = item.get("corag_answer")
                context_chunks = item.get("context", [])
                source_summary = item.get("source_summary", []) or []
                item_mode = item.get("mode", "Compare")
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
                    if item_mode == "RAG":
                        st.markdown(
                            f"""
                            <div class="answer-card rag">
                                <div class="card-label rag">🔵 RAG tieu chuan</div>
                                {safe_rag}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif item_mode == "Co-RAG":
                        st.markdown(
                            f"""
                            <div class="answer-card corag">
                                <div class="card-label corag">🟢 Co-RAG (lap)</div>
                                {safe_corag}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
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

                    render_source_summary(source_summary)

                    render_context_sources(
                        context_chunks=context_chunks,
                        question=item.get("question", ""),
                        key_prefix=f"chat_{history_index}",
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
                date_error = validate_filter_dates()
                if date_error:
                    st.error(date_error)
                    st.stop()

                metadata_filters = build_active_metadata_filters()

                with st.spinner("Đang suy nghi..."):
                    resp, code = ask_question(
                        question.strip(),
                        st.session_state.active_conv_id,
                        st.session_state.corag_rounds,
                        st.session_state.use_hybrid_search,
                        st.session_state.ask_mode,
                        metadata_filters,
                    )

                if code == 200:
                    d = resp.get("data", {})
                    st.session_state.chat_history.append({
                        "question":     question.strip(),
                        "rag_answer":   d.get("rag_answer", ""),
                        "corag_answer": d.get("corag_answer", ""),
                        "mode":         d.get("mode", st.session_state.ask_mode),
                        "context":      d.get("context", []),
                        "source_summary": d.get("source_summary", []),
                        "metadata_filters": d.get("metadata_filters", metadata_filters),
                        "use_hybrid_search": d.get("use_hybrid_search", st.session_state.use_hybrid_search),
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
                    source_summary = turn.get("source_summary", [])
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

                    render_source_summary(source_summary)

                    render_context_sources(
                        context_chunks=turn.get("context", []),
                        question=q,
                        key_prefix=f"history_{idx+1}",
                    )