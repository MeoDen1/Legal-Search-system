import streamlit as st
import os
from app import Core

# --- Configuration ---
DB_CFG_PATH = os.path.join("configs", "database.yaml")


# --- Initialize Core Engine ---
@st.cache_resource
def get_core():
    with st.spinner("Loading Legal AI Engine & Embeddings..."):
        return Core(db_cfg_path=DB_CFG_PATH)


def check_engine_status(core) -> tuple[bool, str]:
    try:
        if core is None:
            return False, "Engine not loaded"
        if not hasattr(core, 'searcher') or core.searcher is None:
            return False, "Searcher not initialized"
        if not hasattr(core, 'database') or core.database is None:
            return False, "Database not initialized"
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def get_first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text[:120]


# --- Page Config ---
st.set_page_config(page_title="Legal AI Search", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Expander styling */
    .stExpander {
        border: 1px solid #2a2a3a !important;
        border-radius: 10px !important;
        margin-bottom: 0.6rem !important;
        background: #16161f !important;
        overflow: hidden;
    }
    .stExpander summary {
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.8rem 1rem !important;
        color: #e0e0f0 !important;
    }
    .stExpander summary:hover {
        background: #1e1e2e !important;
    }

    /* Article content */
    .article-meta {
        font-size: 0.72rem;
        color: #666;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .article-content {
        color: #c8c8d8;
        font-size: 1.05rem;
        line-height: 1.9;
        white-space: pre-wrap;
        padding: 0.2rem 0.4rem 0.8rem 0.4rem;
    }
    </style>
""", unsafe_allow_html=True)


# --- Header ---
st.title("Vietnamese Legal Search")
st.caption("Search through thousands of legal documents with neural precision.")

# --- Search bar ---
col1, col2, col3 = st.columns([1, 12, 1])
with col2:
    with st.form(key="search_form", clear_on_submit=False):
        query = st.text_input(
            "",
            placeholder="Enter your legal question here...",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Search", use_container_width=True)

core = get_core()

# --- Run search, store results in session ---
if submitted and query:
    with st.spinner("Searching..."):
        results = core.search(query)
    st.session_state["results"] = results

# --- Display results ---
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]

    # Flatten to list of (doc_id, doc_name, content)
    all_articles = [
        (doc_id, data.get("document_name", "Unknown"), content)
        for doc_id, data in results.items()
        for content in data.get("articles", [])
        if content.strip()
    ]

    # --- Filter dropdown (by document name) ---
    doc_names = sorted(set(doc_name for _, doc_name, _ in all_articles))
    selected_docs = st.multiselect(
        "Filter by document",
        options=doc_names,
        default=[],
        placeholder="Showing all documents — select to filter"
    )

    filtered = (
        [(did, dn, c) for did, dn, c in all_articles if dn in selected_docs]
        if selected_docs else all_articles
    )

    st.subheader(f"Found {len(filtered)} result{'s' if len(filtered) != 1 else ''}")

    # --- Render each article ---
    for doc_id, doc_name, content in filtered:
        first_line = get_first_line(content)
        preview = first_line[:90] + ("..." if len(first_line) > 90 else "")

        # Bold preview shown in collapsed label; disappears when expanded (label stays)
        with st.expander(f"📄 {doc_name}  —  **{preview}**", expanded=False):
            st.markdown(f"<div class='article-meta'>{doc_name}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='article-content'>{content}</div>", unsafe_allow_html=True)

elif "results" in st.session_state and not st.session_state["results"]:
    st.warning("No results found.")

# --- Sidebar ---
with st.sidebar:
    st.header("Pipeline Status")

    connected, status_msg = check_engine_status(core)
    if connected:
        st.success(f"C++ Engine: {status_msg}")
    else:
        st.error(f"C++ Engine: {status_msg}")

    try:
        if core and core.embedding is not None:
            st.success(f"Model: {core.embedding_model_name}")
        else:
            st.error("Embedding model not loaded")
    except Exception as e:
        st.error(f"Model error: {e}")
