"""IntelliVault — RAG with authentication, OCR, history, and 3 answer modes."""

import hashlib
import os
import time
from pathlib import Path

import pymupdf
import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes

from config import (
    BASE_DIR,
    QUERY_PREFIX,
    TOP_K,
    MIN_SIMILARITY,
    EMBED_BATCH_SIZE,
    get_embedder,
    get_collection,
)
from ingest import splitter
from baseline_rag import call_llm
from hybrid import hybrid_retrieve
from agent import run_agent
from database import (
    register_user,
    login_user,
    save_pdf_history,
    get_pdf_history,
    save_qa_history,
    get_qa_history,
)


# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="IntelliVault",
    page_icon="📚",
    layout="wide",
)

MAX_FILES = 10
MAX_MB = 50
MAX_QUESTIONS = 50

TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
POPPLER_PATH = os.getenv(
    "POPPLER_PATH",
    r"C:\Users\Arya\Downloads\Release-26.09.0-0\poppler-26.09.0\Library\bin",
)

if os.path.isfile(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# --------------------------------------------------
# Cached resources  (loaded lazily on first use)
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading AI model — first run only...")
def load_embedder():
    return get_embedder()


@st.cache_resource(show_spinner="Connecting to vector store...")
def load_collection():
    return get_collection()


# --------------------------------------------------
# Session state
# --------------------------------------------------

def init_state():
    defaults = {
        "user": None,
        "messages": [],
        "indexed_files": {},   # file_key -> True
        "asked_questions": 0,
        "active_file_names": [],
        "answer_mode": "Baseline RAG",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# --------------------------------------------------
# Authentication
# --------------------------------------------------

def show_authentication():
    st.title("📚 IntelliVault")
    st.write("Sign in or create an account to access your documents.")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            email    = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            if not email.strip() or not password:
                st.error("Please enter your email and password.")
            else:
                try:
                    user = login_user(email.strip(), password)
                    if user:
                        st.session_state.user = user
                        st.session_state.messages = []
                        st.session_state.indexed_files = {}
                        st.session_state.asked_questions = 0
                        st.session_state.active_file_names = []
                        # Restore chat history
                        try:
                            history = get_qa_history(user["user_id"], limit=MAX_QUESTIONS)
                            for item in history:
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": item.get("question", ""),
                                    "filename": item.get("filename"),
                                    "mode": item.get("mode"),
                                })
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": item.get("answer", ""),
                                    "filename": item.get("filename"),
                                    "mode": item.get("mode"),
                                })
                        except Exception as exc:
                            st.warning(f"Logged in, but chat history could not be loaded: {exc}")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                except Exception as exc:
                    st.error(f"Login failed: {exc}")

    with register_tab:
        with st.form("register_form"):
            new_email    = st.text_input("Email address", key="register_email")
            new_password = st.text_input("Password (min 12 chars)", type="password", key="register_password")
            confirm      = st.text_input("Confirm password", type="password", key="register_confirm_password")
            submitted    = st.form_submit_button("Create account", use_container_width=True)

        if submitted:
            if not new_email.strip():
                st.error("Please enter your email address.")
            elif len(new_password) < 12:
                st.error("Password must be at least 12 characters.")
            elif new_password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    ok, result = register_user(new_email.strip(), new_password)
                    if ok:
                        st.success("Account created! Please log in.")
                    else:
                        st.error(str(result))
                except Exception as exc:
                    st.error(f"Registration failed: {exc}")


if st.session_state.user is None:
    show_authentication()
    st.stop()


# --------------------------------------------------
# Logged-in user
# --------------------------------------------------

user    = st.session_state.user
user_id = user["user_id"]


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("📚 IntelliVault")
    st.caption(f"Logged in as **{user.get('email', '')}**")

    if st.button("Logout", use_container_width=True):
        for k in ["user", "messages", "indexed_files", "asked_questions", "active_file_names"]:
            st.session_state[k] = None if k == "user" else ([] if isinstance(st.session_state[k], list) else {})
        st.rerun()

    st.divider()

    # ── Answer mode selector ──────────────────────────────────
    st.subheader("🤖 Answer Mode")
    mode = st.radio(
        "Choose how IntelliVault answers your questions:",
        options=["Baseline RAG", "Hybrid RAG", "Agentic AI"],
        index=["Baseline RAG", "Hybrid RAG", "Agentic AI"].index(
            st.session_state.answer_mode
        ),
        help=(
            "**Baseline RAG** — dense vector search → LLM answer\n\n"
            "**Hybrid RAG** — dense + BM25 keyword search (RRF fusion) → LLM\n\n"
            "**Agentic AI** — plans sub-queries, checks evidence, re-queries if needed"
        ),
        key="sidebar_mode",
    )
    st.session_state.answer_mode = mode

    MODE_BADGES = {
        "Baseline RAG": "🔵 Baseline RAG",
        "Hybrid RAG":   "🟣 Hybrid RAG",
        "Agentic AI":   "🟠 Agentic AI",
    }
    st.caption(f"Active: **{MODE_BADGES[mode]}**")

    st.divider()

    # ── Previously indexed files (reuse without re-uploading) ─
    st.subheader("📂 Your Indexed Files")
    st.caption("Tick files to make them active for Q&A without re-uploading.")

    try:
        pdf_history_list = get_pdf_history(user_id)
    except Exception:
        pdf_history_list = []

    if pdf_history_list:
        collection = load_collection()
        for item in pdf_history_list:
            fname     = item.get("filename", "Unnamed")
            fhash     = item.get("file_hash", "")
            file_key  = f"{user_id}:{fhash}"
            already   = file_key in st.session_state.indexed_files

            checked = st.checkbox(
                f"📄 {fname}",
                value=already,
                key=f"reuse_{file_key}",
            )

            if checked and not already:
                # Verify the vectors still exist in ChromaDB
                try:
                    existing = collection.get(
                        where={"$and": [
                            {"user_id": str(user_id)},
                            {"file_hash": fhash},
                        ]},
                        limit=1,
                        include=["metadatas"],
                    )
                    if existing.get("ids"):
                        st.session_state.indexed_files[file_key] = True
                        if fname not in st.session_state.active_file_names:
                            st.session_state.active_file_names.append(fname)
                        st.success(f"✅ {fname} is now active!")
                    else:
                        st.warning(f"⚠️ {fname} vectors not found — please re-upload.")
                except Exception as exc:
                    st.warning(f"Could not verify {fname}: {exc}")

            elif not checked and already:
                st.session_state.indexed_files.pop(file_key, None)
                if fname in st.session_state.active_file_names:
                    st.session_state.active_file_names.remove(fname)
    else:
        st.caption("No saved files yet. Upload a PDF below.")

    st.divider()

    if st.button("🗑️ Clear current chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.asked_questions = 0
        st.rerun()

    st.caption("Clearing chat does not delete saved history.")


# --------------------------------------------------
# Main area header
# --------------------------------------------------

st.title("📚 IntelliVault")
st.caption(
    f"Logged in as **{user.get('email', 'user')}** · "
    f"Mode: **{MODE_BADGES[st.session_state.answer_mode]}**"
)


# --------------------------------------------------
# PDF / TXT extraction with batched OCR
# --------------------------------------------------

def read_pages(filename, data):
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        return [{"page": 1, "text": text}] if text.strip() else []

    if suffix != ".pdf":
        raise ValueError("Only PDF and TXT files are supported.")

    pages = []
    ocr_pages = []

    # Pass 1 — extract selectable text
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for idx, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": idx + 1, "text": text})
            else:
                ocr_pages.append(idx + 1)

    # Pass 2 — batch OCR image-only pages in ONE Poppler call
    if ocr_pages:
        if not os.path.isfile(TESSERACT_PATH):
            raise RuntimeError(f"Tesseract not found at: {TESSERACT_PATH}")
        pdfinfo = os.path.join(POPPLER_PATH, "pdfinfo.exe") if POPPLER_PATH else None
        if not pdfinfo or not os.path.isfile(pdfinfo):
            raise RuntimeError(f"Poppler not found at: {POPPLER_PATH}")

        first, last = min(ocr_pages), max(ocr_pages)
        needed = set(ocr_pages)
        images = convert_from_bytes(
            data, dpi=150, first_page=first, last_page=last,
            poppler_path=POPPLER_PATH, fmt="png", thread_count=2,
        )
        try:
            for i, img in enumerate(images):
                pnum = first + i
                if pnum not in needed:
                    continue
                t = pytesseract.image_to_string(img, lang="eng").strip()
                if t:
                    pages.append({"page": pnum, "text": t})
        finally:
            for img in images:
                img.close()

    pages.sort(key=lambda p: p["page"])
    return pages


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def make_chunks(pages):
    chunks = []
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        if hasattr(splitter, "split_text"):
            parts = splitter.split_text(text)
        elif callable(splitter):
            parts = splitter(text)
        else:
            parts = [text]
        for chunk in parts:
            ct = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            if ct.strip():
                chunks.append({"text": ct.strip(), "page": page["page"]})
    return chunks


# --------------------------------------------------
# Indexing
# --------------------------------------------------

def index_file(filename, data, file_hash):
    t0 = time.perf_counter()
    pages  = read_pages(filename, data)
    chunks = make_chunks(pages)
    t_extract = time.perf_counter() - t0

    if not chunks:
        raise ValueError("No readable text found. The file may be blank or password-protected.")

    embedder   = load_embedder()
    collection = load_collection()
    documents  = [c["text"] for c in chunks]
    ids        = [f"{user_id}_{file_hash}_{i}" for i in range(len(chunks))]
    metadatas  = [
        {"user_id": str(user_id), "filename": filename,
         "file_hash": file_hash, "page": int(c["page"]), "chunk_index": i}
        for i, c in enumerate(chunks)
    ]

    t_emb = time.perf_counter()
    embeddings = embedder.encode(
        documents, batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
    ).tolist()
    t_embed = time.perf_counter() - t_emb

    t_store = time.perf_counter()
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    t_storage = time.perf_counter() - t_store

    try:
        save_pdf_history(user_id, filename, file_hash)
    except Exception:
        pass  # Don't fail indexing just because MongoDB is down

    return {
        "chunks": len(chunks), "pages": len(pages),
        "extraction_time": t_extract, "embedding_time": t_embed,
        "storage_time": t_storage, "total_time": time.perf_counter() - t0,
    }


# --------------------------------------------------
# Upload section
# --------------------------------------------------

st.subheader("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDFs or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help=f"Up to {MAX_FILES} files, max {MAX_MB} MB each.",
)

if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.warning(f"Only the first {MAX_FILES} of {len(uploaded_files)} files will be processed.")

    try:
        collection = load_collection()
    except Exception as exc:
        st.error(f"Could not connect to vector store: {exc}")
        st.stop()

    for uf in uploaded_files[:MAX_FILES]:
        fname    = uf.name
        data     = uf.getvalue()

        if len(data) > MAX_MB * 1024 * 1024:
            st.error(f"❌ {fname} exceeds the {MAX_MB} MB limit.")
            continue

        fhash    = hashlib.sha256(data).hexdigest()
        file_key = f"{user_id}:{fhash}"

        if file_key in st.session_state.indexed_files:
            st.info(f"✅ {fname} is already active in this session.")
            continue

        # Check ChromaDB for existing vectors
        try:
            existing = collection.get(
                where={"$and": [{"user_id": str(user_id)}, {"file_hash": fhash}]},
                limit=1, include=["metadatas"],
            )
            if existing.get("ids"):
                st.session_state.indexed_files[file_key] = True
                if fname not in st.session_state.active_file_names:
                    st.session_state.active_file_names.append(fname)
                st.info(f"✅ {fname} already indexed — reusing existing vectors.")
                continue
        except Exception as exc:
            st.warning(f"Could not verify existing vectors: {exc}")

        with st.spinner(f"⏳ Processing {fname}…"):
            try:
                timings = index_file(fname, data, fhash)
                st.session_state.indexed_files[file_key] = True
                if fname not in st.session_state.active_file_names:
                    st.session_state.active_file_names.append(fname)
                st.success(
                    f"✅ **{fname}** — {timings['chunks']} chunks from {timings['pages']} pages"
                )
                st.caption(
                    f"Extract: {timings['extraction_time']:.1f}s · "
                    f"Embed: {timings['embedding_time']:.1f}s · "
                    f"Store: {timings['storage_time']:.1f}s · "
                    f"Total: {timings['total_time']:.1f}s"
                )
            except Exception as exc:
                st.error(f"❌ Could not index {fname}: {exc}")


# --------------------------------------------------
# Baseline RAG retrieval (used by Baseline mode)
# --------------------------------------------------

def retrieve_baseline(question):
    """Dense-only retrieval filtered to this user. Returns (context_str, sources)."""
    embedder   = load_embedder()
    collection = load_collection()

    q_emb = embedder.encode(
        [QUERY_PREFIX + question.strip()],
        normalize_embeddings=True, convert_to_numpy=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        where={"user_id": str(user_id)},
        include=["documents", "metadatas", "distances"],
    )

    docs   = results.get("documents", [[]])[0]
    metas  = results.get("metadatas", [[]])[0]
    dists  = results.get("distances",  [[]])[0]

    parts, sources = [], []
    for doc, meta, dist in zip(docs, metas, dists):
        sim = 1.0 - float(dist)
        if sim < MIN_SIMILARITY:
            continue
        fn  = meta.get("filename", "Unknown")
        pg  = meta.get("page", "?")
        parts.append(f"[Source: {fn}, page {pg}]\n{doc}")
        sources.append({"filename": fn, "page": pg, "similarity": sim})

    return "\n\n".join(parts), sources


# --------------------------------------------------
# User-filtered ChromaDB collection wrapper for Hybrid/Agent
# --------------------------------------------------

class UserCollection:
    """Thin wrapper that delegates to ChromaDB but pre-filters by user_id."""

    def __init__(self, col, uid):
        self._col  = col
        self._uid  = str(uid)
        self.name  = col.name

    def count(self):
        # ChromaDB count() has no where filter — use get() to count user docs
        result = self._col.get(
            where={"user_id": self._uid},
            include=[],  # fetch only IDs, no content
        )
        return len(result.get("ids", []))

    def query(self, query_embeddings, n_results, **kwargs):
        where = kwargs.pop("where", {})
        if where:
            where = {"$and": [{"user_id": self._uid}, where]}
        else:
            where = {"user_id": self._uid}
        total = self._col.count()  # total docs in collection (no filter needed here)
        safe_n = min(n_results, max(total, 1))
        return self._col.query(
            query_embeddings=query_embeddings,
            n_results=safe_n,
            where=where, **kwargs,
        )

    def get(self, **kwargs):
        where = kwargs.pop("where", {})
        if where:
            where = {"$and": [{"user_id": self._uid}, where]}
        else:
            where = {"user_id": self._uid}
        return self._col.get(where=where, **kwargs)



# --------------------------------------------------
# Chat interface
# --------------------------------------------------

st.divider()
st.subheader("💬 Ask Your Documents")

if st.session_state.active_file_names:
    st.caption(f"Active files: {', '.join(st.session_state.active_file_names)}")
else:
    st.caption("No active files yet. Upload a PDF or tick files in the sidebar.")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        badge = MODE_BADGES.get(msg.get("mode", ""), "")
        if badge or msg.get("filename"):
            parts = []
            if badge:
                parts.append(badge)
            if msg.get("filename"):
                parts.append(f"📄 {msg['filename']}")
            st.caption(" · ".join(parts))

question = st.chat_input("Ask a question about your indexed documents…")

if question:
    if st.session_state.asked_questions >= MAX_QUESTIONS:
        st.warning(f"You reached the {MAX_QUESTIONS}-question session limit.")
        st.stop()

    st.session_state.asked_questions += 1
    current_mode = st.session_state.answer_mode

    st.session_state.messages.append({"role": "user", "content": question, "mode": current_mode})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        spinner_text = {
            "Baseline RAG": "🔵 Searching documents (Baseline RAG)…",
            "Hybrid RAG":   "🟣 Hybrid search + fusion (Hybrid RAG)…",
            "Agentic AI":   "🟠 Planning & multi-step retrieval (Agentic AI)…",
        }[current_mode]

        with st.spinner(spinner_text):
            try:
                answer      = ""
                sources     = []
                agent_trace = []

                # ── Baseline RAG ──────────────────────────────────────────
                if current_mode == "Baseline RAG":
                    context, sources = retrieve_baseline(question)
                    if not context:
                        answer = (
                            "I couldn't find relevant information in your indexed documents. "
                            "Please upload the relevant file or try a more specific question."
                        )
                    else:
                        prompt = (
                            "You are IntelliVault, a document Q&A assistant.\n"
                            "Answer using ONLY the supplied context. "
                            "Cite sources as [filename, page N].\n\n"
                            f"CONTEXT:\n{context}\n\n"
                            f"QUESTION: {question}\n\nANSWER:"
                        )
                        answer = call_llm(prompt)

                # ── Hybrid RAG ────────────────────────────────────────────
                elif current_mode == "Hybrid RAG":
                    collection  = load_collection()
                    user_col    = UserCollection(collection, user_id)
                    hits, best  = hybrid_retrieve(user_col, question, k=TOP_K)

                    if not hits or best < MIN_SIMILARITY:
                        answer = (
                            "I couldn't find relevant information using hybrid search. "
                            "Please upload the relevant file or try rephrasing."
                        )
                    else:
                        from baseline_rag import build_context
                        context = build_context(hits)
                        prompt  = (
                            "You are IntelliVault, a document Q&A assistant.\n"
                            "Answer using ONLY the supplied context. "
                            "Cite sources as [filename, page N].\n\n"
                            f"CONTEXT:\n{context}\n\n"
                            f"QUESTION: {question}\n\nANSWER:"
                        )
                        answer  = call_llm(prompt)
                        sources = [
                            {"filename": h.get("source", "?"), "page": h.get("page", "?"), "similarity": h.get("score", 0)}
                            for h in hits
                        ]

                # ── Agentic AI ────────────────────────────────────────────
                elif current_mode == "Agentic AI":
                    collection  = load_collection()
                    user_col    = UserCollection(collection, user_id)
                    answer, hits, agent_trace = run_agent(user_col, question)
                    sources = [
                        {"filename": h.get("source", "?"), "page": h.get("page", "?"), "similarity": h.get("score", 0)}
                        for h in hits
                    ]

                if not isinstance(answer, str):
                    answer = str(answer)

                st.markdown(answer)

                # Sources expander
                if sources:
                    with st.expander("📚 Sources"):
                        for s in sources:
                            st.write(
                                f"- **{s['filename']}** · page {s['page']} "
                                f"· similarity {s['similarity']:.3f}"
                            )

                # Agent trace expander
                if agent_trace:
                    with st.expander("🔍 Agent reasoning trace"):
                        for step in agent_trace:
                            st.caption(f"• {step}")

                # Mode badge under answer
                st.caption(MODE_BADGES[current_mode])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "mode": current_mode,
                    "filename": sources[0]["filename"] if sources else None,
                })

                # Persist to MongoDB (best-effort)
                try:
                    save_qa_history(
                        user_id, question, answer,
                        filename=sources[0]["filename"] if sources else None,
                        mode=current_mode,
                    )
                except Exception:
                    pass

            except Exception as exc:
                err = f"Something went wrong: {exc}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant", "content": err, "mode": current_mode,
                })
