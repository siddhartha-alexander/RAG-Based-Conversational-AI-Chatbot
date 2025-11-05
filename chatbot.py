# -----------------------------
# 💬 RAG-BASED CHATBOT (STREAMLIT)
# Developed by Damala Siddhartha Alexander
# -----------------------------

import os
import tempfile
from typing import List

import streamlit as st
import pdfplumber  # ✅ Stable for Streamlit Cloud
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

# ---------- CONFIG ----------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # HuggingFace sentence-transformers
GEN_MODEL = "google/flan-t5-base"       # generation model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# ---------- UTILITIES ----------
def load_text_from_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber (safe for Streamlit Cloud)."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def load_documents_from_folder(folder: str, exts=("pdf", "txt")) -> List[Document]:
    """Load all supported files (PDF/TXT) from a folder."""
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            ext = fname.split(".")[-1].lower()
            if ext not in exts:
                continue
            full = os.path.join(root, fname)
            try:
                if ext == "pdf":
                    text = load_text_from_pdf(full)
                else:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                docs.append(Document(page_content=text, metadata={"source": full}))
            except Exception as e:
                st.warning(f"Failed to load {full}: {e}")
    return docs


def split_documents(docs: List[Document], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split documents into smaller chunks for embedding."""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    out = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            meta = dict(d.metadata or {})
            meta.update({"chunk": i})
            out.append(Document(page_content=c, metadata=meta))
    return out


# ---------- VECTORSTORE & EMBEDDINGS (cached) ----------
@st.cache_resource(show_spinner=False)
def build_vectorstore(documents: List[Document]):
    """Build FAISS index from given documents."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitted = split_documents(documents)
    texts = [d.page_content for d in splitted]
    metadatas = [d.metadata for d in splitted]
    store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return store, embeddings


# ---------- GENERATOR (cached) ----------
@st.cache_resource(show_spinner=False)
def load_generator():
    """Load text generation model (Flan-T5)."""
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return gen


def answer_query(store: FAISS, query: str, generator, top_k: int = TOP_K):
    """Retrieve top chunks and generate an answer."""
    results = store.similarity_search(query, k=top_k)
    if not results:
        return "No relevant documents found.", []

    context = "\n\n".join([r.page_content for r in results])
    sources = [r.metadata.get("source", "unknown") for r in results]

    prompt = (
        "Use the following context to answer the question concisely.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    out = generator(prompt, max_length=256, do_sample=False)
    text = out[0].get("generated_text") if isinstance(out, list) else str(out)
    return text.strip(), sources


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="RAG Chatbot by Damala", page_icon="🤖")
st.title("💬 RAG-based Chatbot (Streamlit)")
st.caption("Upload PDF/TXT docs, build FAISS index, and ask questions (HuggingFace embeddings + Flan-T5).")

# Sidebar for documents
st.sidebar.header("📂 Document Options")
use_upload = st.sidebar.checkbox("Upload documents (recommended)", value=True)

documents = []
if use_upload:
    uploaded = st.sidebar.file_uploader("Upload PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded:
        tmpdir = tempfile.TemporaryDirectory()
        for uf in uploaded:
            save_to = os.path.join(tmpdir.name, uf.name)
            with open(save_to, "wb") as fh:
                fh.write(uf.getbuffer())
            if uf.name.lower().endswith(".pdf"):
                text = load_text_from_pdf(save_to)
            else:
                with open(save_to, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            documents.append(Document(page_content=text, metadata={"source": uf.name}))
else:
    folder = st.sidebar.text_input("Enter local folder path", value="docs")
    if st.sidebar.button("Load folder"):
        if os.path.exists(folder):
            documents = load_documents_from_folder(folder)
            st.sidebar.success(f"Loaded {len(documents)} documents.")
        else:
            st.sidebar.error("Folder not found.")

# Build index
if documents:
    st.info(f"{len(documents)} document(s) selected — building vectorstore (first time may take a while)...")
    with st.spinner("Building embeddings and FAISS index..."):
        store, embeddings = build_vectorstore(documents)
    st.success("✅ Vectorstore ready.")
else:
    st.info("No documents loaded yet. Upload files or load a folder to start.")
    store = None

# Query input & generation
st.subheader("💭 Ask a question")
query = st.text_input("Type your question and press Enter")

if st.button("Load Generation Model (Flan-T5)"):
    with st.spinner("Loading model... (takes time only once)"):
        _gen = load_generator()
    st.session_state["gen_loaded"] = True
    st.success("Model loaded successfully.")

# Autoload generator if cached
generator = None
if st.session_state.get("gen_loaded", False):
    generator = load_generator()

if query:
    if store is None:
        st.warning("⚠️ Please upload documents and build the index first.")
    else:
        if generator is None:
            with st.spinner("Loading model..."):
                generator = load_generator()
            st.session_state["gen_loaded"] = True

        with st.spinner("Retrieving and generating answer..."):
            ans, sources = answer_query(store, query, generator)

        st.markdown("### 🧠 Answer:")
        st.write(ans)

        st.markdown("### 📄 Retrieved Sources:")
        for i, s in enumerate(sources):
            st.write(f"{i+1}. {s}")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("**Developed by Damala Siddhartha Alexander**")
st.caption("RAG-based Conversational AI | Built with LangChain, FAISS, and Flan-T5")
