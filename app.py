# app.py
# Streamlit RAG demo: upload data -> chunk -> embed -> store in vector DB -> retrieve -> generate answer
# Requirements:
#   pip install streamlit chromadb sentence-transformers pypdf openai tiktoken

import os
import io
import time
import hashlib
from typing import List, Dict, Any, Tuple

import streamlit as st
from pypdf import PdfReader

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector DB (Chroma)
import chromadb
from chromadb.config import Settings

# LLM (Gemini)
import google.generativeai as genai
import uuid



# -----------------------------
# Utilities
# -----------------------------
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            text.append(txt)
    return "\n".join(text)

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[str]:
    # simple recursive-ish splitter on paragraphs, then rolling window fallback
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    joined = "\n".join(paras)
    tokens = joined.split()  # whitespace tokenization (fast & robust)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk = " ".join(chunk_tokens)
        chunks.append(chunk)
        i += (chunk_size - chunk_overlap)
    return chunks

def ensure_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        # Temporary direct key here
        api_key = ""  

    genai.configure(api_key=api_key)
    return genai

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def call_llm(client, prompt: str, model: str = "gemini-pro") -> str:
    try:
        model_instance = client.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini error] {e}"

def format_citations(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ch in enumerate(chunks, 1):
        src = ch["metadata"].get("source","user-upload")
        cid = ch["id"]
        lines.append(f"[{i}] {src} · id={cid}")
    return "\n".join(lines)

def build_rag_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"---\n{r['text']}\n")
    context = "\n".join(context_blocks)
    return (
        "Use ONLY the context below to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Interactive RAG Demo", layout="wide")
st.title("Interactive RAG Demo (Streamlit)")

with st.sidebar:
    st.subheader("Setup")
    st.write("This app shows the full RAG pipeline end-to-end.")
    st.markdown(
        "- Upload text/PDF or paste text\n"
        "- Chunk & embed\n"
        "- Store in Chroma (local)\n"
        "- Ask questions\n"
        "- Inspect retrieved chunks and scores"
    )

    st.divider()
    st.caption("Models")
    embed_model_name = st.selectbox(
        "Embedding model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        index=0
    )
    gen_model_name = st.selectbox(
        "LLM Model (Gemini)",
        options=[
            "gemini-2.5-flash",
        ],
        index=0
    )

    st.divider()
    st.caption("Chunking")
    chunk_size = st.slider("Chunk size (tokens approx.)", 200, 1600, 800, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 120, 10)

    st.divider()
    st.caption("Vector DB")
    persist_dir = st.text_input("Chroma persist directory", value=".chromadb")
    reset_db = st.button("Reset vector DB (clear memory)")


# -----------------------------
# Session State
# -----------------------------
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer(embed_model_name)
if "client" not in st.session_state:
    st.session_state.client = ensure_client()
if "chroma_client" not in st.session_state or reset_db:
    # Detect Streamlit Cloud
    if "STREAMLIT_RUNTIME" in os.environ:
        # Use ephemeral in-memory vector DB
        st.session_state.chroma_client = chromadb.Client()
    else:
        # Local persistent storage
        st.session_state.chroma_client = chromadb.PersistentClient(path=persist_dir)

if "collection" not in st.session_state or reset_db:
    st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(
        name="rag_demo",
        metadata={"hnsw:space": "cosine"}   # keeps cosine similarity working
    )
if "collection" not in st.session_state or reset_db:
    st.session_state.collection = st.session_state.chroma_client.get_or_create_collection("rag_demo")

# Swap embedder if user selected a different model
if "current_embed_model" not in st.session_state:
    st.session_state.current_embed_model = embed_model_name

if st.session_state.current_embed_model != embed_model_name:
    st.session_state.embedder = SentenceTransformer(embed_model_name)
    st.session_state.current_embed_model = embed_model_name


# -----------------------------
# Flow Visualization
# -----------------------------
st.subheader("RAG Flow")
st.graphviz_chart("""
digraph G {
  rankdir=LR;
  node [shape=box, style=rounded];
  A [label="1) Input Data (paste/upload)"];
  B [label="2) Chunking"];
  C [label="3) Embeddings"];
  D [label="4) Vector DB (Chroma)"];
  E [label="5) Query"];
  F [label="6) Similarity Search"];
  G [label="7) LLM with Retrieved Context"];
  H [label="8) Answer + Citations"];
  A -> B -> C -> D;
  E -> F -> D;
  F -> G -> H;
}
""")

# -----------------------------
# Data Ingestion UI
# -----------------------------
st.header("1) Add Context Data")

col_upload, col_paste = st.columns(2)
with col_upload:
    files = st.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt","pdf"],
        accept_multiple_files=True
    )
with col_paste:
    pasted = st.text_area("Or paste raw text here", height=180, placeholder="Paste context text...")

ingest_clicked = st.button("Ingest → Chunk → Embed → Store")

if ingest_clicked:
    all_texts = []

    # Handle uploads
    if files:
        for f in files:
            content = f.read()
            if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                doc = read_pdf(content)
            else:
                try:
                    doc = content.decode("utf-8")
                except Exception:
                    doc = content.decode("latin-1", errors="ignore")
            if doc.strip():
                all_texts.append((f.name, doc))

    # Handle pasted
    if pasted and pasted.strip():
        all_texts.append(("pasted-text", pasted))

    if not all_texts:
        st.warning("No input data found.")
    else:
        with st.status("Processing & storing in vector DB...", expanded=True) as status:
            embedder = st.session_state.embedder
            collection = st.session_state.collection

            total_chunks = 0
            for src_name, doc_text in all_texts:
                chunks = chunk_text(doc_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                total_chunks += len(chunks)

                st.write(f"• {src_name}: {len(chunks)} chunks")

                # Prepare IDs/texts/metas
                ids = []
                metadatas = []
                for ch in chunks:
                    cid = str(uuid.uuid4())
                    ids.append(cid)
                    metadatas.append({"source": src_name})

                # Embeddings
                embs = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

                # Upsert to Chroma
                collection.upsert(
                    ids=ids,
                    embeddings=embs.tolist(),
                    metadatas=metadatas,
                    documents=chunks
                )

            status.update(label=f"Ingestion complete. Stored {total_chunks} chunks.", state="complete")

# -----------------------------
# Vector DB Inspect
# -----------------------------
st.header("2) Vector DB Overview")
collection = st.session_state.collection

try:
    count = collection.count()
except Exception:
    count = 0

col_a, col_b, col_c = st.columns(3)
col_a.metric("Chunks in DB", f"{count}")
col_b.metric("Embedding model", embed_model_name.split("/")[-1])
col_c.metric("Persist dir", persist_dir)

if st.checkbox("Preview random 5 chunks", value=False):
    if count > 0:
        # Chroma has no random sample API; just get first few
        result = collection.peek()
        docs = result.get("documents", [])[:5]
        metas = result.get("metadatas", [])[:5]
        for i, d in enumerate(docs, 1):
            st.write(f"**{i}.** {d[:500]}{'...' if len(d)>500 else ''}")
            st.caption(metas[i-1])
    else:
        st.info("No chunks yet. Ingest something first.")

st.divider()

# -----------------------------
# RAG Query Interface
# -----------------------------
st.header("3) Ask a Question")
query = st.text_input("Your question", placeholder="Ask something grounded in your uploaded docs...")
top_k = st.slider("Top-K retrieved chunks", 1, 10, 4, 1)

col_run, col_show = st.columns([1,3])
with col_run:
    run_rag = st.button("Run RAG")
with col_show:
    show_debug = st.checkbox("Show retrieved chunks & context sent to LLM", value=True)

if run_rag:
    if not query.strip():
        st.warning("Enter a question.")
    elif count == 0:
        st.warning("Vector DB is empty. Ingest context first.")
    else:
        with st.status("Retrieving relevant chunks...", expanded=True) as s1:
            # Embed the query via the same embedder
            embedder = st.session_state.embedder
            q_emb = embedder.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]

            # Query Chroma by embedding
            res = collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            retrieved = []
            for i in range(len(res["documents"][0])):
                retrieved.append({
                    "id": i,
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "score": res["distances"][0][i],  # lower is closer in Chroma
                })

            s1.update(label="Retrieved chunks ready.", state="complete")

        if show_debug:
            st.subheader("Top matches")
            for i, r in enumerate(retrieved, 1):
                with st.expander(f"{i}. score={r['score']:.4f} · source={r['metadata'].get('source','user-upload')}"):
                    st.write(r["text"])

        # Build prompt & call LLM
        client = st.session_state.client
        prompt = build_rag_prompt(query, retrieved)

        st.subheader("Context sent to LLM")
        if show_debug:
            st.code(prompt[:5000], language="markdown")  # safe truncation

        st.subheader("4) Answer")
        if client is None:
            st.error("No OpenAI client configured. Set OPENAI_API_KEY and rerun.")
        else:
            start = time.time()
            answer = call_llm(client, prompt, model=gen_model_name)
            dur = time.time() - start
            st.write(answer)
            st.caption(f"Generated by {gen_model_name} in {dur:.2f}s")

            st.subheader("Citations")
            st.text(format_citations(retrieved))
