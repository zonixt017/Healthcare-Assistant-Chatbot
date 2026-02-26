import os
import sys

# ── GPU / CUDA path fix ────────────────────────────────────────────────────────
# Fix CUDA_PATH if it's incorrectly set to the bin directory instead of the root.
# llama-cpp-python expects CUDA_PATH to point to the CUDA installation root,
# and it appends "\\bin" internally. If CUDA_PATH already ends with "\\bin",
# it results in a double "\\bin\\bin" path which causes import failures.
_cuda_path = os.environ.get("CUDA_PATH", "")
if _cuda_path:
    # Normalize path separators
    _cuda_path_normalized = _cuda_path.replace("/", "\\").rstrip("\\")
    # Check if the last directory component is "bin"
    if os.path.basename(_cuda_path_normalized).lower() == "bin":
        # Strip the bin directory (the library will add it back)
        _cuda_path = os.path.dirname(_cuda_path_normalized)
        os.environ["CUDA_PATH"] = _cuda_path
    else:
        _cuda_path = _cuda_path_normalized
    
    # Verify the CUDA runtime DLL exists before proceeding
    # Check for both CUDA 12 and CUDA 11 runtime DLLs
    cuda_dll_exists = (
        os.path.isfile(os.path.join(_cuda_path, "bin", "cudart64_12.dll")) or
        os.path.isfile(os.path.join(_cuda_path, "bin", "cudart64_11.dll")) or
        os.path.isfile(os.path.join(_cuda_path, "bin", "cudart64_110.dll"))
    )
    if not cuda_dll_exists:
        # CUDA_PATH points to a non-existent or incomplete installation
        os.environ.pop("CUDA_PATH", None)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (all overridable via .env)
# ─────────────────────────────────────────────────────────────────────────────
PDF_DATA_PATH     = os.getenv("PDF_DATA_PATH",     "data/")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vectorstore")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",   "sentence-transformers/all-MiniLM-L6-v2")
HF_INFERENCE_API  = os.getenv("HF_INFERENCE_API",  "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_TIMEOUT    = int(os.getenv("HF_API_TIMEOUT", "45"))
HF_INFERENCE_TASK  = os.getenv("HF_INFERENCE_TASK", "conversational").strip() or "conversational"
LOCAL_LLM_PATH    = os.getenv("LOCAL_LLM_PATH",    "models/phi-2.Q4_K_M.gguf")
RETRIEVER_K       = int(os.getenv("RETRIEVER_K",   "3"))
HF_INFERENCE_FALLBACKS = [
    model.strip() for model in os.getenv("HF_INFERENCE_FALLBACKS", "").split(",") if model.strip()
]
HF_INFERENCE_TASK_FALLBACKS = [
    task.strip() for task in os.getenv("HF_INFERENCE_TASK_FALLBACKS", "text-generation,conversational").split(",") if task.strip()
]

# GPU layers: how many transformer layers to offload to GPU.
# For GTX 1650 4 GB with Phi-2 Q4_K_M (~1.7 GB) → all 32 layers fit.
# For Mistral-7B Q4_K_M (~4.1 GB) → use 20-24 layers (rest on CPU).
# Set to 0 to force CPU-only.
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "32"))

# Embedding device: "cuda" if GPU available, else "cpu"
def _embedding_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"
    except Exception:
        return "cpu"

EMBED_DEVICE = os.getenv("EMBED_DEVICE", "") or _embedding_device()

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthCare Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — cleaner look, better chat bubbles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General ── */
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stSidebar"]          { background: #1e293b; }
[data-testid="stSidebar"] *        { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #334155; border: 1px solid #475569;
    color: #e2e8f0; border-radius: 8px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #ef4444; border-color: #ef4444; color: white;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 4px;
}

/* ── Title ── */
h1 { color: #0f172a !important; }

/* ── Status badges ── */
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 0;
}
.badge-green  { background:#dcfce7; color:#166534; }
.badge-blue   { background:#dbeafe; color:#1e40af; }
.badge-yellow { background:#fef9c3; color:#854d0e; }
.badge-red    { background:#fee2e2; color:#991b1b; }

/* ── Source expander ── */
[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (only pulled in when actually needed)
# ─────────────────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM


class HFInferenceLLM(LLM):
    """Minimal LangChain-compatible wrapper around huggingface_hub InferenceClient."""

    model_id: str
    token: str
    timeout: float
    task: str = "conversational"
    temperature: float = 0.3
    max_new_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "hf-inference-client"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        client = InferenceClient(model=self.model_id, token=self.token, timeout=self.timeout)

        if self.task == "conversational":
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            text = response.choices[0].message.content if response.choices else ""
        else:
            text = client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        if stop:
            for token in stop:
                if token and token in text:
                    text = text.split(token)[0]
                    break

        return text


# ─────────────────────────────────────────────────────────────────────────────
# LLM loader (no Streamlit UI calls inside cached function)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading language model…")
def _load_llm_internal():
    """
    Internal LLM loader without Streamlit UI calls.
    Returns: (llm, mode, error_message, gpu_info)
    - llm: The loaded LLM or None if failed
    - mode: "cloud", "local-gpu(nL)", or "local-cpu"
    - error_message: Error string if failed, None otherwise
    - gpu_info: Dict with GPU details for toast notification
    """
    hf_token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    )
    cloud_error = None

    # ── 1. Cloud API ──────────────────────────────────────────────────────────
    if hf_token:
        candidate_models = [HF_INFERENCE_API, *HF_INFERENCE_FALLBACKS]
        candidate_tasks = [HF_INFERENCE_TASK, *[t for t in HF_INFERENCE_TASK_FALLBACKS if t != HF_INFERENCE_TASK]]
        cloud_errors = []

        for model_id in candidate_models:
            for task_name in candidate_tasks:
                try:
                    llm = HFInferenceLLM(
                        model_id=model_id,
                        token=hf_token,
                        timeout=HF_API_TIMEOUT,
                        task=task_name,
                        temperature=0.3,
                        max_new_tokens=512,
                    )
                    llm.invoke("Reply with exactly: ok")  # quick connectivity check
                    if model_id == HF_INFERENCE_API and task_name == HF_INFERENCE_TASK:
                        mode = "cloud"
                    else:
                        mode = f"cloud-fallback({model_id}|{task_name})"
                    return llm, mode, None, None
                except Exception as e:
                    cloud_errors.append(f"{model_id}|{task_name}: {type(e).__name__}")

        cloud_error = f"Cloud API unavailable ({'; '.join(cloud_errors)})"

    # ── 2. Local GGUF (GPU-accelerated) ──────────────────────────────────────
    if not os.path.exists(LOCAL_LLM_PATH):
        cloud_hint = (
            f"\n\nCloud attempt detail: {cloud_error}. "
            "Check your Hugging Face token and verify `HF_INFERENCE_API` points to a model "
            "available for Inference API/serverless calls."
            if cloud_error else ""
        )
        error_msg = (
            f"Local model not found at `{LOCAL_LLM_PATH}`.\n\n"
            "**Quick fix:** Download a GGUF model and update `LOCAL_LLM_PATH` in your `.env`.\n\n"
            "Recommended for your GTX 1650 (4 GB VRAM):\n"
            "- `phi-2.Q4_K_M.gguf` — best quality/speed balance (~1.7 GB)\n"
            "- `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` — fastest (~0.7 GB)\n\n"
            "See `WIFI_SETUP_GUIDE.md` for download links.\n\n"
            "Tip: on Hugging Face Spaces, add one of these secrets:\n"
            "- `HUGGINGFACEHUB_API_TOKEN` (preferred)\n"
            "- `HF_TOKEN`\n"
            "- `HUGGINGFACE_API_TOKEN`"
            + cloud_hint
        )
        return None, None, error_msg, None

    try:
        from langchain_community.llms import LlamaCpp

        # Detect GPU availability
        gpu_available = False
        gpu_name = ""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        n_gpu = N_GPU_LAYERS if gpu_available else 0
        gpu_info = None
        if gpu_available:
            gpu_info = {"name": gpu_name, "layers": n_gpu}

        llm = LlamaCpp(
            model_path=LOCAL_LLM_PATH,
            temperature=0.2,
            max_tokens=512,
            n_ctx=2048,          # context window (lower = faster, less memory)
            n_batch=256,         # prompt processing batch size
            n_gpu_layers=n_gpu,  # layers offloaded to GPU (0 = CPU only)
            n_threads=max(1, os.cpu_count() - 1),  # leave 1 core for OS
            verbose=False,
            f16_kv=True,         # use fp16 for key/value cache (saves VRAM)
        )
        mode = f"local-gpu({n_gpu}L)" if n_gpu > 0 else "local-cpu"
        return llm, mode, None, gpu_info

    except Exception as e:
        return None, None, f"Failed to load local model: {e}", None


def load_llm():
    """
    Wrapper that handles UI feedback for LLM loading.
    Priority order:
      1. HuggingFace Inference API (cloud) — if token is set and reachable
      2. Local GGUF via LlamaCpp with GPU offloading — fastest local option
    """
    llm, mode, error, gpu_info = _load_llm_internal()
    
    # Handle errors with UI feedback
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    # Show GPU toast notification
    if gpu_info:
        st.toast(f"🚀 GPU detected: {gpu_info['name']} — offloading {gpu_info['layers']} layers", icon="✅")
    
    return llm, mode


# ─────────────────────────────────────────────────────────────────────────────
# Vector-store loader (no Streamlit UI calls inside cached function)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📚 Loading knowledge base…")
def _load_vectorstore_internal():
    """
    Internal vectorstore loader without Streamlit UI calls.
    Returns: (vector_store, source, error_message, build_info)
    - vector_store: FAISS vector store or None if failed
    - source: "cached" or "built"
    - error_message: Error string if failed, None otherwise
    - build_info: Dict with build progress details for UI display
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    # ── Load cached index ─────────────────────────────────────────────────────
    index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    if os.path.exists(index_file):
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        return vector_store, "cached", None, None

    # ── Build from PDFs ───────────────────────────────────────────────────────
    if not os.path.isdir(PDF_DATA_PATH) or not any(
        f.lower().endswith(".pdf") for f in os.listdir(PDF_DATA_PATH)
    ):
        error_msg = (
            f"No PDF files found in `{PDF_DATA_PATH}`. "
            "Please add at least one PDF to the data directory."
        )
        return None, None, error_msg, None

    # Build vectorstore and collect progress info
    loader = DirectoryLoader(PDF_DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    num_pages = len(documents)
    num_pdfs = len(set(d.metadata.get('source', '') for d in documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    num_chunks = len(chunks)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

    build_info = {
        "num_pages": num_pages,
        "num_pdfs": num_pdfs,
        "num_chunks": num_chunks,
        "device": EMBED_DEVICE.upper(),
    }

    return vector_store, "built", None, build_info


def load_vectorstore():
    """
    Wrapper that handles UI feedback for vectorstore loading.
    Load FAISS index from disk (fast), or build it from PDFs on first run.
    Embeddings run on GPU if available, otherwise CPU.
    """
    vector_store, source, error, build_info = _load_vectorstore_internal()
    
    # Handle errors with UI feedback
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    # Show build progress if freshly built
    if build_info:
        with st.status("🔨 Building knowledge base from PDFs…", expanded=True) as status:
            st.write("📄 Loading PDF files…")
            st.write(f"✅ Loaded {build_info['num_pages']} pages from {build_info['num_pdfs']} PDF(s)")
            st.write("✂️ Splitting into chunks…")
            st.write(f"✅ Created {build_info['num_chunks']} chunks")
            st.write(f"🧮 Embedding chunks on {build_info['device']}…")
            st.write("✅ Vector store saved to disk")
            status.update(label="✅ Knowledge base ready!", state="complete")
    
    return vector_store, source


# ─────────────────────────────────────────────────────────────────────────────
# Initialise resources (cached — only runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
llm, llm_source         = load_llm()
vector_store, vs_source = load_vectorstore()

# ─────────────────────────────────────────────────────────────────────────────
# RAG chain construction
# ─────────────────────────────────────────────────────────────────────────────
_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and the latest user question, "
     "rewrite the question as a fully self-contained query that can be "
     "understood without the history. "
     "Do NOT answer — only rewrite if necessary, otherwise return as-is."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

_QA_SYSTEM_PROMPT = (
    "You are a knowledgeable and empathetic healthcare assistant. "
    "Answer the user's question using ONLY the retrieved medical context below. "
    "If the context does not contain enough information, say so clearly and "
    "recommend consulting a qualified healthcare professional. "
    "Never fabricate medical information. "
    "Keep your answer clear, accurate, and concise (3–5 sentences). "
    "Always end with a brief reminder that this is for informational purposes only "
    "and is not a substitute for professional medical advice.\n\n"
    "Context:\n{context}"
)

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM_PROMPT),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_K * 4},
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, _CONTEXTUALIZE_PROMPT)
qa_chain                = create_stuff_documents_chain(llm, _QA_PROMPT)
rag_chain               = create_retrieval_chain(history_aware_retriever, qa_chain)

_SIMPLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly and professional healthcare assistant. "
     "Greet the user warmly and let them know they can ask any medical or health-related question."),
    ("human", "{input}"),
])
simple_chain = _SIMPLE_PROMPT | llm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_INITIAL_GREETING = (
    "Hello! 👋 I'm your **HealthCare Assistant**. "
    "I can answer medical and health-related questions based on a curated medical knowledge base. "
    "How can I help you today?"
)

_GREETINGS = frozenset({
    "hi", "hello", "hey", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "sup", "what's up", "yo",
})

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def _is_greeting(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in _GREETINGS


def _extract_string(response) -> str:
    if isinstance(response, str):
        return response.strip()
    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()


def _build_lc_history(messages: list) -> list:
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def _llm_mode_label(source: str) -> str:
    if source == "cloud":
        return "☁️ Cloud (HuggingFace)"
    if source.startswith("cloud-fallback("):
        value = source[len("cloud-fallback("):-1]
        model, task = value.split("|", 1) if "|" in value else (value, "unknown-task")
        return f"☁️ Cloud fallback ({model}, {task})"
    if source.startswith("local-gpu"):
        layers = source.split("(")[1].rstrip("L)") if "(" in source else "?"
        return f"🚀 Local GPU ({layers} layers offloaded)"
    return "💻 Local CPU"


def _render_sources(sources: list):
    if not sources:
        return
    with st.expander("📄 Source documents", expanded=False):
        for i, src in enumerate(sources, 1):
            page        = src.metadata.get("page", "?")
            source_file = os.path.basename(src.metadata.get("source", "unknown"))
            st.markdown(f"**[{i}] {source_file} — page {page}**")
            snippet = src.page_content[:400]
            if len(src.page_content) > 400:
                snippet += "…"
            st.caption(snippet)
            if i < len(sources):
                st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=72)
    st.title("HealthCare Assistant")
    st.caption("Powered by RAG + LangChain")

    st.divider()

    # ── System Status ─────────────────────────────────────────────────────────
    st.subheader("🔧 System Status")

    llm_label = _llm_mode_label(llm_source)
    badge_cls = "badge-green" if "GPU" in llm_label or "Cloud" in llm_label else "badge-yellow"
    st.markdown(
        f'<span class="status-badge {badge_cls}">LLM: {llm_label}</span>',
        unsafe_allow_html=True,
    )

    vs_label = "📦 Cached" if vs_source == "cached" else "🔨 Freshly built"
    st.markdown(
        f'<span class="status-badge badge-blue">KB: {vs_label}</span>',
        unsafe_allow_html=True,
    )

    embed_badge = "badge-green" if EMBED_DEVICE == "cuda" else "badge-blue"
    st.markdown(
        f'<span class="status-badge {embed_badge}">Embeddings: {EMBED_DEVICE.upper()}</span>',
        unsafe_allow_html=True,
    )

    st.caption(f"Model: `{EMBEDDING_MODEL.split('/')[-1]}`")
    st.caption(f"Retriever: MMR, k={RETRIEVER_K}")

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": _INITIAL_GREETING}
        ]
        st.rerun()

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        import shutil
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # ── About ─────────────────────────────────────────────────────────────────
    st.subheader("ℹ️ About")
    st.markdown(
        "This chatbot answers health-related questions using a curated medical "
        "knowledge base via **Retrieval-Augmented Generation (RAG)**.\n\n"
        "**⚠️ Disclaimer:** For *informational purposes only*. "
        "Does **not** replace professional medical advice."
    )

    st.divider()
    st.caption("v2.0 · [GitHub](https://github.com/zonixt017/Healthcare-Assistant-Chatbot)")


# ─────────────────────────────────────────────────────────────────────────────
# Main chat UI
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": _INITIAL_GREETING}
    ]

st.title("🩺 HealthCare Assistant")
st.caption("Ask me anything about health, symptoms, medications, or medical conditions.")

# ── Render existing chat history ──────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            _render_sources(message["sources"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a health-related question…"):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        sources  = []
        response = ""

        try:
            history = _build_lc_history(st.session_state.messages[:-1])

            if _is_greeting(prompt) or not history:
                # Lightweight path — no retrieval needed
                with st.spinner("💬 Thinking…"):
                    raw      = simple_chain.invoke({"input": prompt})
                    response = _extract_string(raw)
            else:
                # Full RAG path
                with st.spinner("🔍 Searching knowledge base…"):
                    result   = rag_chain.invoke({"input": prompt, "chat_history": history})
                    response = _extract_string(result.get("answer", ""))
                    sources  = result.get("context", [])

                if not response:
                    response = (
                        "I'm sorry, I couldn't find relevant information in my knowledge base. "
                        "Please consult a qualified healthcare professional for this query."
                    )

        except Exception as e:
            response = (
                "⚠️ An error occurred while processing your request. "
                "Please try again or rephrase your question."
            )
            st.error(f"Error details: {e}")

        st.markdown(response)
        _render_sources(sources)

    # Persist assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })
