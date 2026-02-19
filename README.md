# 🩺 HealthCare Assistant Chatbot

An AI-powered medical chatbot built with **Retrieval-Augmented Generation (RAG)**. It answers health and medical questions grounded in a real PDF knowledge base, with conversational memory and source attribution.

> ⚠️ **Disclaimer:** This tool is for *informational purposes only* and does **not** replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **GPU Accelerated** | Runs on GTX 1650 (4GB VRAM) with CUDA offloading for fast inference |
| 🔍 **RAG Pipeline** | Answers grounded in retrieved documents — no hallucination |
| 🧠 **Conversational Memory** | Multi-turn chat with context-aware follow-up questions |
| ☁️ **Cloud + Local LLM** | HuggingFace Inference API with automatic fallback to local GGUF |
| 📄 **Source Attribution** | Every answer shows exact PDF pages it was drawn from |
| 🎯 **MMR Retrieval** | Maximal Marginal Relevance for diverse, non-redundant context |
| 🖥️ **Modern UI** | Clean Streamlit interface with status badges and controls |
| 🐳 **Docker Ready** | One-command deployment to any cloud platform |

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                 │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│              History-Aware Retriever (LangChain)         │
│  Rewrites question using chat history into standalone   │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│           FAISS Vector Store (MMR Search)               │
│  Retrieves top-k diverse chunks from embedded PDFs      │
└─────────────────────────────────────────────────────────┘
      │  retrieved context
      ▼
┌─────────────────────────────────────────────────────────┐
│                        LLM                               │
│  ☁️ HuggingFace Inference API (Mistral-7B)             │
│  🚀 Local GGUF + GPU (Phi-2 / TinyLlama / Mistral)     │
└─────────────────────────────────────────────────────────┘
      │
      ▼
   Answer  +  Source Documents
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Git
- (Optional) NVIDIA GPU with CUDA support for local inference

### 1. Clone & Setup

```bash
git clone https://github.com/zonixt017/Healthcare-Assistant-Chatbot.git
cd Healthcare-Assistant-Chatbot

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your HuggingFace token (free at https://huggingface.co/settings/tokens):

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

### 3. Add PDF Knowledge Base

Place your medical PDF files in the `data/` directory:

```
data/
└── your-medical-reference.pdf
```

### 4. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🖥️ GPU Acceleration (Recommended)

For **much faster** local inference on your GTX 1650 (4GB VRAM), see **[WIFI_SETUP_GUIDE.md](WIFI_SETUP_GUIDE.md)** for:

1. Installing CUDA-enabled PyTorch
2. Installing CUDA-enabled llama-cpp-python
3. Downloading optimized GGUF models

### Recommended Models for GTX 1650 (4GB VRAM)

| Model | Size | Quality | GPU Layers | Speed |
|-------|------|---------|------------|-------|
| **Phi-2 Q4_K_M** | 1.7 GB | ⭐⭐⭐⭐ | 32 (all) | ~20 tok/s |
| TinyLlama Q4_K_M | 0.7 GB | ⭐⭐⭐ | 22 (all) | ~40 tok/s |
| Mistral-7B Q4_K_M | 4.1 GB | ⭐⭐⭐⭐⭐ | 20-24 | ~10 tok/s |

---

## 📂 Project Structure

```
Healthcare-Assistant-Chatbot/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker Compose for local dev
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── WIFI_SETUP_GUIDE.md       # GPU setup instructions
├── .streamlit/
│   └── config.toml           # Streamlit theme & settings
├── data/                     # PDF knowledge base
│   └── *.pdf
├── models/                   # GGUF models (download separately)
│   └── *.gguf
├── vectorstore/              # FAISS index (auto-generated)
└── extras/                   # Project docs & assets
```

---

## ⚙️ Configuration

All settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACEHUB_API_TOKEN` | *(none)* | HF token for cloud inference |
| `HF_INFERENCE_API` | `mistralai/Mistral-7B-Instruct-v0.2` | Cloud model ID |
| `LOCAL_LLM_PATH` | `models/phi-2.Q4_K_M.gguf` | Local GGUF model path |
| `N_GPU_LAYERS` | `32` | GPU layers to offload (0 = CPU) |
| `PDF_DATA_PATH` | `data/` | PDF directory |
| `VECTOR_STORE_PATH` | `vectorstore` | FAISS index location |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RETRIEVER_K` | `3` | Chunks to retrieve |

---

## 🌐 Deployment

> 📖 **See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.**

### 🏆 Recommended: HuggingFace Spaces (Free)

**Why HuggingFace Spaces?**
- ✅ **Free tier** with 16GB RAM, 2 vCPUs
- ✅ **Persistent storage** for vector store
- ✅ **Native integration** with HuggingFace Inference API
- ✅ **Portfolio visibility** - shows on your HF profile
- ✅ **No credit card required**

**Quick Deploy:**

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
3. Select **Docker** as SDK
4. Upload your code or connect GitHub repo
5. Add secret: `HUGGINGFACEHUB_API_TOKEN` = your_token
6. Deploy! Your app will be at `username-healthcare-assistant.hf.space`

### Alternative Platforms

| Platform | Free Tier | Best For |
|----------|-----------|----------|
| **HuggingFace Spaces** | ✅ Generous | Portfolio, HF integration |
| **Render.com** | ❌ $7+/mo | Professional apps |
| **Railway.app** | ⚠️ $5 credit | Quick deployment |
| **Fly.io** | ✅ Limited | Global distribution |
| **Google Cloud Run** | ✅ Generous | Enterprise apps |

### Docker (Local / Any Cloud)

```bash
# Build
docker build -t healthcare-chatbot .

# Run locally
docker run -p 7860:7860 \
  -e HUGGINGFACEHUB_API_TOKEN=your_token \
  healthcare-chatbot

# Or with docker-compose
docker-compose up -d
```

### Platform-Specific Configs

This project includes ready-to-use configuration files:

- `render.yaml` - Render.com deployment
- `fly.toml` - Fly.io deployment
- `docker-compose.yml` - Docker Compose for local/any cloud

---

## 🔒 Security

- **Never commit `.env`** — it's in `.gitignore`
- **Never commit `models/`** — GGUF files are large binaries
- **Rotate tokens** immediately if accidentally exposed

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| UI | [Streamlit](https://streamlit.io/) |
| Orchestration | [LangChain](https://python.langchain.com/) |
| Embeddings | [sentence-transformers](https://huggingface.co/sentence-transformers) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| Cloud LLM | [HuggingFace Inference API](https://huggingface.co/inference-api) |
| Local LLM | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) |
| PDF Parsing | [pypdf](https://pypdf.readthedocs.io/) |

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient local inference
- [HuggingFace](https://huggingface.co/) for model hosting and inference API