# 🚀 Deployment Guide (Production + Free Demo Hosting)

This project is already structured for deployment with Docker and environment variables. This guide focuses on **free or near-free** ways to put your chatbot online so you can share a live demo.

---

## 1) Production Readiness Checklist

Before deploying publicly, make sure you have the following:

- [ ] `HUGGINGFACEHUB_API_TOKEN` set as a secret on your hosting platform.
- [ ] At least one PDF in `data/` for your knowledge base.
- [ ] `vectorstore/` is **not** committed (it is generated automatically).
- [ ] `models/` is **not** required in cloud deploy (cloud inference is recommended for free hosting).
- [ ] `PORT` is handled dynamically (already handled by `start.sh`).
- [ ] A health endpoint exists (`/_stcore/health`).

---

## 2) Best Free Option: Hugging Face Spaces (Docker)

### Why this is best for demos
- Free CPU hosting for public projects.
- Built-in HTTPS URL.
- Docker-supported workflow.
- Great for AI app showcases.

### Steps
1. Create a new Space at https://huggingface.co/new-space
2. Choose:
   - **SDK**: Docker
   - **Hardware**: CPU Basic (free)
3. Push this repository to the Space.
4. Add this secret in **Settings → Variables and secrets**:
   - `HUGGINGFACEHUB_API_TOKEN`
5. Wait for build, then open your public URL:
   - `https://<username>-<space-name>.hf.space`

### Do I need to upload a local LLM (`.gguf`) to Hugging Face Spaces?
- **No, not for your current hybrid setup.**
- In free-tier hosting, prefer cloud inference via:
  - `HUGGINGFACEHUB_API_TOKEN` (secret)
  - `HF_INFERENCE_API` (model ID)
- Uploading local model weights is optional and usually not ideal for free CPUs due to size + startup time.

### Notes
- First run may be slower while embeddings/vector index are built.
- Free Spaces may sleep when idle (cold starts are normal).

---

## 3) Another Free Option: Fly.io (with usage limits)

Fly.io can be free for very small workloads depending on current credits/promotions. Treat as **low-cost**, not guaranteed permanent free.

### Steps
```bash
fly auth login
fly launch --no-deploy
fly secrets set HUGGINGFACEHUB_API_TOKEN=your_token
fly deploy
```

### URL
- Fly gives you a public HTTPS URL like:
  - `https://<app-name>.fly.dev`

---

## 4) Easy Public Demo Alternatives (No backend infra)

If you only need a quick demo and are fine with occasional downtime:

- **Streamlit Community Cloud** (free)  
  Works best when dependencies are light. This app can run, but build time may be high due to ML packages.

- **Google Cloud Run / Railway / Render**  
  Usually paid or trial-based now; good for stable demos if budget allows.

---

## 5) Environment Variables You Should Set

Minimum for public deployment:

```env
HUGGINGFACEHUB_API_TOKEN=hf_xxx
HF_INFERENCE_API=mistralai/Mistral-7B-Instruct-v0.2
HF_API_TIMEOUT=45
PDF_DATA_PATH=data/
VECTOR_STORE_PATH=vectorstore
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVER_K=3
EMBED_DEVICE=cpu
N_GPU_LAYERS=0
```

For free hosting, keep `EMBED_DEVICE=cpu` and `N_GPU_LAYERS=0`.

### About model callability
Some model IDs may exist on Hugging Face but still fail on serverless inference depending on provider availability, limits, or permissions.
If calls fail, switch `HF_INFERENCE_API` to another instruct model that is currently available for Inference API usage.

---

## 6) Local Smoke Test Before Deploy

```bash
cp .env.example .env
# edit .env with your HF token

pip install -r requirements.txt
streamlit run app.py
```

Docker test:

```bash
docker build -t healthcare-assistant .
docker run --rm -p 7860:7860 --env-file .env healthcare-assistant
```

Open: `http://localhost:7860`

---

## 7) Security + Reliability Notes

- Do **not** commit `.env` or tokens.
- Keep secrets in host platform secret manager only.
- Keep your disclaimer visible in UI (already present).
- Prefer cloud inference token over bundling local LLM weights in production.
- Monitor logs for model API timeouts or rate limits.

---

## 8) Suggested “Showcase” Setup (Free)

For portfolio/demo use:

1. Deploy to **Hugging Face Spaces**.
2. Add screenshots/GIF + live link in `README.md`.
3. Add sample prompts users can try.
4. Keep one stable medical PDF in `data/` for predictable responses.

This gives you a public URL you can share in resumes, LinkedIn, and project portfolios.
