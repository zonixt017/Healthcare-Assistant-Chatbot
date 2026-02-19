# 🚀 Deployment Guide

This guide covers the best deployment options for the Healthcare Assistant Chatbot, with detailed instructions for each platform.

---

## 🏆 Recommended: HuggingFace Spaces

**Why HuggingFace Spaces is the best choice:**

| Feature | HuggingFace Spaces |
|---------|-------------------|
| **Free Tier** | ✅ 16GB RAM, 2 vCPUs |
| **Persistent Storage** | ✅ Yes (for vector store) |
| **Docker Support** | ✅ Docker SDK |
| **Custom Domain** | ✅ Optional |
| **Portfolio Visibility** | ✅ Shows on HF profile |
| **URL Format** | `username-healthcare-assistant.hf.space` |
| **Native Integration** | ✅ Works with HF Inference API |

### Architecture for HF Spaces

```
┌─────────────────────────────────────────────────────────────┐
│                    HuggingFace Space                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Streamlit UI  │  │  FAISS Vector   │  │ Embeddings   │ │
│  │   (app.py)      │  │  Store (local)  │  │ (CPU)        │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┘ │
└───────────┼────────────────────┼─────────────────────────────┘
            │                    │
            ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│              HuggingFace Inference API (Cloud)               │
│              Mistral-7B-Instruct-v0.2                        │
│              (No local model needed!)                        │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step Deployment

#### 1. Prepare Your Repository

Ensure your repo has these files:

```
your-repo/
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── data/
│   └── your-medical-pdf.pdf
└── .env.example (for reference only)
```

#### 2. Create a HuggingFace Account

1. Go to [huggingface.co](https://huggingface.co) and sign up (free)
2. Verify your email

#### 3. Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Owner:** Your username
   - **Space name:** `healthcare-assistant` (or your preference)
   - **License:** MIT
   - **SDK:** Choose **Docker**
   - **Hardware:** CPU basic (free) or upgrade to GPU for faster embeddings
3. Click **Create Space**

#### 4. Set Up Secrets (Environment Variables)

1. Go to your Space → **Settings** → **Variables and secrets**
2. Add a new secret:
   - **Name:** `HUGGINGFACEHUB_API_TOKEN`
   - **Value:** Your HF token (get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
3. Click **Save**

#### 5. Upload Your Code

**Option A: Upload via Web UI**

1. Go to your Space → **Files** tab
2. Click **Add file** → **Upload files**
3. Upload all your project files

**Option B: Push via Git (Recommended)**

```bash
# Clone your Space (replace with your username)
git clone https://huggingface.co/spaces/YOUR_USERNAME/healthcare-assistant

# Copy your project files
cp -r Healthcare-Assistant-Chatbot/* healthcare-assistant/

# Navigate to the Space
cd healthcare-assistant

# Add, commit, push
git add .
git commit -m "Initial deployment"
git push
```

#### 6. Wait for Build

- HuggingFace will automatically build your Docker image
- Check the **Logs** tab for build progress
- First build takes 5-10 minutes

#### 7. Access Your App

Once built, your app will be available at:
```
https://YOUR_USERNAME-healthcare-assistant.hf.space
```

### Important Notes for HF Spaces

1. **Use Cloud LLM:** Set `HUGGINGFACEHUB_API_TOKEN` to use cloud inference (no local model needed)
2. **Vector Store:** First run will build the vector store; subsequent runs use cached version
3. **Memory:** Free tier has 16GB RAM - sufficient for embeddings + FAISS
4. **Cold Starts:** Free tier apps sleep after inactivity; first load takes ~30 seconds

---

## 🔄 Alternative: Render.com

**Best for:** Professional deployment with custom domain

### Pricing
- **Free:** Not available for Docker
- **Starter:** $7/month
- **Standard:** $25/month

### Deployment Steps

1. Go to [render.com](https://render.com) and sign up
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Name:** healthcare-assistant
   - **Region:** Choose closest to you
   - **Branch:** main
   - **Runtime:** Docker
   - **Instance Type:** Starter ($7/mo) or higher
5. Add environment variables:
   - `HUGGINGFACEHUB_API_TOKEN` = your_token
6. Click **Deploy Web Service**

### Render Configuration

Create a `render.yaml` file:

```yaml
services:
  - type: web
    name: healthcare-assistant
    runtime: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: HUGGINGFACEHUB_API_TOKEN
        sync: false
      - key: PYTHON_VERSION
        value: 3.11.0
    plan: starter
    healthCheckPath: /_stcore/health
```

---

## 🚂 Alternative: Railway.app

**Best for:** Quick deployment with good DX

### Pricing
- **Free Trial:** $5 credit (one-time)
- **Hobby:** $5/month (usage-based)

### Deployment Steps

1. Go to [railway.app](https://railway.app) and sign up with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your repository
4. Railway auto-detects Docker
5. Add environment variables:
   - `HUGGINGFACEHUB_API_TOKEN` = your_token
6. Deploy!

### Railway Configuration

Create a `railway.toml` file:

```toml
[build]
builder = "dockerfile"

[deploy]
healthcheckPath = "/_stcore/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
```

---

## ✈️ Alternative: Fly.io

**Best for:** Edge deployment with global distribution

### Pricing
- **Free:** 3 VMs, 3GB volume
- **Paid:** Usage-based

### Deployment Steps

1. Install Fly CLI:
   ```bash
   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   
   # macOS/Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. Login:
   ```bash
   fly auth login
   ```

3. Deploy:
   ```bash
   fly launch
   ```

4. Set secrets:
   ```bash
   fly secrets set HUGGINGFACEHUB_API_TOKEN=your_token
   ```

### Fly.io Configuration

Create a `fly.toml` file:

```toml
app = "healthcare-assistant"
primary_region = "sin"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8501"

[http_service]
  internal_port = 8501
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[http_service.checks]]
  grace_period = "60s"
  interval = "30s"
  method = "GET"
  path = "/_stcore/health"
  timeout = "10s"

[[vm]]
  memory = "2gb"
  cpu_kind = "shared"
  cpus = 2
```

---

## ☁️ Alternative: Google Cloud Run

**Best for:** Enterprise-grade, scalable deployment

### Pricing
- **Free Tier:** 2 million requests/month
- **Paid:** Usage-based

### Deployment Steps

1. Install Google Cloud CLI
2. Authenticate:
   ```bash
   gcloud auth login
   ```

3. Build and deploy:
   ```bash
   gcloud run deploy healthcare-assistant \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars HUGGINGFACEHUB_API_TOKEN=your_token \
     --memory 4Gi \
     --cpu 2
   ```

---

## 📊 Comparison Summary

| Platform | Free Tier | Setup Difficulty | Best For |
|----------|-----------|------------------|----------|
| **HuggingFace Spaces** | ✅ Generous | ⭐ Easy | Portfolio, HF integration |
| Render.com | ❌ No | ⭐ Easy | Professional apps |
| Railway.app | ⚠️ Limited | ⭐ Easy | Quick deployment |
| Fly.io | ✅ Limited | ⭐⭐ Medium | Global distribution |
| Google Cloud Run | ✅ Generous | ⭐⭐⭐ Hard | Enterprise apps |

---

## 🎯 Recommendation

**For your portfolio:** Use **HuggingFace Spaces**

**Reasons:**
1. Completely free with generous resources
2. Native integration with HuggingFace ecosystem
3. Shows on your HF profile (great for portfolio)
4. Easy to share and demo
5. No credit card required

**For production:** Consider **Render.com** or **Google Cloud Run** for:
- Custom domains
- SLA guarantees
- Enterprise features
- Better scaling options

---

## 🔧 Pre-Deployment Checklist

- [ ] Set `HUGGINGFACEHUB_API_TOKEN` in environment variables
- [ ] Ensure PDF files are in `data/` directory
- [ ] Test Docker build locally: `docker build -t test . && docker run -p 8501:8501 test`
- [ ] Verify `.env` is NOT committed (check `.gitignore`)
- [ ] Update README with your Space URL after deployment

---

## 🆘 Troubleshooting

### Build Fails
- Check build logs for specific errors
- Ensure all dependencies in `requirements.txt` are correct
- Try building locally first with Docker

### App Crashes on Start
- Check memory usage - upgrade instance if needed
- Verify environment variables are set correctly
- Check logs for Python errors

### Slow First Load
- Normal for free tiers (cold start)
- Vector store builds on first run
- Subsequent loads are faster

### HuggingFace API Errors
- Verify token is valid and has correct permissions
- Check API rate limits
- Ensure model ID is correct

---

## 📞 Support

- **HuggingFace Docs:** [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Render Docs:** [render.com/docs](https://render.com/docs)
- **Railway Docs:** [docs.railway.app](https://docs.railway.app)
- **Fly.io Docs:** [fly.io/docs](https://fly.io/docs)