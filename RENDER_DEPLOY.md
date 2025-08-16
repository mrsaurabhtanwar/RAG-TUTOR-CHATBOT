# 🚀 Render Deployment Guide

Deploy your RAG Tutor Chatbot to Render for free!

## 📋 Prerequisites

- GitHub account with your code
- Render account (free at [render.com](https://render.com))
- API keys ready (at least one AI provider)

## 🎯 Quick Deploy to Render

### Method 1: Web Service (Recommended)

1. **Go to [Render.com](https://render.com)**
2. **Sign up/Login** with GitHub
3. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `RAG-TUTOR-CHATBOT`
   - Configure deployment settings

### Method 2: Auto-Deploy with render.yaml

Your repository includes a `render.yaml` file for automatic configuration.

## ⚙️ Render Configuration

### Basic Settings:
```
Name: rag-tutor-chatbot
Environment: Python
Build Command: ./build.sh
Start Command: python fastapi_app.py
```

### Advanced Settings:
```
Health Check Path: /health
Auto-Deploy: Yes (on git push)
Plan: Free (sufficient for moderate usage)
```

## 🔑 Environment Variables Setup

**In Render Dashboard, add these environment variables:**

### Required (at least one):
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
GROQ_API_KEY=gsk_your-groq-key-here
HUGGINGFACE_API_KEY=hf_your-hf-key-here
```

### Optional (enhances functionality):
```bash
RAPIDAPI_KEY=your-rapidapi-key
GOOGLE_API_KEY=AIzaSy-your-google-key
GOOGLE_CX=your-custom-search-engine-id
```

### System Variables (Render auto-sets):
```bash
PORT=10000  # Render assigns this automatically
PYTHON_VERSION=3.11.0
```

## 📁 Files for Render Deployment

Your repository now includes:

- ✅ **`build.sh`** - Build script for Render
- ✅ **`render.yaml`** - Render service configuration
- ✅ **`requirements.txt`** - Python dependencies
- ✅ **`fastapi_app.py`** - Updated with PORT handling
- ✅ **`Procfile`** - Can be used alternatively

## 🔧 Step-by-Step Deployment

### Step 1: Push Code to GitHub

```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### Step 2: Create Render Service

1. **Login to Render**: [render.com](https://render.com)
2. **New Web Service**: Click "New +" → "Web Service"
3. **Connect Repository**: Select your `RAG-TUTOR-CHATBOT` repo
4. **Configure Service**:
   ```
   Name: rag-tutor-chatbot
   Environment: Python
   Build Command: ./build.sh
   Start Command: python fastapi_app.py
   ```

### Step 3: Set Environment Variables

In Render service settings:
1. Go to "Environment" tab
2. Add your API keys one by one
3. Click "Save Changes"

### Step 4: Deploy

1. Click "Create Web Service"
2. Render will automatically:
   - Clone your repository
   - Run the build script
   - Install dependencies
   - Start your application

## 🔍 Verification

### Check Deployment Status:
1. **Build Logs**: Monitor build progress
2. **Deploy Logs**: Watch application startup
3. **Live Logs**: See runtime logs

### Test Your API:
Your app will be available at: `https://your-service-name.onrender.com`

**Test endpoints:**
```bash
# Health check
curl https://your-service-name.onrender.com/health

# Simple question
curl "https://your-service-name.onrender.com/api/chat?question=What%20is%202+2?"

# Full API test
curl -X POST https://your-service-name.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain photosynthesis", "max_tokens": 500}'
```

## 🐛 Troubleshooting

### Common Issues:

1. **Build Fails**:
   ```bash
   # Check build logs in Render dashboard
   # Common fixes:
   chmod +x build.sh  # Make build script executable
   ```

2. **App Won't Start**:
   - Check deploy logs
   - Verify environment variables are set
   - Test at least one AI provider key

3. **Timeouts**:
   - Render free tier has request timeouts
   - Optimize your API calls
   - Add retry logic

### Render Free Tier Limits:
- **750 hours/month** (enough for full-time operation)
- **Request timeout**: 30 seconds
- **Sleep after inactivity**: Spins down after 15 minutes
- **Build time**: 15 minutes max

## 🔄 Auto-Deployment

Render automatically redeploys when you push to GitHub:

```bash
# Make changes
git add .
git commit -m "Update application"
git push origin main
# Render auto-deploys!
```

## 🌐 Custom Domain (Optional)

Render free tier includes:
- HTTPS automatically
- `your-app.onrender.com` subdomain
- Custom domains available on paid plans

## 📊 Monitoring

Render provides:
- **Real-time logs** in dashboard
- **Metrics** showing requests and performance
- **Health checks** with automatic restarts
- **Email alerts** for deployment failures

## 🔒 Security

- ✅ Environment variables encrypted
- ✅ HTTPS enabled by default
- ✅ Regular security updates
- ✅ Private GitHub repos supported (paid plans)

## 💰 Render Pricing

- **Free Tier**: Perfect for development and light usage
- **Starter**: $7/month for production apps
- **Standard**: $25/month for high-traffic apps

## 📈 Performance Tips

1. **Cold Starts**: Free tier apps sleep after 15 min inactivity
2. **Keep Alive**: Use uptimerobot.com to ping your app
3. **Caching**: Implement response caching for common queries
4. **Optimize**: Use async operations efficiently

## 🎉 Deployment Complete!

After following these steps, your RAG Tutor Chatbot will be live at:
`https://your-service-name.onrender.com`

### Available Endpoints:
- `POST /api/chat` - Main chat interface
- `GET /api/chat?question=...` - Quick queries
- `GET /health` - Service health check
- `GET /debug` - API debugging info
- `GET /` - Service information

---

## 🆘 Need Help?

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Community**: Render Community Forum
- **Support**: Render support for paid plans

Your educational AI tutor is now accessible worldwide! 🎓✨
