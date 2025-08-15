# Railway Deployment Guide

This guide will help you deploy your RAG Tutor Chatbot to Railway.

## 🚀 Quick Deploy to Railway

### Method 1: GitHub Integration (Recommended)

1. **Push your code to GitHub** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - RAG Tutor Chatbot"
   git branch -M main
   git remote add origin https://github.com/yourusername/RAG-TUTOR-CHATBOT.git
   git push -u origin main
   ```

2. **Deploy on Railway**:
   - Go to [Railway.app](https://railway.app)
   - Click "Login with GitHub"
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `RAG-TUTOR-CHATBOT` repository
   - Railway will automatically detect it's a Python project and deploy

### Method 2: Railway CLI

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy**:
   ```bash
   railway login
   railway init
   railway up
   ```

## 🔧 Environment Variables Setup

After deployment, you need to set your API keys in Railway:

1. **Go to your Railway project dashboard**
2. **Click on "Variables" tab**
3. **Add the following environment variables**:

```bash
# Required: At least one AI provider
OPENROUTER_API_KEY=your_openrouter_api_key
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

# Optional: For enhanced features
RAPIDAPI_KEY=your_rapidapi_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_google_custom_search_engine_id
```

4. **Click "Deploy" to restart with new environment variables**

## 📁 Files for Railway Deployment

The following files are configured for Railway:

- ✅ **Procfile** - Tells Railway how to start your app
- ✅ **railway.json** - Railway-specific configuration
- ✅ **requirements.txt** - Python dependencies
- ✅ **fastapi_app.py** - Updated to use Railway's PORT environment variable

## 🔍 Verification Steps

1. **Check Build Logs**:
   - Go to your Railway project dashboard
   - Click on "Deployments" tab
   - Check the build and deploy logs for any errors

2. **Test Your API**:
   - Railway will provide a URL like `https://your-app-name.up.railway.app`
   - Test the health endpoint: `https://your-app-name.up.railway.app/health`
   - Test a simple query: `https://your-app-name.up.railway.app/api/chat?question=What%20is%202+2?`

3. **Check Service Health**:
   ```bash
   curl https://your-app-name.up.railway.app/health
   ```

## 🐛 Troubleshooting

### Common Issues:

1. **Build Fails**:
   - Check Railway build logs
   - Ensure all dependencies in `requirements.txt` are correct
   - Verify Python version compatibility

2. **App Crashes on Start**:
   - Check Railway deploy logs
   - Ensure environment variables are set correctly
   - Verify at least one AI provider API key is valid

3. **API Returns Errors**:
   - Check if environment variables are properly set
   - Test individual API keys manually
   - Review application logs in Railway dashboard

### Debug Endpoints:

- **Health Check**: `/health` - Check service status and API keys
- **Debug Info**: `/debug` - Detailed API testing and debugging
- **Service Info**: `/` - Basic service information

## 🔄 Automatic Deployments

Railway will automatically redeploy when you push to your main branch:

```bash
git add .
git commit -m "Update application"
git push origin main
```

## 🌐 Custom Domain (Optional)

1. Go to Railway project settings
2. Click "Domains" tab
3. Add your custom domain
4. Configure DNS records as shown

## 📊 Monitoring

Railway provides:
- **Real-time logs** in the dashboard
- **Metrics** showing CPU, memory, and request data  
- **Deployment history** with rollback capability

## 🔒 Security Notes

- ✅ Environment variables are encrypted at rest
- ✅ HTTPS is automatically provided
- ✅ Private GitHub repositories are supported
- ⚠️ Make sure to keep your API keys secure and never commit them to Git

## 💰 Railway Pricing

- **Starter Plan**: $5/month with generous usage limits
- **Developer Plan**: $20/month for higher usage
- **Team Plans**: Available for organizations

Your RAG Tutor Chatbot should work well within Railway's starter plan limits for moderate usage.

---

## 🎉 You're Ready!

After following these steps, your RAG Tutor Chatbot will be live at:
`https://your-app-name.up.railway.app`

The API will be available at all the same endpoints as local development:
- `POST /api/chat` - Main chat endpoint
- `GET /api/chat?question=...` - Quick queries  
- `GET /health` - Health check
- `GET /debug` - Debug information
