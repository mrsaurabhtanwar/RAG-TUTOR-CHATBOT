# 🚀 Render Free Tier Deployment Guide

## ✅ Project Status
Your RAG-TUTOR-CHATBOT has been optimized for Render free tier and uploaded to GitHub!

**Repository**: https://github.com/mrsaurabhtanwar/RAG-TUTOR-CHATBOT

## 🎯 What's Optimized for Free Tier

### ✅ **Optimizations Made:**
- **Removed ML Dependencies**: sentence-transformers, faiss, numpy (saves ~200MB RAM)
- **Streamlined Requirements**: Only essential packages for basic AI chat
- **Optimized Build Script**: Faster build process
- **Basic Mode**: App runs without RAG functionality but with full AI chat

### ⚠️ **Free Tier Limitations:**
- **No RAG**: Context-aware responses disabled
- **512MB RAM**: Limited memory
- **0.1 CPU**: Slower response times (5-15 seconds)
- **15min Sleep**: App sleeps after inactivity
- **90min Build**: Build timeout limit

## 🚀 Deployment Steps

### 1. **Go to Render Dashboard**
- Visit: https://dashboard.render.com/
- Sign in with your GitHub account

### 2. **Create New Web Service**
- Click "New +" → "Web Service"
- Connect your GitHub repository: `mrsaurabhtanwar/RAG-TUTOR-CHATBOT`

### 3. **Configure Service**
```
Name: rag-tutor-chatbot
Environment: Python 3
Build Command: chmod +x build.sh && ./build.sh
Start Command: uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT
```

### 4. **Set Environment Variables**
In Render dashboard, add these environment variables:

**Required:**
```
OPENROUTER_API_KEY=your_openrouter_key_here
```

**Optional (for enhanced features):**
```
GROQ_API_KEY=your_groq_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
RAPIDAPI_KEY=your_rapidapi_key_here
GOOGLE_API_KEY=your_google_key_here
GOOGLE_CX=your_google_cx_here
```

### 5. **Deploy**
- Click "Create Web Service"
- Wait for build to complete (5-10 minutes)
- Your API will be available at: `https://your-app-name.onrender.com`

## 🧪 Testing Your Deployment

### Health Check
```bash
curl https://your-app-name.onrender.com/health
```

### Test Chat API
```bash
curl -X POST "https://your-app-name.onrender.com/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is 2+2?", "max_tokens": 100}'
```

### Interactive Docs
Visit: `https://your-app-name.onrender.com/docs`

## 📊 Expected Performance

### **Free Tier Performance:**
- **Response Time**: 5-15 seconds
- **Concurrent Users**: 1-2 users
- **Uptime**: 99% (with 15min sleep periods)
- **Features**: Basic AI chat + resource suggestions

### **What Works:**
- ✅ AI-powered tutoring responses
- ✅ Video and website resource suggestions
- ✅ Multiple AI provider fallbacks
- ✅ Rate limiting and security
- ✅ Health monitoring
- ✅ Quiz generation endpoint

### **What's Disabled:**
- ❌ RAG (context-aware responses)
- ❌ Vector similarity search
- ❌ Document embeddings

## 🔧 Troubleshooting

### **Build Failures:**
- Check build logs in Render dashboard
- Ensure all environment variables are set
- Verify Python version (3.11.10)

### **Slow Responses:**
- Normal for free tier (0.1 CPU)
- Consider upgrading to Starter plan ($7/month)

### **App Sleeping:**
- Free tier sleeps after 15min inactivity
- First request after sleep takes 30-60 seconds
- Use uptime monitoring to keep app warm

## 💡 Upgrade Options

### **Starter Plan ($7/month):**
- 512MB RAM, 0.5 CPU
- No sleep, better performance
- Can enable RAG functionality

### **Standard Plan ($25/month):**
- 1GB RAM, 1 CPU
- Full RAG capabilities
- 20+ concurrent users

## 🎉 Success!

Your RAG-TUTOR-CHATBOT is now ready for deployment on Render free tier!

**Next Steps:**
1. Deploy using the steps above
2. Test all endpoints
3. Integrate with your EduPlatform
4. Monitor performance and upgrade if needed

**Repository**: https://github.com/mrsaurabhtanwar/RAG-TUTOR-CHATBOT
**Render Dashboard**: https://dashboard.render.com/
