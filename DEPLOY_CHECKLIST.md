# 📋 Railway Deployment Checklist

## Pre-Deployment Setup ✅

- [x] **Procfile created** - Contains web server startup command
- [x] **railway.json created** - Railway-specific configuration  
- [x] **requirements.txt ready** - All Python dependencies listed
- [x] **FastAPI app updated** - Uses Railway's PORT environment variable
- [x] **Environment variables prepared** - API keys ready to be set

## Deployment Steps 📤

### Step 1: Repository Setup
- [ ] Code pushed to GitHub repository
- [ ] Repository is public or accessible to Railway
- [ ] All files committed including Procfile and railway.json

### Step 2: Railway Setup  
- [ ] Account created at [railway.app](https://railway.app)
- [ ] GitHub connected to Railway account
- [ ] Project created from GitHub repository

### Step 3: Environment Configuration
- [ ] **OPENROUTER_API_KEY** set (or GROQ_API_KEY or HUGGINGFACE_API_KEY)
- [ ] **RAPIDAPI_KEY** set (optional - for YouTube search)
- [ ] **GOOGLE_API_KEY** set (optional - for website search)  
- [ ] **GOOGLE_CX** set (optional - for website search)

### Step 4: Deployment Verification
- [ ] Build completed successfully (check Railway logs)
- [ ] Application started without errors
- [ ] Health endpoint responding: `https://your-app.up.railway.app/health`
- [ ] API working: `https://your-app.up.railway.app/api/chat?question=test`

## Required Files for Railway ✅

```
RAG-TUTOR-CHATBOT/
├── Procfile                 ✅ Created
├── railway.json             ✅ Created  
├── requirements.txt         ✅ Ready
├── fastapi_app.py          ✅ Updated with PORT handling
├── .env.example            ✅ Template for local dev
├── README.md               ✅ Updated with Railway info
└── RAILWAY_DEPLOY.md       ✅ Detailed deployment guide
```

## Environment Variables Required 🔑

### Minimum (at least one):
- `OPENROUTER_API_KEY` - OpenRouter AI API key
- `GROQ_API_KEY` - Groq AI API key  
- `HUGGINGFACE_API_KEY` - HuggingFace API key

### Optional (enhances functionality):
- `RAPIDAPI_KEY` - For YouTube video search
- `GOOGLE_API_KEY` - For website search
- `GOOGLE_CX` - Google Custom Search Engine ID

## Testing Endpoints 🧪

Once deployed, test these URLs (replace with your Railway URL):

### Health Check
```
GET https://your-app.up.railway.app/health
```
Expected: `{"status": "healthy", "api_keys": {...}}`

### Simple Question
```  
GET https://your-app.up.railway.app/api/chat?question=What%20is%202+2?
```
Expected: JSON response with answer, video link, website link

### Complex Question
```
POST https://your-app.up.railway.app/api/chat
Content-Type: application/json

{
  "question": "Explain photosynthesis",
  "max_tokens": 800
}
```

## Troubleshooting 🐛

### Build Issues:
- Check Railway build logs in dashboard
- Verify requirements.txt has correct dependencies
- Ensure Python version is supported

### Runtime Issues:
- Check Railway deploy logs  
- Verify environment variables are set
- Test API keys manually

### API Issues:
- Use `/debug` endpoint to test AI providers
- Check API key validity and quotas
- Verify network connectivity to external APIs

## Post-Deployment 🎉

- [ ] Custom domain configured (optional)
- [ ] Monitoring set up (Railway provides metrics)
- [ ] Auto-deployment configured (pushes to main branch)
- [ ] Documentation updated with live URL

---

**Your RAG Tutor Chatbot will be live at:**  
`https://your-app-name.up.railway.app`

**Ready to deploy! 🚀**
