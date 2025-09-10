# 🤖 RAG Tutor Chatbot

A production-ready AI tutoring service with Retrieval-Augmented Generation (RAG) capabilities, built with FastAPI and optimized for deployment on Render.

## ✨ Features

- **🧠 Multi-AI Provider Support**: Groq (primary), OpenRouter, HuggingFace
- **🔍 RAG System**: Vector similarity search with educational content
- **⚡ High Performance**: Caching, rate limiting, and optimized responses
- **🛡️ Production Ready**: Security middleware, structured logging, metrics
- **📚 Educational Focus**: Specialized for tutoring with learning suggestions
- **🌐 Resource Integration**: YouTube and website suggestions for each response

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG-TUTOR-CHATBOT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python fastapi_app.py
   ```

4. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Chat Endpoint: http://localhost:8000/api/chat

### Render Deployment

1. **Connect your GitHub repository to Render**

2. **Create a new Web Service**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python fastapi_app.py`
   - Environment: Python 3

3. **Set Environment Variables** (optional)
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - `HUGGINGFACE_API_KEY`: Your HuggingFace API key
   - `GOOGLE_API_KEY`: Your Google API key
   - `GOOGLE_CX`: Your Google Custom Search Engine ID

4. **Deploy!** 🎉

## 📡 API Endpoints

### Chat Endpoint
```bash
POST /api/chat
```

**Request Body:**
```json
{
  "question": "What is photosynthesis?",
  "include_context": true,
  "max_tokens": 1500
}
```

**Response:**
```json
{
  "answer": "Photosynthesis is the process by which plants...",
  "videoLink": "https://www.youtube.com/watch?v=...",
  "websiteLink": "https://example.com/...",
  "hasContext": true,
  "processingTime": 2.34,
  "apiUsed": "Groq",
  "suggestions": [
    "Create a study plan for biology topics",
    "Look for visual explanations of the process"
  ],
  "confidence_score": 0.95
}
```

### Health Check
```bash
GET /health
```

### Debug Information
```bash
GET /debug
```

### System Metrics
```bash
GET /metrics
```

## 🔧 Configuration

### API Keys

The system uses the following APIs in order of priority:

1. **Groq** (Primary) - Already configured
2. **OpenRouter** - Set `OPENROUTER_API_KEY` environment variable
3. **HuggingFace** - Set `HUGGINGFACE_API_KEY` environment variable

### RAG System

The RAG system includes pre-loaded educational content covering:
- Mathematics fundamentals
- Python programming
- Physics concepts
- Chemistry basics
- Biology (cell structure)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│   RAG System    │────│  Vector Store   │
│                 │    │                 │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │              ┌─────────────────┐
         │              │  AI Providers   │
         │              │  (Groq, etc.)   │
         │              └─────────────────┘
         │
┌─────────────────┐    ┌─────────────────┐
│  Resource Finder│────│  YouTube/Web    │
│                 │    │   Suggestions   │
└─────────────────┘    └─────────────────┘
```

## 📊 Performance Features

- **Rate Limiting**: 100 requests per hour per IP
- **Caching**: Context retrieval and API responses
- **Retry Logic**: Exponential backoff for API failures
- **Metrics**: Request tracking and performance monitoring
- **Security**: CORS, trusted hosts, input validation

## 🧪 Testing

Test the API with a simple request:

```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is 2+2?", "max_tokens": 100}'
```

## 📈 Monitoring

The application provides comprehensive monitoring:

- **Health Check**: `/health` - Service status and API key status
- **Debug Info**: `/debug` - API connectivity tests
- **Metrics**: `/metrics` - Request statistics and performance data
- **Logs**: Structured logging with request tracking

## 🔒 Security

- Input sanitization and validation
- Rate limiting to prevent abuse
- CORS configuration
- Trusted host middleware
- Secure API key handling

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support or questions:
- Create an issue in the repository
- Check the `/health` endpoint for system status
- Review logs for debugging information

---

**Built with ❤️ for education and learning**