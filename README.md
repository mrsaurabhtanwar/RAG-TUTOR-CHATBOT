# 🎓 RAG Tutor Chatbot - Production-Ready AI System

A comprehensive, production-ready AI-powered tutoring service built with FastAPI that provides intelligent responses using Retrieval-Augmented Generation (RAG). The system leverages multiple AI providers, vector similarity search, caching, and robust error handling to deliver accurate educational responses with multimedia resources.

## ✨ Features

### 🤖 AI & RAG Capabilities
- **True RAG Implementation**: Vector similarity search using FAISS and sentence transformers
- **Multi-AI Provider Support**: OpenRouter, Groq, and HuggingFace with intelligent fallback
- **Context-Aware Responses**: Retrieves relevant educational content for enhanced answers
- **Embedding Caching**: Optimized performance with cached embeddings and responses

### 🛡️ Production-Ready Features
- **Rate Limiting**: Configurable rate limiting per client
- **Input Validation**: Comprehensive sanitization and validation
- **Structured Logging**: Detailed logging with configurable levels
- **Metrics Collection**: Performance monitoring and analytics
- **Health Checks**: Comprehensive health and readiness probes
- **Security**: CORS protection, input sanitization, and secure API key handling

### 📚 Educational Features
- **Dynamic Response Adaptation**: Adjusts response length based on question complexity
- **Resource Integration**: Automatic YouTube video and educational website suggestions
- **Learning Suggestions**: Personalized study recommendations
- **Subject-Specific Context**: Specialized handling for math, science, programming, etc.

### 🔧 Technical Features
- **Async/Await**: High-performance asynchronous processing
- **Retry Logic**: Exponential backoff with jitter for API calls
- **Circuit Breaker**: Graceful degradation when services fail
- **Caching Layer**: Redis-based caching for improved performance
- **Vector Store**: FAISS-based similarity search with persistence

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- Redis (optional, for advanced caching)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd RAG-TUTOR-CHATBOT
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv myenv
   
   # On Windows
   myenv\Scripts\activate
   
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Required: At least one AI provider API key
   OPENROUTER_API_KEY=your_openrouter_api_key
   GROQ_API_KEY=your_groq_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   
   # Optional: For enhanced video/website search
   RAPIDAPI_KEY=your_rapidapi_key
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CX=your_google_custom_search_engine_id
   
   # Production settings
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
   REDIS_URL=redis://localhost:6379
   LOG_LEVEL=INFO
   ```

5. **Run the application**
   ```bash
   python fastapi_app.py
   ```

The service will be available at `http://localhost:8000`

## 📡 API Endpoints

### Chat Endpoints

#### POST `/api/chat`
Send a question and receive an AI-generated response with RAG context.

**Request Body:**
```json
{
  "question": "Explain photosynthesis",
  "include_context": true,
  "max_tokens": 1500
}
```

**Response:**
```json
{
  "answer": "Photosynthesis is the process by which plants...",
  "videoLink": "https://www.youtube.com/watch?v=...",
  "websiteLink": "https://example.com/photosynthesis",
  "hasContext": true,
  "processingTime": 2.34,
  "apiUsed": "OpenRouter",
  "suggestions": [
    "Create a study plan for biology topics",
    "Look for visual explanations of the process"
  ],
  "context_sources": ["Biology: Cell Structure", "Physics Core Concepts"],
  "confidence_score": 0.9,
  "debug_info": {...}
}
```

#### GET `/api/chat`
Simplified GET endpoint for quick queries.

**Parameters:**
- `question` (required): The question to ask

**Example:**
```
GET /api/chat?question=What is 2+2?
```

### Monitoring Endpoints

#### GET `/health`
Comprehensive health check with system status.

#### GET `/metrics`
Performance metrics and analytics.

#### GET `/debug`
Detailed debugging information and API testing.

#### GET `/`
Service information and available endpoints.

## 🧠 RAG System Architecture

### Components

1. **EmbeddingManager**: Handles text embeddings using sentence transformers
2. **VectorStore**: FAISS-based similarity search with persistence
3. **RAGProcessor**: Orchestrates retrieval and context generation
4. **AIProvider**: Multi-provider AI integration with fallback
5. **ResourceFinder**: Educational resource discovery

### How It Works

1. **Query Processing**: User question is sanitized and validated
2. **Embedding Generation**: Question is converted to vector representation
3. **Similarity Search**: FAISS finds relevant educational content
4. **Context Enhancement**: Retrieved content is used to enhance AI prompt
5. **Response Generation**: AI generates context-aware response
6. **Resource Discovery**: Relevant videos and websites are found
7. **Caching**: Results are cached for future similar queries

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | No | None |
| `GROQ_API_KEY` | Groq API key | No | None |
| `HUGGINGFACE_API_KEY` | HuggingFace API key | No | None |
| `RAPIDAPI_KEY` | RapidAPI key for YouTube search | No | None |
| `GOOGLE_API_KEY` | Google API key | No | None |
| `GOOGLE_CX` | Google Custom Search Engine ID | No | None |
| `ALLOWED_ORIGINS` | CORS allowed origins | No | localhost:3000,8000 |
| `REDIS_URL` | Redis connection URL | No | None |
| `LOG_LEVEL` | Logging level | No | INFO |

### Rate Limiting

Default rate limits:
- 100 requests per hour per client
- Configurable via `RateLimiter` class

### Caching

- **Embedding Cache**: In-memory cache for embeddings
- **Response Cache**: Redis-based caching (optional)
- **Context Cache**: Cached similarity search results

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest test_fast_app.py -v

# Run specific test categories
pytest test_fast_app.py::TestAPIEndpoints -v
pytest test_fast_app.py::TestIntegration -v
```

### Test Coverage
- **API Endpoints**: All endpoints tested with various scenarios
- **RAG Functionality**: Vector search and context retrieval
- **Error Handling**: Graceful degradation and fallbacks
- **Performance**: Response time and concurrent request handling
- **Security**: Input validation and sanitization

## 📁 Project Structure

```
RAG-TUTOR-CHATBOT/
├── fastapi_app.py          # Main FastAPI application with RAG
├── test_fast_app.py        # Comprehensive test suite
├── requirements.txt        # Python dependencies
├── build.sh               # Render build script
├── render.yaml            # Render configuration
├── README.md              # This file
├── RENDER_DEPLOY.md       # Render deployment guide
├── API_EXAMPLES.md        # API usage examples
├── CONTRIBUTING.md        # Contribution guidelines
├── CHANGELOG.md           # Version history
├── LICENSE                # MIT license
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
└── myenv/                # Virtual environment
```

## 🛠️ Development

### Adding New AI Providers

1. Create a new method in the `AIProvider` class:
   ```python
   @staticmethod
   async def call_new_provider(prompt: str, max_tokens: int = 1500):
       # Implementation here
       pass
   ```

2. Add the provider to the fallback chain in `chat_endpoint()`

3. Add configuration in `load_api_keys()`

### Extending RAG System

1. **Add New Documents**:
   ```python
   # In RAGProcessor._load_sample_documents()
   new_doc = {
       'id': 'unique_id',
       'title': 'Document Title',
       'content': 'Document content...',
       'source': 'source_name',
       'tags': ['tag1', 'tag2']
   }
   ```

2. **Custom Embedding Models**:
   ```python
   # In EmbeddingManager.__init__()
   self.model = SentenceTransformer("your-model-name")
   ```

3. **Vector Store Persistence**:
   ```python
   # Save vector store
   rag_processor.vector_store.save("vector_store.pkl")
   
   # Load vector store
   rag_processor.vector_store.load("vector_store.pkl")
   ```

## 🔒 Security Features

- **Input Sanitization**: HTML and script tag removal
- **Rate Limiting**: Per-client request limiting
- **CORS Protection**: Configurable cross-origin policies
- **API Key Security**: Secure handling and masking in logs
- **Validation**: Comprehensive input validation with Pydantic

## 🌐 Deployment

### Render (Recommended)
**Easy deployment to Render:**

1. Push your code to GitHub
2. Go to [Render.com](https://render.com) → New Web Service
3. Connect your repository and configure
4. Add your API keys as environment variables
5. Your app will be live at `https://your-app-name.onrender.com`

See [RENDER_DEPLOY.md](RENDER_DEPLOY.md) for detailed instructions.

### Local Development
```bash
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Performance & Monitoring

### Metrics Available
- **Request Count**: Total requests processed
- **Success Rate**: Percentage of successful responses
- **Response Time**: Average processing time
- **API Usage**: Usage statistics per provider
- **Cache Hit Rate**: Embedding and response cache efficiency
- **Error Rate**: Error frequency by type

### Performance Optimization
- **Embedding Caching**: Reduces computation overhead
- **Async Processing**: Non-blocking API calls
- **Vector Search**: Fast similarity search with FAISS
- **Connection Pooling**: Efficient HTTP client usage
- **Response Caching**: Redis-based response caching

### Monitoring
```bash
# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics

# Debug information
curl http://localhost:8000/debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **"No API key provided" errors**
   - Ensure your `.env` file contains valid API keys
   - Check that the `.env` file is in the root directory

2. **Import errors**
   - Activate your virtual environment
   - Install dependencies: `pip install -r requirements.txt`

3. **RAG system not working**
   - Check that sentence-transformers is installed
   - Verify FAISS installation: `pip install faiss-cpu`

4. **Slow response times**
   - Check API key validity
   - Monitor `/metrics` endpoint for performance data
   - Consider enabling Redis caching

### Debug Information

Use the `/debug` endpoint to test API connectivity:
```bash
curl http://localhost:8000/debug
```

## 📞 Support

For issues and questions:
- Check the [troubleshooting section](#-troubleshooting)
- Review test files for usage examples
- Open an issue on GitHub

## 🔄 Recent Updates

- **v2.0.0**: Complete RAG system implementation
  - Vector similarity search with FAISS
  - Sentence transformer embeddings
  - Context-aware responses
  - Production-ready features
  - Comprehensive testing suite
  - Performance monitoring
  - Security enhancements

---

Made with ❤️ for educational excellence
