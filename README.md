# 🎓 RAG Tutor Chatbot

An intelligent AI-powered tutoring service built with FastAPI that provides comprehensive educational responses with multimedia resources. The chatbot leverages multiple AI providers and offers contextual learning suggestions along with relevant video and website resources.

## ✨ Features

- **Multi-AI Provider Support**: Integrates with OpenRouter, Groq, and HuggingFace APIs with automatic fallback
- **Intelligent Response Adaptation**: Adjusts response length and complexity based on question type
- **Resource Integration**: Automatically finds relevant YouTube videos and educational websites
- **Dynamic Learning Suggestions**: Provides personalized study recommendations
- **Comprehensive Error Handling**: Robust fallback mechanisms ensure continuous service
- **RESTful API**: Clean API endpoints for easy integration
- **Health Monitoring**: Built-in health checks and debugging endpoints

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

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
   ```

5. **Run the application**
   ```bash
   python fastapi_app.py
   ```

The service will be available at `http://localhost:8000`

## 📡 API Endpoints

### Chat Endpoints

#### POST `/api/chat`
Send a question and receive an AI-generated response with resources.

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
  "hasContext": false,
  "processingTime": 2.34,
  "apiUsed": "OpenRouter",
  "suggestions": [
    "Create a study plan for biology topics",
    "Look for visual explanations of the process"
  ],
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

### Utility Endpoints

#### GET `/health`
Check service health and API key status.

#### GET `/debug`
Detailed debugging information and API testing.

#### GET `/`
Service information and available endpoints.

## 🧠 AI Provider Configuration

### OpenRouter
- **Models**: Uses `meta-llama/llama-3.1-8b-instruct:free`
- **Best for**: General tutoring questions
- **Setup**: Get API key from [OpenRouter](https://openrouter.ai/)

### Groq
- **Models**: Uses `llama3-8b-8192`
- **Best for**: Fast responses and coding questions
- **Setup**: Get API key from [Groq](https://groq.com/)

### HuggingFace
- **Models**: Uses `google/flan-t5-large`
- **Best for**: Fallback option for basic queries
- **Setup**: Get API key from [HuggingFace](https://huggingface.co/)

## 📚 Response Intelligence

The system automatically adapts responses based on question type:

- **Simple Questions** (e.g., "What is 2+2?"): Brief, direct answers
- **Factual Questions** (e.g., "What is photosynthesis?"): Concise definitions with examples
- **Complex Questions** (e.g., "Explain quantum mechanics"): Detailed explanations with context

## 🔧 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest test_fast_app.py -v

# Run specific test categories
pytest test_fast_app.py::TestAPIEndpoints -v
pytest test_fast_app.py::TestAIProvider -v
```

## 📁 Project Structure

```
RAG-TUTOR-CHATBOT/
├── fastapi_app.py          # Main FastAPI application
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

### Customizing Response Types

Modify the pattern matching in `chat_endpoint()` to add new question categories:

```python
custom_patterns = [
    r'^custom_pattern_here',
    # Add more patterns
]
```

## 🔒 Security Notes

- Never commit API keys to version control
- Use environment variables for all sensitive configuration
- The `.env` file is included in `.gitignore` for security
- Consider rate limiting for production deployments

## 🌐 Deployment

### Render (Recommended - Free Tier Available)
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
See `Dockerfile` for containerization setup.

## 📈 Performance Optimization

- Response time typically < 3 seconds
- Automatic caching for repeated queries (can be implemented)
- Concurrent request handling
- Graceful degradation when APIs are unavailable

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

3. **Port already in use**
   - Kill existing processes: `lsof -ti:8000 | xargs kill -9` (macOS/Linux)
   - Or use a different port: `uvicorn fastapi_app:app --port 8080`

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

- **v1.0.0**: Initial release with multi-provider AI support
- Enhanced response intelligence based on question complexity
- Comprehensive test suite with >90% coverage
- Robust error handling and fallback mechanisms

---

Made with ❤️ for educational excellence
