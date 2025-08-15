# RAG Tutor Chatbot API Examples

This document provides practical examples of how to use the RAG Tutor Chatbot API.

## 📡 Basic Usage Examples

### 1. Simple POST Request

```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is photosynthesis?",
       "include_context": true,
       "max_tokens": 500
     }'
```

### 2. GET Request (Quick Query)

```bash
curl "http://localhost:8000/api/chat?question=What%20is%202+2?"
```

### 3. Health Check

```bash
curl "http://localhost:8000/health"
```

## 🐍 Python Examples

### Basic Python Client

```python
import requests
import json

class TutorChatbot:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def ask_question(self, question, max_tokens=1500):
        """Ask a question to the tutor chatbot"""
        url = f"{self.base_url}/api/chat"
        data = {
            "question": question,
            "include_context": True,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, json=data)
        return response.json()
    
    def get_health(self):
        """Check service health"""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        return response.json()

# Usage example
if __name__ == "__main__":
    chatbot = TutorChatbot()
    
    # Ask a question
    result = chatbot.ask_question("Explain quantum mechanics")
    print(f"Answer: {result['answer']}")
    print(f"Video: {result['videoLink']}")
    print(f"Website: {result['websiteLink']}")
    print(f"Suggestions: {result['suggestions']}")
```

### Async Python Client

```python
import aiohttp
import asyncio

class AsyncTutorChatbot:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def ask_question(self, question, max_tokens=1500):
        """Async question asking"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/chat"
            data = {
                "question": question,
                "include_context": True,
                "max_tokens": max_tokens
            }
            
            async with session.post(url, json=data) as response:
                return await response.json()
    
    async def ask_multiple_questions(self, questions):
        """Ask multiple questions concurrently"""
        tasks = [self.ask_question(q) for q in questions]
        return await asyncio.gather(*tasks)

# Usage example
async def main():
    chatbot = AsyncTutorChatbot()
    
    questions = [
        "What is Python?",
        "Explain machine learning",
        "How does photosynthesis work?"
    ]
    
    results = await chatbot.ask_multiple_questions(questions)
    
    for i, result in enumerate(results):
        print(f"\nQuestion {i+1}: {questions[i]}")
        print(f"Answer: {result['answer'][:100]}...")
        print(f"Processing time: {result['processingTime']:.2f}s")

# Run async example
# asyncio.run(main())
```

## 🌐 JavaScript Examples

### Browser JavaScript (Fetch API)

```javascript
class TutorChatbot {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async askQuestion(question, maxTokens = 1500) {
        const url = `${this.baseUrl}/api/chat`;
        const data = {
            question: question,
            include_context: true,
            max_tokens: maxTokens
        };
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error asking question:', error);
            throw error;
        }
    }
    
    async getHealth() {
        const url = `${this.baseUrl}/health`;
        const response = await fetch(url);
        return await response.json();
    }
}

// Usage example
async function example() {
    const chatbot = new TutorChatbot();
    
    try {
        const result = await chatbot.askQuestion("What is artificial intelligence?");
        console.log('Answer:', result.answer);
        console.log('Video:', result.videoLink);
        console.log('Website:', result.websiteLink);
        console.log('Suggestions:', result.suggestions);
    } catch (error) {
        console.error('Failed to get response:', error);
    }
}
```

### Node.js Example

```javascript
const axios = require('axios');

class TutorChatbot {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }
    
    async askQuestion(question, maxTokens = 1500) {
        try {
            const response = await this.client.post('/api/chat', {
                question: question,
                include_context: true,
                max_tokens: maxTokens
            });
            
            return response.data;
        } catch (error) {
            if (error.response) {
                console.error('API Error:', error.response.data);
            } else {
                console.error('Network Error:', error.message);
            }
            throw error;
        }
    }
    
    async getHealth() {
        const response = await this.client.get('/health');
        return response.data;
    }
}

// Usage example
async function main() {
    const chatbot = new TutorChatbot();
    
    try {
        // Check health first
        const health = await chatbot.getHealth();
        console.log('Service status:', health.status);
        
        // Ask a question
        const result = await chatbot.askQuestion("Explain the water cycle");
        console.log('\n=== Tutor Response ===');
        console.log(`Answer: ${result.answer}`);
        console.log(`Video Resource: ${result.videoLink}`);
        console.log(`Website Resource: ${result.websiteLink}`);
        console.log(`Processing Time: ${result.processingTime}s`);
        console.log(`API Used: ${result.apiUsed}`);
        console.log('Learning Suggestions:');
        result.suggestions.forEach((suggestion, index) => {
            console.log(`  ${index + 1}. ${suggestion}`);
        });
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Run the example
main();
```

## 🚀 Advanced Usage Examples

### Batch Processing Questions

```python
import asyncio
import aiohttp
from typing import List, Dict

class BatchTutorProcessor:
    def __init__(self, base_url="http://localhost:8000", max_concurrent=5):
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_question(self, session, question):
        """Process a single question with rate limiting"""
        async with self.semaphore:
            url = f"{self.base_url}/api/chat"
            data = {"question": question, "max_tokens": 800}
            
            try:
                async with session.post(url, json=data) as response:
                    result = await response.json()
                    return {
                        "question": question,
                        "success": True,
                        "result": result
                    }
            except Exception as e:
                return {
                    "question": question,
                    "success": False,
                    "error": str(e)
                }
    
    async def process_batch(self, questions: List[str]):
        """Process multiple questions concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.process_question(session, question) 
                for question in questions
            ]
            return await asyncio.gather(*tasks)

# Example usage
async def batch_example():
    processor = BatchTutorProcessor()
    
    questions = [
        "What is the Pythagorean theorem?",
        "Explain Newton's first law",
        "How does DNA replication work?",
        "What is machine learning?",
        "Describe the water cycle"
    ]
    
    print("Processing questions in batch...")
    results = await processor.process_batch(questions)
    
    for result in results:
        if result["success"]:
            print(f"\n✓ {result['question']}")
            print(f"  Answer: {result['result']['answer'][:100]}...")
            print(f"  Time: {result['result']['processingTime']:.2f}s")
        else:
            print(f"\n✗ {result['question']}: {result['error']}")

# asyncio.run(batch_example())
```

### React Component Example

```jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const TutorChatbot = () => {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);
    const [health, setHealth] = useState(null);
    
    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    
    // Check service health on component mount
    useEffect(() => {
        const checkHealth = async () => {
            try {
                const result = await axios.get(`${apiUrl}/health`);
                setHealth(result.data);
            } catch (error) {
                console.error('Health check failed:', error);
            }
        };
        
        checkHealth();
    }, [apiUrl]);
    
    const askQuestion = async (e) => {
        e.preventDefault();
        if (!question.trim()) return;
        
        setLoading(true);
        setResponse(null);
        
        try {
            const result = await axios.post(`${apiUrl}/api/chat`, {
                question: question,
                include_context: true,
                max_tokens: 1000
            });
            
            setResponse(result.data);
        } catch (error) {
            console.error('Error asking question:', error);
            setResponse({
                error: 'Failed to get response. Please try again.'
            });
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="tutor-chatbot">
            <div className="status">
                Status: {health ? 
                    <span className="healthy">✓ Online</span> : 
                    <span className="unhealthy">✗ Offline</span>
                }
            </div>
            
            <form onSubmit={askQuestion} className="question-form">
                <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask your question here..."
                    rows={3}
                    disabled={loading}
                />
                <button type="submit" disabled={loading || !question.trim()}>
                    {loading ? 'Thinking...' : 'Ask Question'}
                </button>
            </form>
            
            {response && (
                <div className="response">
                    {response.error ? (
                        <div className="error">{response.error}</div>
                    ) : (
                        <>
                            <div className="answer">
                                <h3>Answer:</h3>
                                <p>{response.answer}</p>
                            </div>
                            
                            <div className="resources">
                                <h3>Learning Resources:</h3>
                                <div className="links">
                                    <a href={response.videoLink} target="_blank" rel="noopener noreferrer">
                                        📺 Video Tutorial
                                    </a>
                                    <a href={response.websiteLink} target="_blank" rel="noopener noreferrer">
                                        🌐 Learn More
                                    </a>
                                </div>
                            </div>
                            
                            <div className="suggestions">
                                <h3>Study Suggestions:</h3>
                                <ul>
                                    {response.suggestions.map((suggestion, index) => (
                                        <li key={index}>{suggestion}</li>
                                    ))}
                                </ul>
                            </div>
                            
                            <div className="metadata">
                                <small>
                                    Response time: {response.processingTime.toFixed(2)}s | 
                                    API: {response.apiUsed}
                                </small>
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
};

export default TutorChatbot;
```

## 📊 Response Format Examples

### Successful Response

```json
{
  "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
  "videoLink": "https://www.youtube.com/watch?v=example123",
  "websiteLink": "https://khanacademy.org/science/biology/photosynthesis",
  "hasContext": false,
  "processingTime": 2.34,
  "apiUsed": "OpenRouter",
  "suggestions": [
    "Create a study plan for biology topics",
    "Look for visual explanations of the process",
    "Practice explaining the concept to someone else"
  ],
  "debug_info": {
    "openrouter": {
      "attempted": true,
      "success": true,
      "status_code": 200,
      "response_length": 1205
    }
  }
}
```

### Error Response

```json
{
  "answer": "I apologize, but I encountered an error processing your question. Please try again.",
  "videoLink": "https://www.youtube.com/results?search_query=your%20topic%20tutorial",
  "websiteLink": "https://www.google.com/search?q=your%20topic",
  "hasContext": false,
  "processingTime": 0.45,
  "apiUsed": "error",
  "suggestions": [
    "Try rephrasing your question",
    "Check the suggested resources"
  ],
  "debug_info": {
    "error": "Connection timeout"
  }
}
```

## 🔧 Testing Examples

### API Testing Script

```python
import requests
import time
import json

def test_api_endpoints():
    """Test all main API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing RAG Tutor Chatbot API")
    print("=" * 40)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200
        print("   ✓ Health check passed")
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return
    
    # Test basic chat
    print("2. Testing basic chat...")
    try:
        data = {"question": "What is 2+2?", "max_tokens": 100}
        response = requests.post(f"{base_url}/api/chat", json=data, timeout=30)
        assert response.status_code == 200
        result = response.json()
        assert "answer" in result
        print(f"   ✓ Basic chat works (API: {result['apiUsed']})")
    except Exception as e:
        print(f"   ✗ Basic chat failed: {e}")
    
    # Test complex question
    print("3. Testing complex question...")
    try:
        data = {"question": "Explain quantum mechanics", "max_tokens": 500}
        start_time = time.time()
        response = requests.post(f"{base_url}/api/chat", json=data, timeout=30)
        end_time = time.time()
        
        assert response.status_code == 200
        result = response.json()
        response_time = end_time - start_time
        
        print(f"   ✓ Complex question works")
        print(f"     Response time: {response_time:.2f}s")
        print(f"     Answer length: {len(result['answer'])} chars")
        print(f"     Suggestions: {len(result['suggestions'])}")
        
    except Exception as e:
        print(f"   ✗ Complex question failed: {e}")
    
    # Test edge cases
    print("4. Testing edge cases...")
    
    # Empty question
    try:
        response = requests.post(f"{base_url}/api/chat", json={"question": ""})
        assert response.status_code == 400
        print("   ✓ Empty question properly rejected")
    except AssertionError:
        print("   ✗ Empty question should return 400")
    except Exception as e:
        print(f"   ✗ Empty question test error: {e}")
    
    print("\n🎉 API testing completed!")

if __name__ == "__main__":
    test_api_endpoints()
```

These examples should help you get started with integrating and using the RAG Tutor Chatbot API in various environments and programming languages!
