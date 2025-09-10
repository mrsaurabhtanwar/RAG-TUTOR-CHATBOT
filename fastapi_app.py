"""
AI Tutor FastAPI Service - Production-Ready RAG System
A comprehensive tutoring service that provides AI-generated responses with educational resources.
"""

# type: ignore

import os
import sys
import logging
import asyncio
import time
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union, TYPE_CHECKING
from urllib.parse import quote
from functools import lru_cache
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import pickle
import threading
from collections import defaultdict, deque
import uuid

# Load environment variables from .env file
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Optional ML dependencies - fallback to basic mode if not available
ML_AVAILABLE = False
np: Any = None
SentenceTransformer: Any = None
faiss: Any = None
if TYPE_CHECKING:
    # Help type-checkers without requiring runtime packages
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    globals()['ML_AVAILABLE'] = True
    logger.info("ML dependencies loaded successfully")
except ImportError as e:
    logger.warning(f"ML dependencies not available: {e}. Running in basic mode.")

# Initialize FastAPI app
app = FastAPI(
    title="AI Tutor Service - RAG System",
    description="An intelligent tutoring service with RAG capabilities that provides comprehensive answers with video and website suggestions",
    version="2.0.0"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add CORS middleware with more restrictive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        # map client_id -> deque of request timestamps
        # Using plain runtime collections to avoid heavy typing in method scope
        self.requests = defaultdict(deque)  # type: ignore
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        with self.lock:
            # Clean old requests
            while self.requests[client_id] and self.requests[client_id][0] < now - self.window_seconds:
                self.requests[client_id].popleft()

            # Check if under limit
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(now)
                return True
            return False


rate_limiter = RateLimiter()

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    include_context: bool = Field(True, description="Whether to include RAG context")
    max_tokens: int = Field(1500, ge=100, le=4000, description="Maximum tokens for response")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        # Basic sanitization
        v = re.sub(r'<script.*?</script>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'<.*?>', '', v)
        if len(v.strip()) == 0:
            raise ValueError('Question cannot be empty after sanitization')
        return v.strip()

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str
    videoLink: Optional[str] = None
    websiteLink: Optional[str] = None
    hasContext: bool
    processingTime: float
    apiUsed: str
    suggestions: List[str] = []
    debug_info: Dict[str, Any] = {}
    context_sources: List[str] = []
    confidence_score: Optional[float] = None


# Models for quiz-generation payload from platform
class StudentBehavior(BaseModel):
    hint_count: int = 0
    bottom_hint: bool = False
    attempt_count: int = 0
    ms_first_response: int = 0
    duration: int = 0
    confidence_frustrated: float = 0.0
    confidence_confused: float = 0.0
    confidence_concentrating: float = 0.0
    confidence_bored: float = 0.0
    action_count: int = 0
    hint_dependency: float = 0.0
    response_speed: Optional[str] = None
    confidence_balance: float = 0.0
    engagement_ratio: float = 0.0
    efficiency_indicator: float = 0.0
    predicted_score: float = 0.0
    performance_category: Optional[str] = None
    learner_profile: Optional[str] = None


class QuizRequest(BaseModel):
    """Payload for quiz generation coming from the platform"""
    context_refs: List[str] = []
    topics: List[str] = Field(..., description="List of topic strings")
    difficulty: Optional[str] = Field("medium")
    type: Optional[str] = Field("mcq")
    n_questions: int = Field(10, ge=1, le=100)
    include_explanations: bool = True
    include_resources: bool = True
    student_behavior: Optional[StudentBehavior] = None

    @field_validator('topics')
    @classmethod
    def check_topics(cls, v: List[str]) -> List[str]:
        if not v or len(v) < 1:
            raise ValueError('topics must contain at least one topic')
        return v



class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""
    total_requests: int
    success_rate: float
    average_response_time: float
    api_usage: Dict[str, int]
    cache_hit_rate: float
    error_rate: float
    timestamp: str

# Global metrics
class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0

        # runtime collections
        self.response_times = deque(maxlen=1000)
        self.api_usage = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = defaultdict(int)
        self.lock = threading.Lock()

    def record_request(self, success: bool, response_time: float, api_used: str, cache_hit: bool = False):
        with self.lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            self.response_times.append(response_time)
            self.api_usage[api_used] += 1
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def record_error(self, error_type: str):
        with self.lock:
            self.errors[error_type] += 1

    def get_metrics(self) -> MetricsResponse:
        with self.lock:
            success_rate = self.successful_requests / max(self.total_requests, 1)
            avg_response_time = sum(self.response_times) / max(len(self.response_times), 1)
            cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            error_rate = sum(self.errors.values()) / max(self.total_requests, 1)

            return MetricsResponse(
                total_requests=self.total_requests,
                success_rate=success_rate,
                average_response_time=avg_response_time,
                api_usage=dict(self.api_usage),
                cache_hit_rate=cache_hit_rate,
                error_rate=error_rate,
                timestamp=datetime.now().isoformat()
            )


metrics = MetricsCollector()


@app.post('/generate_quiz')
async def generate_quiz_endpoint(payload: QuizRequest):
    """Accept quiz-generation payload from external platform and return generated questions.

    This endpoint expects the platform to supply student behavior and topic/context references.
    We merge the payload with local defaults and return a placeholder quiz structure. Replace
    the placeholder logic with calls to your question-generation AI or RAG pipeline as needed.
    """
    # Merge incoming payload with defaults
    default_question_count = 10
    n_questions = payload.n_questions or default_question_count

    # Build a simple placeholder quiz
    questions = []
    for i in range(n_questions):
        q = {
            'id': str(uuid.uuid4()),
            'type': payload.type or 'mcq',
            'topic': payload.topics[i % len(payload.topics)],
            'difficulty': payload.difficulty,
            'question_text': f"Placeholder question {i+1} on {payload.topics[i % len(payload.topics)]}",
            'choices': ["A", "B", "C", "D"],
            'answer': "A",
            'explanation': "This is a placeholder explanation." if payload.include_explanations else None
        }
        if payload.include_resources:
            q['resources'] = payload.context_refs or []
        questions.append(q)

    questions: List[Dict[str, Any]] = []
    for i in range(n_questions):
        q: Dict[str, Any] = {
            'id': str(uuid.uuid4()),
            'type': payload.type or 'mcq',
            'topic': payload.topics[i % len(payload.topics)],
            'difficulty': payload.difficulty,
            'question_text': f"Placeholder question {i+1} on {payload.topics[i % len(payload.topics)]}",
            'choices': ["A", "B", "C", "D"],
            'answer': "A",
            'explanation': "This is a placeholder explanation." if payload.include_explanations else None
        }
        if payload.include_resources:
            q['resources'] = payload.context_refs or []
        questions.append(q)

    result: Dict[str, Any] = {
        'quiz_id': str(uuid.uuid4()),
        'n_questions': len(questions),
        'questions': questions,
        'student_behavior_accepted': payload.student_behavior.model_dump() if payload.student_behavior else None,
        'generated_by': 'RAG-Tutor-Chatbot (placeholder)'
    }

    return JSONResponse(status_code=200, content=result)

def load_api_keys() -> Dict[str, Optional[str]]:
    """Load and validate API keys from environment variables or direct assignment"""
    keys = {
        'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'GOOGLE_CX': os.getenv('GOOGLE_CX'),
        'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY'),
        'GROQ_API_KEY': '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'  # Direct assignment as requested
    }
    
    logger.info("=== API Keys Status ===")
    for key_name, key_value in keys.items():
        if key_value:
            # Remove quotes if they exist (common issue)
            clean_value = key_value.strip("'\"")
            keys[key_name] = clean_value
            masked_key = f"{clean_value[:10]}...{clean_value[-10:]}" if len(clean_value) > 20 else clean_value
            logger.info(f"OK {key_name}: {masked_key}")
        else:
            logger.warning(f"X {key_name}: Not found")
    logger.info("=====================")
    
    return keys

# Load API keys globally
api_keys = load_api_keys()
OPENROUTER_API_KEY = api_keys.get('OPENROUTER_API_KEY')
GOOGLE_API_KEY = api_keys.get('GOOGLE_API_KEY')
GOOGLE_CX = api_keys.get('GOOGLE_CX')
HUGGINGFACE_API_KEY = api_keys.get('HUGGINGFACE_API_KEY')
GROQ_API_KEY = api_keys.get('GROQ_API_KEY')

# RAG Components
class EmbeddingManager:
    """Manages embeddings for RAG functionality"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        if ML_AVAILABLE:
            self._load_model()
        else:
            logger.warning("ML dependencies not available. RAG functionality disabled.")
    
    def _load_model(self):
        """Load the embedding model"""
        if not ML_AVAILABLE:
            return
            
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a simpler approach
            self.model = None
    
    def get_embedding(self, text: str) -> Optional[Any]:
        """Get embedding for text"""
        if not ML_AVAILABLE or not self.model:
            return None
        
        try:
            # Clean and truncate text
            text = text.strip()[:1000]  # Limit length
            if not text:
                return None
            
            embedding = self.model.encode([text])[0]
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

class VectorStore:
    """Simple in-memory vector store using FAISS"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.document_embeddings = []
        self.lock = threading.Lock()
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        if not ML_AVAILABLE:
            logger.warning("FAISS not available - vector store disabled")
            return
            
        try:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            logger.info("FAISS index initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = None
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[Any]):
        """Add documents and their embeddings to the store"""
        if not ML_AVAILABLE or not self.index:
            logger.warning("FAISS index not available")
            return
        
        with self.lock:
            try:
                # Add embeddings to FAISS
                embeddings_array = np.array(embeddings).astype('float32')
                self.index.add(embeddings_array)
                
                # Store documents
                self.documents.extend(documents)
                self.document_embeddings.extend(embeddings)
                
                logger.info(f"Added {len(documents)} documents to vector store")
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
    
    def search(self, query_embedding: Optional[Any], k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        if not self.index or self.index.ntotal == 0:
            return []
        
        try:
            # Search in FAISS
            query_array = query_embedding.reshape(1, -1).astype('float32')
            scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            # Return documents with scores
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        if not self.index:
            return
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'index': faiss.serialize_index(self.index),
                    'documents': self.documents,
                    'embedding_dim': self.embedding_dim
                }, f)
            logger.info(f"Vector store saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.index = faiss.deserialize_index(data['index'])
            self.documents = data['documents']
            self.embedding_dim = data['embedding_dim']
            logger.info(f"Vector store loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")

class RAGProcessor:
    """Handles RAG (Retrieval Augmented Generation) functionality"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(self.embedding_manager.embedding_dim)
        self.cache = {}
        self.cache_lock = threading.Lock()
        self._load_sample_documents()
    
    def _load_sample_documents(self):
        """Load sample educational documents for demonstration"""
        sample_docs = [
            {
                'id': 'math_basics',
                'title': 'Mathematics Fundamentals',
                'content': 'Mathematics is the study of numbers, quantities, shapes, and patterns. Basic operations include addition, subtraction, multiplication, and division. Algebra introduces variables and equations.',
                'source': 'educational_content',
                'tags': ['mathematics', 'algebra', 'basics']
            },
            {
                'id': 'python_intro',
                'title': 'Python Programming Introduction',
                'content': 'Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.',
                'source': 'educational_content',
                'tags': ['programming', 'python', 'computer_science']
            },
            {
                'id': 'physics_concepts',
                'title': 'Physics Core Concepts',
                'content': 'Physics is the natural science that studies matter, energy, and their interactions. Key concepts include force, motion, energy, and gravity. Newton\'s laws of motion are fundamental principles.',
                'source': 'educational_content',
                'tags': ['physics', 'science', 'motion', 'energy']
            },
            {
                'id': 'chemistry_basics',
                'title': 'Chemistry Fundamentals',
                'content': 'Chemistry is the study of matter and its properties. Atoms are the basic building blocks of matter. Chemical reactions involve the rearrangement of atoms to form new substances.',
                'source': 'educational_content',
                'tags': ['chemistry', 'science', 'atoms', 'reactions']
            },
            {
                'id': 'biology_cells',
                'title': 'Biology: Cell Structure',
                'content': 'Cells are the basic units of life. All living organisms are composed of cells. Plant cells have cell walls and chloroplasts, while animal cells do not. The nucleus contains genetic material.',
                'source': 'educational_content',
                'tags': ['biology', 'science', 'cells', 'life']
            }
        ]
        
        # Generate embeddings and add to vector store
        embeddings = []
        for doc in sample_docs:
            embedding = self.embedding_manager.get_embedding(doc['content'])
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # Fallback: create a simple embedding
                embeddings.append(np.random.rand(self.embedding_manager.embedding_dim))
        
        self.vector_store.add_documents(sample_docs, embeddings)
        logger.info(f"Loaded {len(sample_docs)} sample documents into RAG system")
    
    def get_relevant_context(self, query: str, max_chunks: int = 3) -> List[Dict[str, Any]]:
        """Get relevant context for a query using vector similarity"""
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        with self.cache_lock:
            if cache_key in self.cache:
                logger.info("Cache hit for context retrieval")
                return self.cache[cache_key]
        
        # Generate query embedding
        query_embedding = self.embedding_manager.get_embedding(query)
        if query_embedding is None:
            logger.warning("Could not generate embedding for query")
            return []
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=max_chunks)
        
        # Format results
        context_chunks = []
        for doc, score in results:
            if score > 0.3:  # Similarity threshold
                context_chunks.append({
                    'content': doc['content'],
                    'title': doc['title'],
                    'source': doc['source'],
                    'relevance_score': score,
                    'tags': doc.get('tags', [])
                })
        
        # Cache results
        with self.cache_lock:
            self.cache[cache_key] = context_chunks
        
        logger.info(f"Retrieved {len(context_chunks)} relevant context chunks")
        return context_chunks

class RetryManager:
    """Manages retries with exponential backoff and jitter"""
    
    @staticmethod
    async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
        """Retry function with exponential backoff and jitter"""
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Final attempt failed: {e}")
                    raise e
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + np.random.uniform(0, 0.1)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)

class AIProvider:
    """Handles AI API interactions with different providers"""
    
    
    @staticmethod
    async def call_openrouter(prompt: str, max_tokens: int = 1500) -> Tuple[Optional[str], Dict[str, Any]]:
        """Call OpenRouter API with retry logic"""
        debug_info: Dict[str, Any] = {"attempted": True, "error": None}
        
        if not OPENROUTER_API_KEY:
            debug_info["error"] = "No API key provided"
            return None, debug_info
        
        async def _make_request():
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "AI Tutor Service"
            }
            
            data: Dict[str, Any] = {
                "model": "meta-llama/llama-3.1-8b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert AI tutor. Provide accurate, helpful responses based on the context provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    answer = result['choices'][0]['message']['content'].strip()
                    debug_info["success"] = True
                    debug_info["response_length"] = len(answer)
                    return answer
                else:
                    raise Exception("No choices in response")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        try:
            answer = await RetryManager.retry_with_backoff(_make_request)
            return answer, debug_info
        except Exception as e:
            debug_info["error"] = str(e)
            logger.error(f"OpenRouter API error: {e}")
            return None, debug_info
    
    @staticmethod
    async def call_groq(prompt: str, max_tokens: int = 1500) -> Tuple[Optional[str], Dict[str, Any]]:
        """Call Groq API with retry logic"""
        debug_info: Dict[str, Any] = {"attempted": True, "error": None}
        
        if not GROQ_API_KEY:
            debug_info["error"] = "No API key provided"
            return None, debug_info
        
        async def _make_request():
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data: Dict[str, Any] = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an intelligent AI tutor. Provide accurate, helpful responses."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    answer = result['choices'][0]['message']['content'].strip()
                    debug_info["success"] = True
                    debug_info["response_length"] = len(answer)
                    return answer
                else:
                    raise Exception("No choices in response")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        try:
            answer = await RetryManager.retry_with_backoff(_make_request)
            return answer, debug_info
        except Exception as e:
            debug_info["error"] = str(e)
            logger.error(f"Groq API error: {e}")
            return None, debug_info
    
    @staticmethod
    async def call_huggingface(prompt: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Call Hugging Face API with retry logic"""
        debug_info: Dict[str, Any] = {"attempted": True, "error": None}
        
        if not HUGGINGFACE_API_KEY:
            debug_info["error"] = "No API key provided"
            return None, debug_info
        
        async def _make_request():
            url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            headers = {
                "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data: Dict[str, Any] = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 500,
                    "temperature": 0.7,
                    "do_sample": True
                },
                "options": {"wait_for_model": True}
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result: Any = response.json()
                if isinstance(result, list) and len(result) > 0:
                    first_item = result[0]
                    if isinstance(first_item, dict) and 'generated_text' in first_item:
                        generated_text = str(first_item['generated_text'])
                        answer: str = generated_text.strip()
                        if len(answer) > 20:
                            debug_info["success"] = True
                            debug_info["response_length"] = len(answer)
                            return answer
                raise Exception("No valid response generated")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        try:
            answer = await RetryManager.retry_with_backoff(_make_request)
            return answer, debug_info
        except Exception as e:
            debug_info["error"] = str(e)
            logger.error(f"HuggingFace API error: {e}")
            return None, debug_info

class ResourceFinder:
    """Handles finding relevant videos and websites"""
    
    @staticmethod
    async def search_youtube(query: str) -> str:
        """Search YouTube with manual search"""
        return f"https://www.youtube.com/results?search_query={quote(query + ' tutorial')}"
    
    @staticmethod
    async def search_website(query: str) -> str:
        """Search websites with fallback to Google search"""
        if not GOOGLE_API_KEY or not GOOGLE_CX:
            return f"https://www.google.com/search?q={quote(query)}"

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params: Dict[str, str] = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query, "num": "3"}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and data['items']:
                    return data['items'][0].get('link', f"https://www.google.com/search?q={quote(query)}")
        
        except Exception as e:
            logger.error(f"Google search error: {e}")

        return f"https://www.google.com/search?q={quote(query)}"

def generate_comprehensive_fallback(question: str) -> str:
    """Generate a simple fallback response when APIs fail"""
    return f"""I'm currently experiencing some technical difficulties with my AI models, but I'd be happy to help you learn about your question: "{question}"

Here are some ways you can explore this topic:
• Check out the educational video and website resources provided below
• Break down your question into smaller, specific parts
• Look for reliable educational sources and tutorials
• Consider discussing this topic with teachers, peers, or online communities

The resources below should provide you with detailed explanations from educational experts. Feel free to ask me again later, or try rephrasing your question in a different way!"""

def generate_suggestions(question: str) -> List[str]:
    """Generate dynamic learning suggestions based on question content"""
    question_lower = question.lower()
    suggestions: List[str] = []
    
    # Extract key topics from the question
    question_words = set(question_lower.split())
    
    # Dynamic suggestions based on question content
    if any(word in question_words for word in ['learn', 'study', 'understand', 'master']):
        suggestions.append(f"Create a study plan to break down {question.split()[-3:][0] if len(question.split()) > 3 else 'this topic'} into manageable parts")
    
    if any(word in question_words for word in ['how', 'why', 'what', 'explain']):
        suggestions.append("Look for visual explanations and diagrams to better understand the concept")
    
    if any(word in question_words for word in ['practice', 'exercise', 'problem', 'solve']):
        suggestions.append("Find practice exercises and work through them step by step")
    
    if any(word in question_words for word in ['example', 'application', 'use', 'real']):
        suggestions.append("Explore real-world applications and case studies related to this topic")
    
    # Subject-specific suggestions
    if any(word in question_words for word in ['math', 'mathematics', 'equation', 'formula', 'calculate']):
        suggestions.append("Practice similar problems and verify your solutions")
    elif any(word in question_words for word in ['science', 'experiment', 'theory', 'hypothesis']):
        suggestions.append("Try conducting simple experiments or simulations")
    elif any(word in question_words for word in ['history', 'historical', 'past', 'ancient']):
        suggestions.append("Create a timeline to better understand the sequence of events")
    elif any(word in question_words for word in ['language', 'grammar', 'vocabulary', 'writing']):
        suggestions.append("Practice through reading and writing exercises")
    elif any(word in question_words for word in ['programming', 'code', 'software', 'algorithm']):
        suggestions.append("Build a small project to apply these programming concepts")
    
    # Generic helpful suggestions if none of the above matched
    if not suggestions:
        suggestions = [
            "Join online communities or forums to discuss this topic with others",
            "Find additional resources like tutorials or courses on this subject",
            "Practice explaining the concept to someone else to test your understanding"
        ]
    
    # Ensure we don't exceed 3 suggestions and they're all unique
    unique_suggestions = list(dict.fromkeys(suggestions))  # Remove duplicates while preserving order
    return unique_suggestions[:3]

# Initialize components
ai_provider = AIProvider()
resource_finder = ResourceFinder()

# Initialize RAG processor only if ML dependencies are available
if ML_AVAILABLE:
    try:
        rag_processor = RAGProcessor()
        logger.info("RAG processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG processor: {e}")
        rag_processor = None
else:
    rag_processor = None
    logger.info("RAG functionality disabled - running in basic mode")

# Dependency for rate limiting
async def check_rate_limit(request: Request):
    try:
        client_id = request.client.host if request.client else "unknown"
    except:
        client_id = "unknown"
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest, request: Request) -> ChatResponse:
    """Enhanced chat endpoint with RAG capabilities"""
    start_time = time.time()
    
    # Check rate limit
    await check_rate_limit(request)
    
    try:
        question = chat_request.question
        logger.info(f"Processing question: {question[:100]}...")
        
        # Get relevant context using RAG
        context_chunks = []
        context_sources = []
        if chat_request.include_context and rag_processor:
            try:
                context_chunks = rag_processor.get_relevant_context(question)
                context_sources = [chunk['title'] for chunk in context_chunks]
            except Exception as e:
                logger.error(f"RAG context retrieval failed: {e}")
                context_chunks = []
                context_sources = []
        
        # Build enhanced prompt with context
        if context_chunks:
            context_text = "\n\n".join([chunk['content'] for chunk in context_chunks])
            enhanced_prompt = f"""Based on the following educational context, please answer the question:

Context:
{context_text}

Question: {question}

Please provide a comprehensive answer that incorporates the relevant information from the context."""
        else:
            enhanced_prompt = f"Please answer this question: {question}"
        
        # Try AI providers with fallback
        debug_info: Dict[str, Any] = {}
        answer: Optional[str] = None
        api_used = "none"
        confidence_score = None
        
        # Try Groq first (highest priority - most reliable)
        if GROQ_API_KEY:
            logger.info("Trying Groq API...")
            answer, groq_debug = await ai_provider.call_groq(enhanced_prompt, chat_request.max_tokens)
            debug_info["groq"] = groq_debug
            if answer:
                api_used = "Groq"
                confidence_score = 0.95
                logger.info("Groq succeeded!")
        
        # Try OpenRouter if Groq failed
        if not answer and OPENROUTER_API_KEY:
            logger.info("Trying OpenRouter API...")
            answer, openrouter_debug = await ai_provider.call_openrouter(enhanced_prompt, chat_request.max_tokens)
            debug_info["openrouter"] = openrouter_debug
            if answer:
                api_used = "OpenRouter"
                confidence_score = 0.9
                logger.info("OpenRouter succeeded!")
        
        # Try HuggingFace if others failed
        if not answer and HUGGINGFACE_API_KEY:
            logger.info("Trying HuggingFace API...")
            answer, hf_debug = await ai_provider.call_huggingface(enhanced_prompt)
            debug_info["huggingface"] = hf_debug
            if answer:
                api_used = "HuggingFace"
                confidence_score = 0.8
                logger.info("HuggingFace succeeded!")
        
        # Use comprehensive fallback if all APIs failed
        if not answer:
            logger.warning("All APIs failed, using comprehensive fallback")
            api_used = "comprehensive_fallback"
            answer = generate_comprehensive_fallback(question)
            confidence_score = 0.3
        
        # Get resources
        video_link = await resource_finder.search_youtube(question)
        website_link = await resource_finder.search_website(question)
        suggestions = generate_suggestions(question)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        success = answer is not None and len(answer) > 0
        metrics.record_request(success, processing_time, api_used, cache_hit=len(context_chunks) > 0)
        
        response = ChatResponse(
            answer=answer,
            videoLink=video_link,
            websiteLink=website_link,
            hasContext=len(context_chunks) > 0,
            processingTime=processing_time,
            apiUsed=api_used,
            suggestions=suggestions,
            debug_info=debug_info,
            context_sources=context_sources,
            confidence_score=confidence_score
        )
        
        logger.info(f"Response completed in {processing_time:.2f}s using {api_used}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        metrics.record_error("chat_endpoint_error")
        processing_time = time.time() - start_time
        
        return ChatResponse(
            answer="I apologize, but I encountered an error processing your question. Please try again.",
            videoLink=await resource_finder.search_youtube(chat_request.question),
            websiteLink=await resource_finder.search_website(chat_request.question),
            hasContext=False,
            processingTime=processing_time,
            apiUsed="error",
            suggestions=["Try rephrasing your question", "Check the suggested resources"],
            debug_info={"error": str(e)}
        )

@app.get("/api/chat")
async def chat_get(question: str = Query(..., description="The question to ask")) -> ChatResponse:
    """GET endpoint for chat"""
    chat_request = ChatRequest(question=question)
    return await chat_endpoint(chat_request, Request)

@app.get("/debug")
async def debug_endpoint() -> Dict[str, Any]:
    """Debug endpoint to check API status"""
    test_question = "What is 2+2?"
    results: Dict[str, Any] = {}
    
    
    # Test OpenRouter
    if OPENROUTER_API_KEY:
        answer, debug_info = await ai_provider.call_openrouter(test_question, 100)
        results["openrouter"] = {
            "configured": True,
            "working": bool(answer),
            "debug": debug_info,
            "sample_response": answer[:100] if answer else None
        }
    else:
        results["openrouter"] = {"configured": False}
    
    # Test Groq
    if GROQ_API_KEY:
        answer, debug_info = await ai_provider.call_groq(test_question, 100)
        results["groq"] = {
            "configured": True,
            "working": bool(answer),
            "debug": debug_info,
            "sample_response": answer[:100] if answer else None
        }
    else:
        results["groq"] = {"configured": False}
    
    # Test HuggingFace
    if HUGGINGFACE_API_KEY:
        answer, debug_info = await ai_provider.call_huggingface(test_question)
        results["huggingface"] = {
            "configured": True,
            "working": bool(answer),
            "debug": debug_info,
            "sample_response": answer[:100] if answer else None
        }
    else:
        results["huggingface"] = {"configured": False}
    
    # Test RAG system
    if rag_processor:
        try:
            context_chunks = rag_processor.get_relevant_context(test_question)
            results["rag_system"] = {
                "configured": True,
                "working": len(context_chunks) > 0,
                "context_chunks_found": len(context_chunks),
                "sample_context": context_chunks[0]['content'][:100] if context_chunks else None
            }
        except Exception as e:
            results["rag_system"] = {
                "configured": True,
                "working": False,
                "error": str(e)
            }
    else:
        results["rag_system"] = {"configured": False, "reason": "ML dependencies not available"}
    
    return {"debug_results": results, "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check with comprehensive status"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'api_keys': {
            'groq': bool(GROQ_API_KEY),
            'openrouter': bool(OPENROUTER_API_KEY),
            'huggingface': bool(HUGGINGFACE_API_KEY),
            'google': bool(GOOGLE_API_KEY and GOOGLE_CX)
        },
        'rag_system': {
            'embedding_model': rag_processor.embedding_manager.model_name if rag_processor else "disabled",
            'vector_store_documents': len(rag_processor.vector_store.documents) if rag_processor else 0,
            'cache_size': len(rag_processor.cache) if rag_processor else 0,
            'ml_available': ML_AVAILABLE
        },
        'system': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'uptime': 'running'
        }
    }

@app.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint() -> MetricsResponse:
    """Get system metrics"""
    return metrics.get_metrics()

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with service information"""
    return {
        'service': 'AI Tutor Service - RAG System',
        'status': 'running',
        'version': '2.0.0',
        'endpoints': {
            'chat': '/api/chat',
            'debug': '/debug',
            'health': '/health',
            'metrics': '/metrics'
        },
        'features': [
            'RAG (Retrieval-Augmented Generation)',
            'Multi-AI Provider Support (Groq, OpenRouter, HuggingFace)',
            'Vector Similarity Search',
            'Caching and Performance Optimization',
            'Rate Limiting and Security',
            'Structured Logging and Metrics'
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Tutor Service with RAG capabilities...")
    
    # Use Render's PORT environment variable or default to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")