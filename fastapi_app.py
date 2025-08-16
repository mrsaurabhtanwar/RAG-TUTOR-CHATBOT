"""
AI Tutor FastAPI Service
A comprehensive tutoring service that provides AI-generated responses with educational resources.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import quote
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Tutor Service",
    description="An intelligent tutoring service that provides comprehensive answers with video and website suggestions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str
    include_context: bool = True
    max_tokens: int = 1500


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


def load_api_keys() -> Dict[str, Optional[str]]:
    """Load and validate API keys from environment variables"""
    keys = {
        'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
        'RAPIDAPI_KEY': os.getenv('RAPIDAPI_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'GOOGLE_CX': os.getenv('GOOGLE_CX'),
        'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY')
    }
    
    print("=== API Keys Status ===")
    for key_name, key_value in keys.items():
        if key_value:
            # Remove quotes if they exist (common issue)
            clean_value = key_value.strip("'\"")
            keys[key_name] = clean_value
            masked_key = f"{clean_value[:10]}...{clean_value[-10:]}" if len(clean_value) > 20 else clean_value
            print(f"✓ {key_name}: {masked_key}")
        else:
            print(f"✗ {key_name}: Not found")
    print("=====================")
    
    return keys


# Load API keys globally
api_keys = load_api_keys()
OPENROUTER_API_KEY = api_keys.get('OPENROUTER_API_KEY')
RAPIDAPI_KEY = api_keys.get('RAPIDAPI_KEY')
GOOGLE_API_KEY = api_keys.get('GOOGLE_API_KEY')
GOOGLE_CX = api_keys.get('GOOGLE_CX')
HUGGINGFACE_API_KEY = api_keys.get('HUGGINGFACE_API_KEY')
GROQ_API_KEY = api_keys.get('GROQ_API_KEY')


class AIProvider:
    """Handles AI API interactions with different providers"""
    
    @staticmethod
    async def call_openrouter(prompt: str, max_tokens: int = 1500) -> Tuple[Optional[str], Dict[str, Any]]:
        """Call OpenRouter API with detailed debugging"""
        debug_info: Dict[str, Any] = {"attempted": True, "error": None}
        
        if not OPENROUTER_API_KEY:
            debug_info["error"] = "No API key provided"
            return None, debug_info
        
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "AI Tutor Service"
            }
            
            data: Dict[str, Any] = {
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert AI tutor. Adjust your response style based on the question complexity. For simple questions (like basic math), give direct, brief answers. For complex topics, provide comprehensive explanations with examples."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            print("Making request to OpenRouter...")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            debug_info["status_code"] = response.status_code
            debug_info["response_headers"] = dict(response.headers)
            
            print(f"OpenRouter Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    answer = result['choices'][0]['message']['content'].strip()
                    debug_info["success"] = True
                    debug_info["response_length"] = len(answer)
                    return answer, debug_info
                else:
                    debug_info["error"] = "No choices in response"
                    debug_info["response_body"] = result
            else:
                error_text = response.text
                debug_info["error"] = f"HTTP {response.status_code}: {error_text}"
                print(f"OpenRouter Error: {error_text}")
                
        except requests.RequestException as e:
            debug_info["error"] = f"Request Exception: {str(e)}"
            print(f"OpenRouter Exception: {e}")
        except Exception as e:
            debug_info["error"] = f"Exception: {str(e)}"
            print(f"OpenRouter Exception: {e}")
        
        return None, debug_info
    
    @staticmethod
    async def call_groq(prompt: str, max_tokens: int = 1500) -> Tuple[Optional[str], Dict[str, Any]]:
        """Call Groq API with detailed debugging"""
        debug_info: Dict[str, Any] = {"attempted": True, "error": None}
        
        if not GROQ_API_KEY:
            debug_info["error"] = "No API key provided"
            return None, debug_info
        
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data: Dict[str, Any] = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an intelligent AI tutor. Match your response length to the question complexity. Simple questions need brief, direct answers. Complex topics need detailed explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            print("Making request to Groq...")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            debug_info["status_code"] = response.status_code
            print(f"Groq Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    answer = result['choices'][0]['message']['content'].strip()
                    debug_info["success"] = True
                    debug_info["response_length"] = len(answer)
                    return answer, debug_info
                else:
                    debug_info["error"] = "No choices in response"
            else:
                debug_info["error"] = f"HTTP {response.status_code}: {response.text}"
                print(f"Groq Error: {response.text}")
                
        except requests.RequestException as e:
            debug_info["error"] = f"Request Exception: {str(e)}"
            print(f"Groq Exception: {e}")
        except Exception as e:
            debug_info["error"] = f"Exception: {str(e)}"
            print(f"Groq Exception: {e}")
        
        return None, debug_info
    
    @staticmethod
    async def call_huggingface(prompt: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Call Hugging Face API with detailed debugging"""
        debug_info: Dict[str, Any] = {"attempted": True, "error": None}
        
        if not HUGGINGFACE_API_KEY:
            debug_info["error"] = "No API key provided"
            return None, debug_info
        
        try:
            # Use a model that's good for Q&A
            url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
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
            
            print("Making request to HuggingFace...")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            debug_info["status_code"] = response.status_code
            print(f"HuggingFace Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result: Any = response.json()
                if isinstance(result, list) and len(result) > 0:  # type: ignore
                    first_item = result[0]  # type: ignore
                    if isinstance(first_item, dict) and 'generated_text' in first_item:
                        generated_text = str(first_item['generated_text'])  # type: ignore
                        answer: str = generated_text.strip()
                        if len(answer) > 20:
                            debug_info["success"] = True
                            debug_info["response_length"] = len(answer)
                            return answer, debug_info
                debug_info["error"] = "No valid response generated"
                debug_info["response_body"] = result
            else:
                debug_info["error"] = f"HTTP {response.status_code}: {response.text}"
                print(f"HuggingFace Error: {response.text}")
                
        except requests.RequestException as e:
            debug_info["error"] = f"Request Exception: {str(e)}"
            print(f"HuggingFace Exception: {e}")
        except Exception as e:
            debug_info["error"] = f"Exception: {str(e)}"
            print(f"HuggingFace Exception: {e}")
        
        return None, debug_info


class ResourceFinder:
    """Handles finding relevant videos and websites"""
    
    @staticmethod
    async def search_youtube(query: str) -> str:
        """Search YouTube with fallback to manual search"""
        if not RAPIDAPI_KEY:
            return f"https://www.youtube.com/results?search_query={quote(query + ' tutorial')}"

        try:
            url = 'https://youtube138.p.rapidapi.com/search/'
            params: Dict[str, str] = {'q': query + ' tutorial', 'hl': 'en', 'gl': 'US'}
            headers: Dict[str, str] = {
                'X-RapidAPI-Key': RAPIDAPI_KEY,
                'X-RapidAPI-Host': 'youtube138.p.rapidapi.com'
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'contents' in data and data['contents']:
                    for item in data['contents']:
                        video = item.get('video')
                        if video and video.get('videoId'):
                            return f"https://www.youtube.com/watch?v={video['videoId']}"
        
        except requests.RequestException as e:
            print(f"YouTube search error: {e}")
        except Exception as e:
            print(f"YouTube search error: {e}")

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
        
        except requests.RequestException as e:
            print(f"Google search error: {e}")
        except Exception as e:
            print(f"Google search error: {e}")

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


# Initialize providers
ai_provider = AIProvider()
resource_finder = ResourceFinder()



@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest) -> ChatResponse:
    """Enhanced chat endpoint with refined prompts for natural responses"""
    start_time = datetime.now()
    
    try:
        question = chat_request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"\n=== Processing Question ===")
        print(f"Question: {question}")
        
        # Analyze question complexity to determine response style
        simple_patterns = [
            r'^\d+\s*[\+\-\*\/]\s*\d+\s*$',                       # "2+2"
            r'^what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+',              # "what is 2+2"
            r'^(?:hi|hello|hey|hola|yo|sup)(?:[!.?]|\s|$)',       # greetings
            r'^(?:yes|no|yep|nah|y|n)(?:[!.?]|\s|$)',             # yes/no
            r'^(?:thanks?|thank\s+you|thx)(?:[!.?]|\s|$)',        # thanks
            r'^[\u263a-\U0001f645\s\t\.,!?:-]+$',                # emoji-only or punctuation
            r'^(?:ok|okay|kk|roger|got\s+it)(?:[!.?]|\s|$)',      # ack
            r'^[\w\.\-]+@[\w\.\-]+\.\w{2,}$',                    # email-like (short reply)
            r'^(?:what\'?s)?\s*(?:up|new)\s*[?!.]?$'              # "what's up"
        ]
        
        # Questions that need concise answers (factual/definitional)
        concise_patterns = [
            r'^(?:what|who|when|where)\s+is\s+[\w\s\?\-\,]{1,80}\?*$',      # "what is gravity?"
            r'^(?:define|definition\s+of)\s+.+',                          # "define photosynthesis"
            r'^what\s+is\s+(?:the\s+)?(?:formula|chemical\s+formula)\s+(?:of|for)\s+',  # "what is the formula of water"
            r'^(?:capital\s+of|currency\s+of)\s+[\w\s]+$',                 # "capital of France"
            r'^(?:convert|how\s+many)\s+\d+(\.\d+)?\s+\w+\s+(?:to|in)\s+\w+', # "convert 5 km to miles"
            r'^(?:who\s+is|who\s+was)\s+[\w\s]+$',                         # "who is Einstein"
            r'^(?:how\s+much|how\s+many)\s+.+',                            # numeric queries
            r'^[A-Za-z0-9\-\_]{1,20}\s*\?$',                              # short single-word Q "Python?"
            r'^[\w\s]{1,40}\s*\?$'                           # short question under ~40 chars
        ]

        complex_patterns = [
            r'^(?:explain|describe|tell\s+me\s+about)\s+.+',             # "explain photosynthesis"
            r'^(?:how|what|why)\s+do\s+[\w\s\?\-\,]{1,80}\?*$',          # "how do plants grow?"
            r'^(?:compare|contrast)\s+.+',                               # "compare apples and oranges"
            r'^(?:summarize|overview)\s+.+',                             # "summarize the main points"
            r'^(?:analyze|break down)\s+.+',                             # "analyze the data"
            r'^(?:what|who|when|where)\s+was\s+[\w\s\?\-\,]{1,80}\?*$',  # "what was the impact?"
            r'^(?:how|what|why)\s+does\s+[\w\s\?\-\,]{1,80}\?*$',        # "how does photosynthesis work?"
            r'\bexplain\b', r'\bwhy\b', r'\bhow\b',                          # often need explanation
            r'\bcompare\b', r'\bdifference\s+between\b',                    # comparisons
            r'(?:(?:step|steps)\s+to|step-by-step|walk\s+me\s+through)',    # step instructions
            r'\bproof\b|\bderive\b|\bshow\s+that\b',                        # math proof/derivation
            r'\bimplement\b|\bwrite\s+code\b|\bdebug\b|\brun\b\s+this\b',   # coding requests
            r'\boptimize\b|\bimprove\b|\brecommend\b',                      # design/optimisation
            r'\bplan\b|\bproposal\b|\barchitecture\b|\bdesign\b',          # open-ended
            r'\bpros\s+and\s+cons\b|\bcase\s+study\b|\buse\s+case\b'        # long-form
        ]

        is_simple_question = any(
            __import__('re').match(pattern, question.lower().strip()) 
            for pattern in simple_patterns
        ) or len(question.split()) <= 3
        
        is_concise_question = any(
            __import__('re').match(pattern, question.lower().strip()) 
            for pattern in concise_patterns
        )
        
        is_complex_question = any(
            __import__('re').match(pattern, question.lower().strip())
            for pattern in complex_patterns
        )

        # Create refined prompts that encourage natural, appropriate responses
        if is_simple_question:
            prompt = f"""Answer this simple question naturally and briefly within 1 sentence:

{question}

Keep it conversational and direct - just give the answer without unnecessary elaboration."""

        elif is_concise_question:
            prompt = f"""Answer this question clearly and concisely with 1 to 2 sentences:

{question}

Provide the key information in a natural, conversational way. Be direct but complete - include the essential details without over-explaining."""

        elif is_complex_question:
            prompt = f"""Provide a thorough answer to this question 3-4 sentences long:

{question}

Give a comprehensive response that covers the important aspects. Use a natural, educational tone and include examples where helpful."""

        else:
            # Default balanced approach
            prompt = f"""Answer this question in a helpful, natural way:

{question}

Provide appropriate detail - enough to be informative but not overwhelming. Be conversational and direct."""
        
        # Adjust max_tokens based on question type
        if is_simple_question:
            adjusted_max_tokens = min(400, chat_request.max_tokens)  # Very short responses
        elif is_concise_question:
            adjusted_max_tokens = min(700, chat_request.max_tokens)  # Concise responses
        else:
            adjusted_max_tokens = chat_request.max_tokens  # Full responses
        
        # Try AI providers with detailed debugging
        debug_info: Dict[str, Any] = {}
        answer: Optional[str] = None
        api_used = "none"
        
        # Try OpenRouter
        if OPENROUTER_API_KEY:
            print("Trying OpenRouter API...")
            answer, openrouter_debug = await ai_provider.call_openrouter(prompt, adjusted_max_tokens)
            debug_info["openrouter"] = openrouter_debug
            if answer:
                api_used = "OpenRouter"
                print("✓ OpenRouter succeeded!")
        
        # Try Groq if OpenRouter failed
        if not answer and GROQ_API_KEY:
            print("Trying Groq API...")
            answer, groq_debug = await ai_provider.call_groq(prompt, adjusted_max_tokens)
            debug_info["groq"] = groq_debug
            if answer:
                api_used = "Groq"
                print("✓ Groq succeeded!")
        
        # Try HuggingFace if others failed
        if not answer and HUGGINGFACE_API_KEY:
            print("Trying HuggingFace API...")
            answer, hf_debug = await ai_provider.call_huggingface(prompt)
            debug_info["huggingface"] = hf_debug
            if answer:
                api_used = "HuggingFace"
                print("✓ HuggingFace succeeded!")
        
        # Use comprehensive fallback if all APIs failed
        if not answer:
            print("All APIs failed, using comprehensive fallback")
            api_used = "comprehensive_fallback"
            answer = generate_comprehensive_fallback(question)
        
        # Get resources
        video_link = await resource_finder.search_youtube(question)
        website_link = await resource_finder.search_website(question)
        suggestions = generate_suggestions(question)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = ChatResponse(
            answer=answer,
            videoLink=video_link,
            websiteLink=website_link,
            hasContext=False,
            processingTime=processing_time,
            apiUsed=api_used,
            suggestions=suggestions,
            debug_info=debug_info
        )
        
        print(f"✓ Response completed in {processing_time:.2f}s using {api_used}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
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
    return await chat_endpoint(chat_request)


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
    
    return {"debug_results": results, "timestamp": datetime.now().isoformat()}


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check with API status"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'api_keys': {
            'openrouter': bool(OPENROUTER_API_KEY),
            'groq': bool(GROQ_API_KEY),
            'huggingface': bool(HUGGINGFACE_API_KEY),
            'google': bool(GOOGLE_API_KEY and GOOGLE_CX),
            'rapidapi': bool(RAPIDAPI_KEY)
        }
    }


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with service information"""
    return {
        'service': 'AI Tutor Service',
        'status': 'running',
        'endpoints': {
            'chat': '/api/chat',
            'debug': '/debug',
            'health': '/health'
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Tutor Service with Enhanced Debugging...")
    
    # Use Railway's PORT environment variable or default to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")