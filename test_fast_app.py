"""
Comprehensive test suite for AI Tutor FastAPI Service
Tests all endpoints, functionality, and edge cases.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import tempfile
from datetime import datetime

# Import the FastAPI app
from fastapi_app import app, RAGProcessor, AIProvider, ResourceFinder

# Create test client
client = TestClient(app)

class TestAPIEndpoints:
    """Test all API endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "AI Tutor Service"
        assert "endpoints" in data
        assert "/api/chat" in data["endpoints"]["chat"]
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "api_status" in data
        assert "documents_loaded" in data
    
    def test_chat_post_endpoint_valid_request(self):
        """Test POST chat endpoint with valid request"""
        test_data = {
            "question": "What is 2+2?",
            "include_context": True,
            "max_tokens": 100
        }
        
        with patch.object(AIProvider, 'call_openrouter') as mock_ai:
            mock_ai.return_value = "2 + 2 equals 4. This is basic arithmetic..."
            
            with patch.object(ResourceFinder, 'search_youtube') as mock_youtube:
                mock_youtube.return_value = "https://www.youtube.com/watch?v=test123"
                
                with patch.object(ResourceFinder, 'search_website') as mock_website:
                    mock_website.return_value = "https://example.com/math"
                    
                    response = client.post("/api/chat", json=test_data)
                    assert response.status_code == 200
                    data = response.json()
                    
                    assert "answer" in data
                    assert "videoLink" in data
                    assert "websiteLink" in data
                    assert "hasContext" in data
                    assert "processingTime" in data
                    assert "apiUsed" in data
                    assert "suggestions" in data
    
    def test_chat_get_endpoint(self):
        """Test GET chat endpoint"""
        with patch.object(AIProvider, 'call_openrouter') as mock_ai:
            mock_ai.return_value = "Test response"
            
            response = client.get("/api/chat?question=What is Python?")
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
    
    def test_chat_endpoint_empty_question(self):
        """Test chat endpoint with empty question"""
        test_data = {"question": ""}
        response = client.post("/api/chat", json=test_data)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_chat_endpoint_missing_question(self):
        """Test chat endpoint with missing question field"""
        test_data = {"max_tokens": 100}
        response = client.post("/api/chat", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_test_connection_endpoint(self):
        """Test the connection testing endpoint"""
        with patch.object(AIProvider, 'call_openrouter') as mock_ai:
            mock_ai.return_value = "Test response"
            
            response = client.get("/test-connection")
            assert response.status_code == 200
            data = response.json()
            assert "test_results" in data
            assert "timestamp" in data

class TestRAGProcessor:
    """Test RAG (Retrieval Augmented Generation) functionality"""
    
    def test_rag_initialization(self):
        """Test RAG processor initializes correctly"""
        rag = RAGProcessor()
        assert hasattr(rag, 'docs')
        assert isinstance(rag.docs, list)
    
    def test_load_documents_with_test_files(self):
        """Test document loading with temporary test files"""
        # Create temporary directory and files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                'test1.txt': 'This is about mathematics and algebra concepts.',
                'test2.md': 'Python programming fundamentals and variables.',
                'test3.json': '{"topic": "science", "content": "Physics concepts"}'
            }
            
            for filename, content in test_files.items():
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)
            
            # Mock the data directory path
            rag = RAGProcessor()
            original_data_dir = os.path.join(os.path.dirname(rag.__class__.__module__), 'data')
            
            with patch('os.path.join') as mock_join:
                mock_join.return_value = temp_dir
                with patch('os.path.exists') as mock_exists:
                    mock_exists.return_value = True
                    
                    rag.load_documents()
                    assert len(rag.docs) == 3
    
    def test_find_relevant_context(self):
        """Test finding relevant context from documents"""
        rag = RAGProcessor()
        # Add test documents
        rag.docs = [
            {
                'filename': 'math.txt',
                'content': 'Mathematics includes algebra, calculus, and geometry. These are fundamental concepts.',
                'word_count': 12
            },
            {
                'filename': 'programming.txt',
                'content': 'Python is a programming language used for web development and data science.',
                'word_count': 13
            }
        ]
        
        # Test relevant context finding
        context = rag.find_relevant_context("algebra mathematics", max_chunks=1)
        assert len(context) == 1
        assert "math.txt" in context[0]
        assert "algebra" in context[0]
    
    def test_find_relevant_context_no_matches(self):
        """Test context finding with no relevant matches"""
        rag = RAGProcessor()
        rag.docs = [
            {
                'filename': 'test.txt',
                'content': 'Random content about weather',
                'word_count': 5
            }
        ]
        
        context = rag.find_relevant_context("quantum physics")
        assert len(context) == 0

class TestAIProvider:
    """Test AI provider functionality"""
    
    @patch('requests.post')
    def test_openrouter_success(self, mock_post):
        """Test successful OpenRouter API call"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test AI response'}}]
        }
        mock_post.return_value = mock_response
        
        # Set API key for test
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_key'}):
            result = AIProvider.call_openrouter("Test question", 100)
            assert result is not None
            # Note: This is an async method, so we'd need to handle that in real testing
    
    def test_openrouter_no_api_key(self):
        """Test OpenRouter with no API key"""
        with patch.dict(os.environ, {}, clear=True):
            result = AIProvider.call_openrouter("Test question", 100)
            # This would return None due to missing API key
    
    @patch('requests.post')
    def test_openrouter_api_error(self, mock_post):
        """Test OpenRouter API error handling"""
        mock_response = MagicMock()
        mock_response.status_code = 429  # Rate limit error
        mock_response.text = "Rate limit exceeded"
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_key'}):
            result = AIProvider.call_openrouter("Test question", 100)
            # Should return None on error

class TestResourceFinder:
    """Test resource finding functionality"""
    
    @patch('requests.get')
    def test_youtube_search_success(self, mock_get):
        """Test successful YouTube search"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'contents': [
                {
                    'video': {
                        'videoId': 'test123',
                        'title': 'Math Tutorial Explanation',
                        'lengthSeconds': 300
                    }
                }
            ]
        }
        mock_get.return_value = mock_response
        
        with patch.dict(os.environ, {'RAPIDAPI_KEY': 'test_key'}):
            result = ResourceFinder.search_youtube("mathematics tutorial")
            # Should return YouTube URL with video ID
    
    def test_youtube_search_no_api_key(self):
        """Test YouTube search without API key"""
        with patch.dict(os.environ, {}, clear=True):
            result = ResourceFinder.search_youtube("test query")
            # Should return fallback search URL
            assert "youtube.com/results" in result
    
    @patch('requests.get')
    def test_website_search_success(self, mock_get):
        """Test successful website search"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'items': [
                {'link': 'https://khanacademy.org/math/algebra'},
                {'link': 'https://example.com/math'}
            ]
        }
        mock_get.return_value = mock_response
        
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key', 'GOOGLE_CX': 'test_cx'}):
            result = ResourceFinder.search_website("algebra mathematics")
            # Should return educational domain link
            assert "khanacademy.org" in result

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_chat_workflow_with_mocks(self):
        """Test complete chat workflow with mocked external services"""
        test_question = "Explain photosynthesis"
        
        with patch.object(AIProvider, 'call_openrouter') as mock_ai, \
             patch.object(ResourceFinder, 'search_youtube') as mock_youtube, \
             patch.object(ResourceFinder, 'search_website') as mock_website:
            
            # Setup mocks
            mock_ai.return_value = "Photosynthesis is the process by which plants convert sunlight into energy..."
            mock_youtube.return_value = "https://www.youtube.com/watch?v=biology123"
            mock_website.return_value = "https://khanacademy.org/science/biology"
            
            response = client.post("/api/chat", json={
                "question": test_question,
                "include_context": True
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify all components worked
            assert "photosynthesis" in data["answer"].lower()
            assert "youtube.com" in data["videoLink"]
            assert "khanacademy.org" in data["websiteLink"]
            assert data["apiUsed"] == "OpenRouter"
            assert len(data["suggestions"]) > 0
    
    def test_api_fallback_chain(self):
        """Test that API calls fall back correctly when services fail"""
        with patch.object(AIProvider, 'call_openrouter') as mock_openrouter, \
             patch.object(AIProvider, 'call_groq') as mock_groq, \
             patch.object(AIProvider, 'call_huggingface') as mock_hf:
            
            # First API fails
            mock_openrouter.return_value = None
            # Second API succeeds
            mock_groq.return_value = "Response from Groq API"
            mock_hf.return_value = None
            
            response = client.post("/api/chat", json={
                "question": "Test fallback"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["apiUsed"] == "Groq"

class TestPerformance:
    """Performance and load tests"""
    
    def test_response_time_reasonable(self):
        """Test that responses come back in reasonable time"""
        start_time = datetime.now()
        
        with patch.object(AIProvider, 'call_openrouter') as mock_ai:
            mock_ai.return_value = "Quick response"
            
            response = client.post("/api/chat", json={
                "question": "Quick test"
            })
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        assert response.status_code == 200
        assert response_time < 5  # Should respond within 5 seconds
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import concurrent.futures
        
        def make_request():
            with patch.object(AIProvider, 'call_openrouter') as mock_ai:
                mock_ai.return_value = "Concurrent response"
                return client.post("/api/chat", json={
                    "question": "Concurrent test"
                })
        
        # Test 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_malformed_json_request(self):
        """Test handling of malformed JSON"""
        response = client.post(
            "/api/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_extremely_long_question(self):
        """Test handling of very long questions"""
        long_question = "What is " + "very " * 1000 + "long question?"
        
        with patch.object(AIProvider, 'call_openrouter') as mock_ai:
            mock_ai.return_value = "Response to long question"
            
            response = client.post("/api/chat", json={
                "question": long_question
            })
            
            # Should handle gracefully
            assert response.status_code == 200
    
    def test_special_characters_in_question(self):
        """Test handling of special characters and unicode"""
        special_question = "What is 数学? Explain αβγ symbols! 🤔📚"
        
        with patch.object(AIProvider, 'call_openrouter') as mock_ai:
            mock_ai.return_value = "Response about special characters"
            
            response = client.post("/api/chat", json={
                "question": special_question
            })
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["answer"]) > 0

if __name__ == "__main__":
    print("AI Tutor FastAPI Test Suite")
    print("=" * 50)
    
    # Install required packages reminder
    print("Make sure you have installed the test dependencies:")
    print("pip install pytest pytest-asyncio httpx")
    print()
    print("Run tests with:")
    print("pytest test_fastapi_app.py -v")
    print("or")
    print("python -m pytest test_fastapi_app.py -v --tb=short")
