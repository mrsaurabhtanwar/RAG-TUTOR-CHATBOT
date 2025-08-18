#!/usr/bin/env python3
"""
Load Testing Script for RAG Tutor Chatbot
Tests system performance under various load conditions.
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict, Any
import json
import argparse
from datetime import datetime

class LoadTester:
    """Load testing utility for the RAG Tutor API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
        
    async def make_request(self, session: aiohttp.ClientSession, question: str) -> Dict[str, Any]:
        """Make a single API request"""
        start_time = time.time()
        
        try:
            payload = {
                "question": question,
                "include_context": True,
                "max_tokens": 500
            }
            
            async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "status_code": response.status,
                        "api_used": data.get("apiUsed", "unknown"),
                        "has_context": data.get("hasContext", False),
                        "processing_time": data.get("processingTime", 0)
                    }
                else:
                    return {
                        "success": False,
                        "response_time": response_time,
                        "status_code": response.status,
                        "error": await response.text()
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            }
    
    async def run_concurrent_requests(self, questions: List[str], concurrency: int = 10) -> List[Dict[str, Any]]:
        """Run multiple requests concurrently"""
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_request(question: str):
                async with semaphore:
                    return await self.make_request(session, question)
            
            tasks = [bounded_request(q) for q in questions]
            results = await asyncio.gather(*tasks)
            return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results"""
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if not successful_requests:
            return {
                "total_requests": len(results),
                "success_rate": 0.0,
                "error": "No successful requests"
            }
        
        response_times = [r["response_time"] for r in successful_requests]
        processing_times = [r.get("processing_time", 0) for r in successful_requests]
        
        api_usage = {}
        for r in successful_requests:
            api = r.get("api_used", "unknown")
            api_usage[api] = api_usage.get(api, 0) + 1
        
        context_usage = sum(1 for r in successful_requests if r.get("has_context", False))
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(results),
            "response_time_stats": {
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
            },
            "processing_time_stats": {
                "mean": statistics.mean(processing_times),
                "median": statistics.median(processing_times),
                "min": min(processing_times),
                "max": max(processing_times)
            },
            "api_usage": api_usage,
            "context_usage": {
                "with_context": context_usage,
                "without_context": len(successful_requests) - context_usage,
                "context_rate": context_usage / len(successful_requests)
            },
            "errors": [r.get("error", "Unknown error") for r in failed_requests]
        }
    
    async def run_load_test(self, num_requests: int = 100, concurrency: int = 10, 
                          test_type: str = "mixed") -> Dict[str, Any]:
        """Run a comprehensive load test"""
        print(f"🚀 Starting load test: {num_requests} requests, {concurrency} concurrent")
        print(f"Test type: {test_type}")
        print(f"Target URL: {self.base_url}")
        print("-" * 50)
        
        # Generate test questions based on type
        questions = self.generate_test_questions(num_requests, test_type)
        
        # Run the test
        start_time = time.time()
        results = await self.run_concurrent_requests(questions, concurrency)
        end_time = time.time()
        
        # Analyze results
        analysis = self.analyze_results(results)
        analysis["test_duration"] = end_time - start_time
        analysis["requests_per_second"] = num_requests / (end_time - start_time)
        analysis["test_type"] = test_type
        analysis["concurrency"] = concurrency
        analysis["timestamp"] = datetime.now().isoformat()
        
        return analysis
    
    def generate_test_questions(self, count: int, test_type: str) -> List[str]:
        """Generate test questions based on type"""
        questions = []
        
        if test_type == "simple":
            base_questions = [
                "What is 2+2?",
                "What is the capital of France?",
                "How many planets are in our solar system?",
                "What is the largest ocean?",
                "Who wrote Romeo and Juliet?"
            ]
        elif test_type == "complex":
            base_questions = [
                "Explain the process of photosynthesis in detail",
                "How does quantum computing work?",
                "What are the implications of climate change?",
                "Explain the theory of relativity",
                "How do neural networks function?"
            ]
        elif test_type == "math":
            base_questions = [
                "What is the derivative of x^2?",
                "How do you solve quadratic equations?",
                "What is the Pythagorean theorem?",
                "Explain calculus concepts",
                "How do you find the area of a circle?"
            ]
        elif test_type == "science":
            base_questions = [
                "How do cells divide?",
                "What is the structure of DNA?",
                "How do chemical reactions work?",
                "Explain the water cycle",
                "What causes earthquakes?"
            ]
        else:  # mixed
            base_questions = [
                "What is 2+2?",
                "Explain photosynthesis",
                "What is the capital of Japan?",
                "How do computers work?",
                "What is gravity?",
                "Who was Albert Einstein?",
                "How do plants grow?",
                "What is democracy?",
                "Explain the water cycle",
                "What is the speed of light?"
            ]
        
        # Repeat questions to reach the desired count
        for i in range(count):
            questions.append(base_questions[i % len(base_questions)])
        
        return questions

def print_results(analysis: Dict[str, Any]):
    """Print test results in a formatted way"""
    print("\n📊 LOAD TEST RESULTS")
    print("=" * 50)
    print(f"Total Requests: {analysis['total_requests']}")
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"Successful: {analysis['successful_requests']}")
    print(f"Failed: {analysis['failed_requests']}")
    print(f"Success Rate: {analysis['success_rate']:.2%}")
    print(f"Test Duration: {analysis['test_duration']:.2f}s")
    print(f"Requests/Second: {analysis['requests_per_second']:.2f}")
    
    if 'response_time_stats' in analysis:
        print(f"\n⏱️ Response Time Statistics:")
        rt_stats = analysis['response_time_stats']
        print(f"  Mean: {rt_stats['mean']:.3f}s")
        print(f"  Median: {rt_stats['median']:.3f}s")
        print(f"  Min: {rt_stats['min']:.3f}s")
        print(f"  Max: {rt_stats['max']:.3f}s")
        print(f"  Std Dev: {rt_stats['std_dev']:.3f}s")
    
    if 'processing_time_stats' in analysis:
        print(f"\n🔧 Processing Time Statistics:")
        pt_stats = analysis['processing_time_stats']
        print(f"  Mean: {pt_stats['mean']:.3f}s")
        print(f"  Median: {pt_stats['median']:.3f}s")
        print(f"  Min: {pt_stats['min']:.3f}s")
        print(f"  Max: {pt_stats['max']:.3f}s")
    
    if 'api_usage' in analysis and analysis['api_usage']:
        print(f"\n🤖 API Usage:")
        for api, count in analysis['api_usage'].items():
            percentage = (count / analysis['successful_requests']) * 100
            print(f"  {api}: {count} ({percentage:.1f}%)")
    
    if 'context_usage' in analysis:
        print(f"\n📚 Context Usage:")
        context_stats = analysis['context_usage']
        print(f"  With Context: {context_stats['with_context']} ({context_stats['context_rate']:.1%})")
        print(f"  Without Context: {context_stats['without_context']}")
    
    if analysis.get('errors'):
        print(f"\n❌ Errors:")
        for error in set(analysis['errors']):
            count = analysis['errors'].count(error)
            print(f"  {error}: {count} times")

async def main():
    """Main function to run load tests"""
    parser = argparse.ArgumentParser(description="Load test the RAG Tutor API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--type", choices=["simple", "complex", "math", "science", "mixed"], 
                       default="mixed", help="Type of test questions")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create load tester
    tester = LoadTester(args.url)
    
    # Run the test
    results = await tester.run_load_test(
        num_requests=args.requests,
        concurrency=args.concurrency,
        test_type=args.type
    )
    
    # Print results
    print_results(results)
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
