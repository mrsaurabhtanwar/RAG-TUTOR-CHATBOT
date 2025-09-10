#!/bin/bash
# Startup script for Render deployment

echo "Starting RAG Tutor Chatbot..."

# Get the port from Render environment variable
PORT=${PORT:-8000}

echo "Using port: $PORT"

# Start the application with uvicorn
uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --workers 1
