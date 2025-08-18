#!/usr/bin/env bash
# Build script for Render deployment

set -e  # Exit on error

echo "🚀 Starting Render deployment build..."

# Upgrade pip and install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating application directories..."
mkdir -p logs
mkdir -p data

# Set proper permissions
echo "🔐 Setting file permissions..."
chmod +x fastapi_app.py

# Verify installation
echo "✅ Verifying installation..."
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
python -c "import uvicorn; print(f'Uvicorn version: {uvicorn.__version__}')"

echo "🎉 Build completed successfully!"
