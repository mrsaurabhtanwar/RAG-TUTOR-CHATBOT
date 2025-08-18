#!/usr/bin/env #!/usr/bin/env bash
# Build script for Render deployment - Conservative approach

set -e  # Exit on error

echo "🚀 Starting Render deployment build..."

# Check Python version
echo "🐍 Python version:"
python --version

# Upgrade pip and build tools first
echo "📦 Upgrading pip and build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install core dependencies first (most stable approach)
echo "📚 Installing core FastAPI dependencies..."
python -m pip install --no-cache-dir fastapi==0.115.0
python -m pip install --no-cache-dir uvicorn[standard]==0.30.6
python -m pip install --no-cache-dir pydantic==2.9.2

# Install remaining dependencies
echo "📦 Installing remaining dependencies..."
python -m pip install --no-cache-dir -r requirements.txt

# Create necessary directories
echo "📁 Creating application directories..."
mkdir -p logs
mkdir -p data
mkdir -p cache

# Set proper permissions
echo "🔐 Setting file permissions..."
chmod +x fastapi_app.py

# Verify critical imports
echo "✅ Verifying critical imports..."
python -c "
try:
    import fastapi
    import uvicorn
    import pydantic
    import openai
    print(f'✓ FastAPI version: {fastapi.__version__}')
    print(f'✓ Uvicorn version: {uvicorn.__version__}')
    print(f'✓ Pydantic version: {pydantic.__version__}')
    print(f'✓ OpenAI version: {openai.__version__}')
    print('✓ All critical imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

# Test app import (basic functionality)
echo "🧪 Testing app import..."
python -c "
try:
    from fastapi_app import app
    print('✓ FastAPI app import successful (basic mode)')
except Exception as e:
    print(f'✗ App import error: {e}')
    exit(1)
"

echo "🎉 Build completed successfully (basic mode - ML features will be disabled)!"
echo "ℹ️  Note: RAG functionality disabled due to missing ML dependencies."
echo "ℹ️  The app will run with core AI chat functionality only."
# Build script for Render deployment - Python 3.11 Compatible

set -e  # Exit on error

echo "🚀 Starting Render deployment build (Python 3.11)..."

# Check Python version
echo "🐍 Python version:"
python --version

# Upgrade pip and build tools first
echo "📦 Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Install build dependencies explicitly
echo "� Installing build dependencies..."
pip install --no-cache-dir setuptools-scm build

# Install core dependencies first
echo "📚 Installing core dependencies..."
pip install --no-cache-dir fastapi==0.115.0 uvicorn[standard]==0.30.6 pydantic==2.9.2

# Install remaining dependencies
echo "📦 Installing remaining dependencies..."
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
echo "📁 Creating application directories..."
mkdir -p logs
mkdir -p data
mkdir -p cache

# Set proper permissions
echo "🔐 Setting file permissions..."
chmod +x fastapi_app.py

# Verify critical imports
echo "✅ Verifying critical imports..."
python -c "
try:
    import fastapi
    import uvicorn
    import pydantic
    import openai
    print(f'✓ FastAPI version: {fastapi.__version__}')
    print(f'✓ Uvicorn version: {uvicorn.__version__}')
    print(f'✓ Pydantic version: {pydantic.__version__}')
    print(f'✓ OpenAI version: {openai.__version__}')
    print('✓ All critical imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

# Test app import
echo "🧪 Testing app import..."
python -c "
try:
    from fastapi_app import app
    print('✓ FastAPI app import successful')
except Exception as e:
    print(f'✗ App import error: {e}')
    exit(1)
"

echo "🎉 Build completed successfully!"
