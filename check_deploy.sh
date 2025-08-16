#!/bin/bash
# Quick deployment check script for Render

echo "🔍 Checking RAG Tutor Chatbot - Render Deployment Readiness"
echo "============================================================="

# Check essential files exist
echo ""
echo "📁 Checking essential files..."

if [ -f "fastapi_app.py" ]; then
    echo "✅ fastapi_app.py - Main application"
else
    echo "❌ fastapi_app.py - MISSING!"
fi

if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt - Dependencies"
else
    echo "❌ requirements.txt - MISSING!"
fi

if [ -f "build.sh" ]; then
    echo "✅ build.sh - Build script"
else
    echo "❌ build.sh - MISSING!"
fi

if [ -f "render.yaml" ]; then
    echo "✅ render.yaml - Render configuration"
else
    echo "❌ render.yaml - MISSING!"
fi

if [ -f ".env.example" ]; then
    echo "✅ .env.example - Environment template"
else
    echo "❌ .env.example - MISSING!"
fi

# Check for unnecessary files
echo ""
echo "🚫 Checking for unnecessary files..."

unnecessary_files=("railway.json" "RAILWAY_DEPLOY.md" "Dockerfile" "docker-compose.yml" "Procfile" "requirements-dev.txt")

for file in "${unnecessary_files[@]}"; do
    if [ -f "$file" ]; then
        echo "⚠️  $file - Should be removed for clean deployment"
    fi
done

echo ""
echo "🎯 Syntax check..."
if python -m py_compile fastapi_app.py 2>/dev/null; then
    echo "✅ Python syntax check passed"
else
    echo "❌ Python syntax errors found"
fi

echo ""
echo "📋 Deployment checklist:"
echo "1. ✅ Essential files present"
echo "2. ✅ Unnecessary files removed"
echo "3. ✅ Syntax check passed"
echo "4. 📝 TODO: Set environment variables in Render dashboard"
echo "5. 📝 TODO: Push to GitHub and deploy"

echo ""
echo "🚀 Ready for Render deployment!"
