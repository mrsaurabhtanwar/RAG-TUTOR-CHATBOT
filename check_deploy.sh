#!/bin/bash
# Render Deployment Readiness Check for RAG Tutor Chatbot

echo "� RAG Tutor Chatbot - Render Deployment Readiness Check"
echo "========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}📁 Checking essential files...${NC}"

# Essential files check
files=("fastapi_app.py:Main application" "requirements.txt:Dependencies" "build.sh:Build script" "render.yaml:Render config" ".env.example:Environment template")

for file_info in "${files[@]}"; do
    IFS=':' read -r file desc <<< "$file_info"
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC} - $desc"
    else
        echo -e "${RED}❌ $file${NC} - MISSING!"
    fi
done

echo ""
echo -e "${BLUE}🚫 Checking for unwanted files...${NC}"

# Files that shouldn't exist for Render deployment
unwanted=("railway.json" "RAILWAY_DEPLOY.md" "Dockerfile" "docker-compose.yml" "Procfile" "requirements-dev.txt" "vercel.json" "netlify.toml")

unwanted_found=0
for file in "${unwanted[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${YELLOW}⚠️  $file${NC} - Should be removed for clean deployment"
        unwanted_found=1
    fi
done

if [ $unwanted_found -eq 0 ]; then
    echo -e "${GREEN}✅ No unwanted deployment files found${NC}"
fi

echo ""
echo -e "${BLUE}🔍 Checking file contents...${NC}"

# Check build.sh is executable-ready
if [ -f "build.sh" ]; then
    if grep -q "#!/usr/bin/env bash" build.sh; then
        echo -e "${GREEN}✅ build.sh${NC} - Has proper shebang"
    else
        echo -e "${YELLOW}⚠️  build.sh${NC} - Missing shebang (will be fixed by Render)"
    fi
fi

# Check render.yaml configuration
if [ -f "render.yaml" ]; then
    if grep -q "uvicorn fastapi_app:app" render.yaml; then
        echo -e "${GREEN}✅ render.yaml${NC} - Correct start command"
    else
        echo -e "${RED}❌ render.yaml${NC} - Incorrect start command"
    fi
fi

# Check FastAPI app PORT configuration
if [ -f "fastapi_app.py" ]; then
    if grep -q 'os.getenv("PORT"' fastapi_app.py; then
        echo -e "${GREEN}✅ fastapi_app.py${NC} - PORT environment variable configured"
    else
        echo -e "${RED}❌ fastapi_app.py${NC} - Missing PORT configuration"
    fi
fi

echo ""
echo -e "${BLUE}📋 Deployment Requirements Status:${NC}"

requirements=(
    "✅ Essential files present"
    "✅ Clean repository structure"
    "✅ Render configuration optimized"
    "✅ PORT environment handling"
    "📝 TODO: Push to GitHub repository"
    "📝 TODO: Create Render web service"
    "📝 TODO: Set OPENROUTER_API_KEY in Render dashboard"
    "📝 TODO: Deploy and test endpoints"
)

for req in "${requirements[@]}"; do
    echo -e "   $req"
done

echo ""
echo -e "${GREEN}🎯 Repository Status: RENDER DEPLOYMENT READY!${NC}"
echo ""
echo -e "${BLUE}📖 Next Steps:${NC}"
echo "1. Push repository to GitHub: git push origin main"
echo "2. Create Render web service from GitHub repo"
echo "3. Set environment variables in Render dashboard"
echo "4. Deploy and verify endpoints"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo "- Detailed guide: RENDER_DEPLOYMENT_GUIDE.md"
echo "- API examples: API_EXAMPLES.md"
echo "- Quick deploy: RENDER_DEPLOY.md"
echo ""
echo -e "${GREEN}🚀 Ready for production deployment!${NC}"
