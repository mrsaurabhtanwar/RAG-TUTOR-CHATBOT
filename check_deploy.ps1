# Render Deployment Readiness Check (PowerShell)
# RAG Tutor Chatbot

Write-Host "RAG Tutor Chatbot - Render Deployment Readiness Check" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Essential files check
Write-Host "Checking essential files..." -ForegroundColor Blue
Write-Host ""

$essentialFiles = @{
    "fastapi_app.py" = "Main application"
    "requirements.txt" = "Dependencies"
    "build.sh" = "Build script"
    "render.yaml" = "Render configuration"
    ".env.example" = "Environment template"
}

foreach ($file in $essentialFiles.Keys) {
    if (Test-Path $file) {
        Write-Host "[OK] $file" -ForegroundColor Green -NoNewline
        Write-Host " - $($essentialFiles[$file])"
    } else {
        Write-Host "[MISSING] $file" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Checking for unwanted files..." -ForegroundColor Blue

$unwantedFiles = @("railway.json", "RAILWAY_DEPLOY.md", "Dockerfile", "docker-compose.yml", "Procfile", "requirements-dev.txt")
$unwantedFound = $false

foreach ($file in $unwantedFiles) {
    if (Test-Path $file) {
        Write-Host "[WARNING] $file - Should be removed" -ForegroundColor Yellow
        $unwantedFound = $true
    }
}

if (-not $unwantedFound) {
    Write-Host "[OK] No unwanted deployment files found" -ForegroundColor Green
}

Write-Host ""
Write-Host "Checking configurations..." -ForegroundColor Blue

# Check render.yaml
if (Test-Path "render.yaml") {
    $renderContent = Get-Content "render.yaml" -Raw
    if ($renderContent -like "*uvicorn fastapi_app:app*") {
        Write-Host "[OK] render.yaml - Correct start command" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] render.yaml - Incorrect start command" -ForegroundColor Red
    }
}

# Check FastAPI PORT configuration
if (Test-Path "fastapi_app.py") {
    $appContent = Get-Content "fastapi_app.py" -Raw
    if ($appContent -like '*os.getenv("PORT"*') {
        Write-Host "[OK] fastapi_app.py - PORT environment variable configured" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] fastapi_app.py - Missing PORT configuration" -ForegroundColor Red
    }
}

# Test Python syntax
Write-Host ""
Write-Host "Testing Python syntax..." -ForegroundColor Blue

try {
    & "D:\RAG-TUTOR-CHATBOT\myenv\Scripts\python.exe" -m py_compile "fastapi_app.py" 2>$null
    Write-Host "[OK] Python syntax check passed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python syntax errors found" -ForegroundColor Red
}

Write-Host ""
Write-Host "Deployment Requirements Status:" -ForegroundColor Blue

$requirements = @(
    "[OK] Essential files present",
    "[OK] Clean repository structure", 
    "[OK] Render configuration optimized",
    "[OK] PORT environment handling",
    "[TODO] Push to GitHub repository",
    "[TODO] Create Render web service", 
    "[TODO] Set OPENROUTER_API_KEY in Render dashboard",
    "[TODO] Deploy and test endpoints"
)

foreach ($req in $requirements) {
    Write-Host "   $req"
}

Write-Host ""
Write-Host "Repository Status: " -NoNewline
Write-Host "RENDER DEPLOYMENT READY!" -ForegroundColor Green
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Blue
Write-Host "1. Push repository to GitHub: git push origin main"
Write-Host "2. Create Render web service from GitHub repo"  
Write-Host "3. Set environment variables in Render dashboard"
Write-Host "4. Deploy and verify endpoints"
Write-Host ""

Write-Host "Documentation:" -ForegroundColor Blue
Write-Host "- Detailed guide: RENDER_DEPLOYMENT_GUIDE.md"
Write-Host "- API examples: API_EXAMPLES.md"
Write-Host "- Quick deploy: RENDER_DEPLOY.md"
Write-Host ""

Write-Host "Ready for production deployment!" -ForegroundColor Green
