#!/bin/bash
# Build script for Render deployment

echo "Starting build process..."

# Upgrade pip
pip install --upgrade pip

# Check if we're on Render free tier
if [ "$RENDER" = "true" ] && [ "$MEMORY_LIMIT" = "512" ]; then
    echo "Detected Render free tier - using lightweight requirements"
    pip install -r requirements-free.txt
else
    echo "Using full requirements with ML dependencies"
    pip install -r requirements.txt
fi

echo "Build completed successfully!"
