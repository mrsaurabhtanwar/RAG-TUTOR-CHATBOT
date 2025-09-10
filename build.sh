#!/bin/bash
# Build script for Render deployment

echo "Starting build process..."

# Upgrade pip
pip install --upgrade pip

# Check if we're on Render free tier (more aggressive detection)
if [ "$RENDER" = "true" ] || [ "$MEMORY_LIMIT" = "512" ] || [ "$RENDER_FREE_TIER" = "true" ]; then
    echo "Detected Render free tier - using minimal requirements"
    pip install -r requirements-minimal.txt
    # Set environment variable to ensure app knows it's free tier
    export RENDER_FREE_TIER=true
else
    echo "Using full requirements with ML dependencies"
    pip install -r requirements.txt
fi

echo "Build completed successfully!"
