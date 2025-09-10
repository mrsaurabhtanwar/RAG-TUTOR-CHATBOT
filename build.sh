#!/bin/bash
# Build script for Render deployment

echo "Starting build process..."

# Upgrade pip
pip install --upgrade pip

# Install minimal requirements optimized for Render free tier
echo "Installing minimal requirements for Render free tier (512MB limit)"
pip install -r requirements.txt

# Set environment variable to ensure app knows it's free tier
export RENDER_FREE_TIER=true

echo "Build completed successfully!"
