#!/usr/bin/env python3
"""
RAG Tutor Chatbot Setup Script
Automates the initial setup process for the project.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {description} completed")
            return True
        else:
            print(f"   ❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e}")
        return False

def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def setup_virtual_environment():
    """Set up virtual environment"""
    venv_path = Path("myenv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    # Create virtual environment
    if run_command("python -m venv myenv", "Creating virtual environment"):
        print("📁 Virtual environment created at ./myenv")
        return True
    return False

def get_activation_command():
    """Get the appropriate activation command for the OS"""
    if os.name == 'nt':  # Windows
        return "myenv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        return "source myenv/bin/activate"

def install_dependencies():
    """Install project dependencies"""
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "myenv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "myenv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install main dependencies
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing main dependencies"):
        return False
    
    # Install development dependencies if requested
    dev_deps = input("📦 Install development dependencies? (y/n): ").lower().strip()
    if dev_deps in ['y', 'yes']:
        if Path("requirements-dev.txt").exists():
            run_command(f"{pip_cmd} install -r requirements-dev.txt", "Installing development dependencies")
    
    return True

def setup_environment_file():
    """Set up environment variables file"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("🔧 .env file already exists")
        return True
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("🔧 Created .env file from template")
        print("⚠️  Please edit .env file and add your API keys before running the application")
        return True
    else:
        # Create basic .env file
        env_content = """# RAG Tutor Chatbot Environment Variables
# Add your API keys here

# Required: At least one AI provider API key
OPENROUTER_API_KEY=your_openrouter_api_key_here
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Optional: For enhanced resource search
RAPIDAPI_KEY=your_rapidapi_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CX=your_google_custom_search_engine_id_here
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("🔧 Created basic .env file")
        print("⚠️  Please edit .env file and add your API keys before running the application")
        return True

def run_tests():
    """Run tests to verify setup"""
    test_choice = input("🧪 Run tests to verify setup? (y/n): ").lower().strip()
    if test_choice not in ['y', 'yes']:
        return True
    
    # Determine python command based on OS
    if os.name == 'nt':  # Windows
        python_cmd = "myenv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "myenv/bin/python"
    
    return run_command(f"{python_cmd} -m pytest test_fast_app.py -v", "Running tests", check=False)

def main():
    """Main setup function"""
    print("🎓 RAG Tutor Chatbot Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup virtual environment
    if not setup_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup environment file
    setup_environment_file()
    
    # Run tests
    run_tests()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit the .env file and add your API keys")
    print("2. Activate the virtual environment:")
    print(f"   {get_activation_command()}")
    print("3. Start the application:")
    print("   python fastapi_app.py")
    print("4. Visit http://localhost:8000 to test the API")
    print("\n📚 Check README.md for detailed documentation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
