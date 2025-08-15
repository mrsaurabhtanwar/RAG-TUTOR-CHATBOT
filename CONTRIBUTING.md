# Contributing to RAG Tutor Chatbot

Thank you for your interest in contributing to the RAG Tutor Chatbot! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Code Style](#code-style)
8. [Documentation](#documentation)

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful, inclusive, and constructive in all interactions.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of FastAPI and async programming
- Familiarity with AI APIs (OpenAI, HuggingFace, etc.)

### Areas for Contribution

We welcome contributions in the following areas:

- **AI Provider Integration**: Adding support for new AI services
- **Feature Development**: New tutoring features and capabilities
- **Performance Optimization**: Improving response times and efficiency
- **Documentation**: Improving guides, examples, and API documentation
- **Testing**: Adding test cases and improving test coverage
- **Bug Fixes**: Resolving issues and edge cases
- **UI/Frontend**: Creating web interfaces for the API

## 🛠️ Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/RAG-TUTOR-CHATBOT.git
   cd RAG-TUTOR-CHATBOT
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run tests**
   ```bash
   pytest test_fast_app.py -v
   ```

6. **Start development server**
   ```bash
   python fastapi_app.py
   ```

## 📝 Making Changes

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### Development Workflow

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Follow the existing code style
   - Add appropriate error handling
   - Include type hints where applicable

3. **Write or update tests**
   ```bash
   pytest test_fast_app.py::TestYourFeature -v
   ```

4. **Test your changes**
   ```bash
   # Run all tests
   pytest test_fast_app.py -v
   
   # Check code style
   black fastapi_app.py
   isort fastapi_app.py
   
   # Type checking
   mypy fastapi_app.py
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest test_fast_app.py -v

# Run specific test categories
pytest test_fast_app.py::TestAPIEndpoints -v
pytest test_fast_app.py::TestAIProvider -v

# Run with coverage
pytest test_fast_app.py --cov=fastapi_app --cov-report=html
```

### Writing Tests

When adding new features, please include:

- **Unit tests** for individual functions
- **Integration tests** for API endpoints
- **Edge case tests** for error handling
- **Performance tests** if applicable

Example test structure:
```python
class TestNewFeature:
    """Test new feature functionality"""
    
    def test_feature_success(self):
        """Test successful feature execution"""
        # Arrange
        test_data = {...}
        
        # Act
        result = your_function(test_data)
        
        # Assert
        assert result.status == "success"
        assert "expected_value" in result.data
    
    def test_feature_edge_case(self):
        """Test edge case handling"""
        # Test implementation
        pass
```

## 📤 Submitting Changes

### Pull Request Process

1. **Ensure your code passes all tests**
   ```bash
   pytest test_fast_app.py -v
   black --check fastapi_app.py
   isort --check-only fastapi_app.py
   ```

2. **Update documentation**
   - Update README.md if you've changed functionality
   - Add docstrings to new functions
   - Update API documentation

3. **Create a Pull Request**
   - Use a clear, descriptive title
   - Provide a detailed description of changes
   - Reference any related issues
   - Include screenshots for UI changes

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Documentation updated
```

## 🎨 Code Style

### Python Code Style

We follow PEP 8 with some project-specific conventions:

```python
# Use type hints
def process_question(question: str, max_tokens: int = 1500) -> Dict[str, Any]:
    """Process a user question with proper typing."""
    pass

# Use descriptive variable names
user_question = request.question
ai_response = await provider.generate_response(user_question)

# Add comprehensive docstrings
class AIProvider:
    """Handles AI API interactions with multiple providers.
    
    This class manages different AI service providers and implements
    fallback mechanisms for robust response generation.
    
    Attributes:
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for failed requests
    """
```

### Code Formatting Tools

```bash
# Format code
black fastapi_app.py
isort fastapi_app.py

# Check formatting
black --check fastapi_app.py
isort --check-only fastapi_app.py

# Type checking
mypy fastapi_app.py
```

## 📚 Documentation

### Documentation Guidelines

- **API Documentation**: Use FastAPI's automatic documentation features
- **Code Comments**: Explain complex logic and business rules
- **Docstrings**: Follow Google or NumPy docstring conventions
- **README Updates**: Keep the main README.md current with new features

### Documentation Examples

```python
async def chat_endpoint(chat_request: ChatRequest) -> ChatResponse:
    """Process a chat request and return AI-generated response.
    
    Args:
        chat_request: The chat request containing question and parameters
        
    Returns:
        ChatResponse: Complete response with answer, resources, and metadata
        
    Raises:
        HTTPException: If request validation fails or processing errors occur
        
    Example:
        >>> request = ChatRequest(question="What is Python?")
        >>> response = await chat_endpoint(request)
        >>> print(response.answer)
    """
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected vs actual behavior**
4. **Environment details** (Python version, OS, etc.)
5. **Error messages** and stack traces
6. **Minimal code example** if applicable

## 💡 Feature Requests

For new features, please provide:

1. **Clear description** of the proposed feature
2. **Use case** and motivation
3. **Proposed implementation** approach
4. **Potential challenges** or considerations
5. **Alternative solutions** considered

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub Contributors** page

## 📞 Getting Help

If you need help:

- **Discord/Slack**: [Community chat link]
- **GitHub Issues**: For bugs and feature requests
- **Email**: [maintainer email]
- **Documentation**: Check existing docs and examples

## 🔄 Review Process

1. **Automated checks** must pass (tests, linting, etc.)
2. **Peer review** by at least one maintainer
3. **Manual testing** for significant changes
4. **Documentation review** for public-facing changes

Thank you for contributing to RAG Tutor Chatbot! Your contributions help make educational AI more accessible and effective for everyone. 🎓✨
