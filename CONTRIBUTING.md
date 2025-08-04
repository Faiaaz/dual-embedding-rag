# 🤝 Contributing to Advanced Dual-Embedding RAG System

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- 🐛 Reporting a bug
- 💡 Discussing the current state of the code
- 📝 Submitting a fix
- 🚀 Proposing new features
- 📚 Becoming a maintainer

## 🚀 Quick Start

### 1. Fork the Repository
Click the "Fork" button in the top-right corner of this repository.

### 2. Clone Your Fork
```bash
git clone https://github.com/YOUR_USERNAME/rag-system.git
cd rag-system
```

### 3. Set Up Development Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements_simple.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 4. Create a Feature Branch
```bash
git checkout -b feature/amazing-feature
```

## 📋 Development Guidelines

### Code Style
We use **Black** for code formatting and **Flake8** for linting:

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking (optional)
mypy .
```

### Testing
Always test your changes:

```bash
# Run existing tests
python test_chunking.py
python compare_embeddings.py

# Test the APIs
curl http://localhost:5004/health
curl http://localhost:5005/health
```

### Commit Messages
Use clear, descriptive commit messages:

```bash
# Good examples
git commit -m "feat: add support for Excel files"
git commit -m "fix: resolve memory leak in embedding generation"
git commit -m "docs: update API documentation"

# Bad examples
git commit -m "fix stuff"
git commit -m "update"
```

## 🐛 Bug Reports

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/rag-system/issues/new).

### Bug Report Template
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. macOS, Ubuntu]
 - Python Version: [e.g. 3.13]
 - Dependencies: [e.g. requirements_simple.txt]

**Additional context**
Add any other context about the problem here.
```

## 💡 Feature Requests

We love feature requests! Please use the [feature request template](https://github.com/yourusername/rag-system/issues/new?template=feature_request.md).

### Feature Request Guidelines
- **Be specific**: Describe exactly what you want to see
- **Explain the use case**: Why is this feature needed?
- **Consider implementation**: Is this feasible with current architecture?
- **Check existing issues**: Has this been requested before?

## 🔧 Pull Request Process

### 1. Update Documentation
- Update the README.md if needed
- Add comments to your code
- Update any relevant documentation

### 2. Test Your Changes
```bash
# Test both RAG systems
python app_document_rag.py &
python app_sentence_transformers_rag.py &

# Run your tests
python test_chunking.py
python compare_embeddings.py

# Test API endpoints
curl http://localhost:5004/health
curl http://localhost:5005/health
```

### 3. Submit Pull Request
1. Push your changes to your fork
2. Create a Pull Request
3. Fill out the PR template
4. Wait for review

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
- [ ] I have tested my changes locally
- [ ] I have added tests for new functionality
- [ ] All existing tests pass

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

## 🏗️ Development Areas

### High Priority
- **Performance Optimization**: Improve embedding generation speed
- **Error Handling**: Better error messages and recovery
- **Testing**: More comprehensive test coverage
- **Documentation**: API documentation and examples

### Medium Priority
- **New Document Formats**: Excel, PowerPoint, Markdown
- **Advanced Chunking**: More sophisticated text segmentation
- **Caching**: Redis integration for better performance
- **Authentication**: API key management

### Low Priority
- **Web UI**: Simple web interface for non-technical users
- **Docker**: Containerization support
- **Monitoring**: Metrics and logging
- **CI/CD**: Automated testing and deployment

## 🧪 Testing Guidelines

### Unit Tests
```python
# Example test structure
def test_chunking_system():
    """Test the enhanced chunking system"""
    text = "Sample text for testing chunking."
    chunks = chunk_text(text)
    assert len(chunks) > 0
    assert all(len(chunk) >= 200 for chunk in chunks)
```

### Integration Tests
```python
# Example API test
def test_upload_document():
    """Test document upload endpoint"""
    with open("test_document.txt", "rb") as f:
        response = requests.post(
            "http://localhost:5004/upload-document",
            files={"file": f},
            data={"topic": "test"}
        )
    assert response.status_code == 201
```

## 📚 Code Review Process

### What We Look For
- **Functionality**: Does the code work as intended?
- **Performance**: Is it efficient and scalable?
- **Security**: Are there any security vulnerabilities?
- **Maintainability**: Is the code readable and well-documented?
- **Testing**: Are there adequate tests?

### Review Checklist
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No security issues introduced
- [ ] Performance impact is acceptable

## 🎯 Getting Help

### Before Asking for Help
1. **Check the documentation**: README.md and inline comments
2. **Search existing issues**: Your question might already be answered
3. **Try the examples**: Run the provided test scripts

### Where to Get Help
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private or sensitive matters

## 🏆 Recognition

Contributors will be recognized in:
- **README.md**: List of contributors
- **Release Notes**: Credit for significant contributions
- **GitHub Profile**: Your contributions will be visible on your profile

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the Advanced Dual-Embedding RAG System! 🚀** 