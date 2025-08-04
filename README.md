# 🚀 **Advanced Dual-Embedding RAG System**

> **A production-ready Retrieval-Augmented Generation (RAG) system with dual embedding capabilities, supporting multiple document formats and advanced text processing.**

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/yourusername/rag-system)

---

## 🌟 **Features Overview**

### 🧠 **Dual Embedding Systems**
- **TF-IDF RAG** (Port 5004): Fast, lightweight, perfect for exact keyword matching
- **SentenceTransformer RAG** (Port 5005): Advanced semantic understanding with `all-MiniLM-L6-v2`

### 📄 **Multi-Format Document Support**
- **PDF Files** with OCR capabilities (Tesseract integration)
- **Word Documents** (.docx) with full text extraction
- **Text Files** (.txt) with multi-encoding support
- **CSV Files** with Bengali text support and statistical analysis
- **SQLite Databases** (.db) with complete schema analysis

### 🔧 **Advanced Features**
- **Semantic Chunking**: Intelligent text segmentation with overlap
- **OCR Integration**: Image-based PDF processing
- **Real-time Processing**: Live embedding generation with progress tracking
- **Comprehensive API**: RESTful endpoints with detailed documentation
- **Data Persistence**: JSON-based storage with UTF-8 support

---

## 🎯 **Quick Start**

### Prerequisites
```bash
# Python 3.13+ required
python --version

# Install Tesseract for OCR (macOS)
brew install tesseract

# Install Tesseract for OCR (Ubuntu/Debian)
sudo apt-get install tesseract-ocr
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements_simple.txt
```

### Running the System
```bash
# Start TF-IDF RAG System (Port 5004)
python app_document_rag.py

# In another terminal, start SentenceTransformer RAG System (Port 5005)
python app_sentence_transformers_rag.py
```

### Test the System
```bash
# Health check
curl http://localhost:5004/health

# Upload a document
curl -X POST http://localhost:5004/upload-document \
  -F "file=@your_document.pdf" \
  -F "topic=your_topic"

# Search documents
curl -X POST http://localhost:5004/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 3}'

# Generate RAG response
curl -X POST http://localhost:5004/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "top_k": 3}'
```

---

## 📊 **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Processing     │    │   Storage       │
│   Upload        │───▶│   Pipeline       │───▶│   System        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Multi-Format  │    │   Dual Embedding │    │   JSON Files    │
│   Support       │    │   Systems        │    │   (UTF-8)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OCR/Text      │    │   TF-IDF +       │    │   SentenceTrans  │
│   Extraction    │    │   SentenceTrans  │    │   + ST version  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 🔧 **API Endpoints**

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System status and configuration |
| `/documents` | GET | List all stored documents |
| `/upload-document` | POST | Upload and process documents |
| `/search` | POST | Search documents with similarity scoring |
| `/generate` | POST | Generate RAG-based responses |

### Example API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:5004/health")
print(response.json())

# Upload document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    data = {"topic": "research_paper"}
    response = requests.post("http://localhost:5004/upload-document", 
                           files=files, data=data)

# Search documents
search_data = {
    "query": "machine learning algorithms",
    "top_k": 5
}
response = requests.post("http://localhost:5004/search", 
                        json=search_data)
results = response.json()

# Generate response
generate_data = {
    "query": "What are the main findings?",
    "top_k": 3
}
response = requests.post("http://localhost:5004/generate", 
                        json=generate_data)
answer = response.json()
```

---

## 📁 **Project Structure**

```
rag-system/
├── 📄 app_document_rag.py              # TF-IDF RAG system (Port 5004)
├── 🧠 app_sentence_transformers_rag.py # SentenceTransformer RAG (Port 5005)
├── 📋 requirements_simple.txt          # Python dependencies
├── 📊 documents.json                   # TF-IDF document storage
├── 🧠 documents_sentence_transformers.json # ST document storage
├── 🔧 reprocess_documents.py          # Enhanced chunking reprocessing
├── 🧪 compare_embeddings.py           # Embedding comparison tool
├── 🧪 test_chunking.py                # Chunking system testing
├── 🗑️ clean_documents.py              # Document cleanup utilities
├── 🗑️ clean_csv_documents.py          # CSV document cleanup
├── 🗄️ create_sample_db.py             # Sample database creation
├── 👁️ view_database.py                # Database content viewer
└── 📚 CHUNKING_IMPROVEMENTS.md        # Chunking system documentation
```

---

## 🎨 **Advanced Features**

### 🔍 **Semantic Chunking System**
- **Intelligent Boundaries**: Breaks text at natural sentence/paragraph boundaries
- **Overlap Management**: 150-character overlap for context continuity
- **Text Cleaning**: Normalizes whitespace, quotes, and formatting
- **Minimum/Maximum Sizes**: 200-800 character optimal chunks

### 🌐 **Multi-Language Support**
- **Bengali Text**: Full UTF-8 support for Bengali content
- **Unicode Handling**: Comprehensive character encoding support
- **International Content**: Ready for global document processing

### 📈 **Performance Optimization**
- **Batch Processing**: Efficient embedding generation
- **Progress Tracking**: Real-time processing feedback
- **Memory Management**: Optimized for large document collections
- **Caching**: Persistent embeddings for fast searches

---

## 🧪 **Testing & Comparison**

### Compare Embedding Systems
```bash
python compare_embeddings.py
```

**Sample Output:**
```
🚀 Starting Embedding Comparison Test
📝 Test 1: 'Dragon Ball Z characters'
🔤 TF-IDF Results: Top similarity: 0.7501
🧠 SentenceTransformer Results: Top similarity: 0.6752

📝 Test 2: 'main protagonists and heroes'
🔤 TF-IDF Results: Top similarity: 0.1042
🧠 SentenceTransformer Results: Top similarity: 0.4763
```

### Test Chunking System
```bash
python test_chunking.py
```

---

## 🔧 **Configuration**

### Environment Variables
```bash
# Optional: Set for production
export FLASK_ENV=production
export MAX_CONTENT_LENGTH=104857600  # 100MB file limit
```

### Model Configuration
- **TF-IDF**: Configurable features, n-grams, stop words
- **SentenceTransformer**: `all-MiniLM-L6-v2` (384 dimensions)
- **Chunking**: Adjustable size and overlap parameters

---

## 🚀 **Deployment**

### Production Setup
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5004 app_document_rag:app
gunicorn -w 4 -b 0.0.0.0:5005 app_sentence_transformers_rag:app
```

### Docker Support (Coming Soon)
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements_simple.txt .
RUN pip install -r requirements_simple.txt
COPY . .
EXPOSE 5004 5005
CMD ["python", "app_document_rag.py"]
```

---

## 📊 **Performance Metrics**

| Metric | TF-IDF | SentenceTransformer |
|--------|--------|-------------------|
| **Speed** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ |
| **Memory** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ |
| **Semantic Understanding** | ⚡⚡ | ⚡⚡⚡⚡⚡ |
| **Keyword Matching** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ |
| **Multilingual** | ⚡⚡⚡ | ⚡⚡⚡⚡⚡ |

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
python test_chunking.py
python compare_embeddings.py

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

---

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **SentenceTransformers**: For advanced semantic embeddings
- **PyMuPDF**: For robust PDF processing
- **Tesseract**: For OCR capabilities
- **Flask**: For the web framework
- **scikit-learn**: For TF-IDF implementation

---

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/yourusername/rag-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag-system/discussions)
- **Email**: your.email@example.com

---

## ⭐ **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/rag-system&type=Date)](https://star-history.com/#yourusername/rag-system&Date)

---

<div align="center">

**Made with ❤️ for the AI/ML Community**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/rag-system?style=social)](https://github.com/yourusername/rag-system)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/rag-system?style=social)](https://github.com/yourusername/rag-system)
[![GitHub issues](https://img.shields.io/github/issues/yourusername/rag-system)](https://github.com/yourusername/rag-system/issues)

</div> 