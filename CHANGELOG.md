# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Docker and Docker Compose support
- GitHub issue templates and PR templates
- Comprehensive documentation
- Performance comparison tools

### Changed
- Enhanced README with detailed feature overview
- Improved project structure documentation

## [1.0.0] - 2024-08-04

### Added
- **Dual Embedding RAG Systems**
  - TF-IDF RAG system (Port 5004)
  - SentenceTransformer RAG system (Port 5005)
  - `all-MiniLM-L6-v2` model integration

- **Multi-Format Document Support**
  - PDF files with OCR capabilities (Tesseract integration)
  - Word documents (.docx) with full text extraction
  - Text files (.txt) with multi-encoding support
  - CSV files with Bengali text support and statistical analysis
  - SQLite databases (.db) with complete schema analysis

- **Advanced Text Processing**
  - Semantic chunking system with intelligent boundaries
  - Text cleaning and normalization
  - Overlap management (150-character overlap)
  - Minimum/maximum chunk size optimization (200-800 characters)

- **RESTful API Endpoints**
  - `/health` - System status and configuration
  - `/documents` - List all stored documents
  - `/upload-document` - Upload and process documents
  - `/search` - Search documents with similarity scoring
  - `/generate` - Generate RAG-based responses

- **Data Management**
  - JSON-based persistent storage with UTF-8 support
  - Metadata tracking (file type, upload date, chunk information)
  - Automatic backup and persistence across server restarts

- **Performance Features**
  - Batch processing for efficient embedding generation
  - Real-time processing feedback with progress tracking
  - Memory management optimization for large document collections
  - Caching for persistent embeddings

- **Utility Scripts**
  - `reprocess_documents.py` - Enhanced chunking reprocessing
  - `compare_embeddings.py` - Embedding comparison tool
  - `test_chunking.py` - Chunking system testing
  - `clean_documents.py` - Document cleanup utilities
  - `create_sample_db.py` - Sample database creation
  - `view_database.py` - Database content viewer

- **Security & Reliability**
  - Secure filename processing
  - File size limits (100MB maximum)
  - Extension validation
  - Comprehensive error handling
  - Graceful degradation

### Technical Specifications
- **Python Version**: 3.13+
- **Flask Version**: 2.0+
- **Embedding Models**: 
  - TF-IDF with configurable features
  - SentenceTransformer `all-MiniLM-L6-v2` (384 dimensions)
- **OCR Engine**: Tesseract
- **File Size Limit**: 100MB
- **Chunk Size**: 200-800 characters with 150-character overlap

### Performance Metrics
- **TF-IDF**: Fast, lightweight, excellent for exact keyword matching
- **SentenceTransformer**: Advanced semantic understanding, better for paraphrases
- **Memory Usage**: Optimized for large document collections
- **Processing Speed**: Real-time with progress tracking

### Supported Platforms
- macOS (with Homebrew for Tesseract)
- Ubuntu/Debian Linux
- Windows (with WSL recommended)

## [0.9.0] - 2024-08-03

### Added
- Initial RAG system implementation
- Basic document upload functionality
- Simple search and retrieval
- In-memory document storage

### Changed
- Basic Flask API structure
- Simple embedding generation

## [0.8.0] - 2024-08-02

### Added
- Project initialization
- Basic requirements setup
- Simple Flask application structure

---

## Version History

- **1.0.0**: Production-ready dual-embedding RAG system
- **0.9.0**: Initial RAG implementation
- **0.8.0**: Project foundation

## Future Roadmap

### Planned Features (v1.1.0)
- [ ] Web UI for non-technical users
- [ ] Redis caching integration
- [ ] Authentication and API key management
- [ ] Advanced monitoring and metrics
- [ ] CI/CD pipeline

### Planned Features (v1.2.0)
- [ ] Additional document formats (Excel, PowerPoint, Markdown)
- [ ] Advanced chunking algorithms
- [ ] Hybrid search (keyword + semantic)
- [ ] Re-ranking capabilities
- [ ] Streaming responses

### Planned Features (v2.0.0)
- [ ] Distributed processing
- [ ] Multi-language model support
- [ ] Advanced analytics dashboard
- [ ] Enterprise features
- [ ] Cloud deployment support 