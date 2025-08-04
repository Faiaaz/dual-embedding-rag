#!/bin/bash

# 🚀 Advanced Dual-Embedding RAG System Setup Script
# This script will help you set up the RAG system quickly and easily

set -e  # Exit on any error

echo "🚀 Welcome to the Advanced Dual-Embedding RAG System Setup!"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.13+ is installed
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.13+ required, found $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
        else
            print_error "Python 3.13+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.13+"
        exit 1
    fi
}

# Check and install Tesseract
check_tesseract() {
    print_status "Checking Tesseract installation..."
    if command -v tesseract &> /dev/null; then
        TESSERACT_VERSION=$(tesseract --version | head -n 1)
        print_success "Tesseract found: $TESSERACT_VERSION"
    else
        print_warning "Tesseract not found. Installing..."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install tesseract
                print_success "Tesseract installed via Homebrew"
            else
                print_error "Homebrew not found. Please install Homebrew first:"
                echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
                print_success "Tesseract installed via apt-get"
            elif command -v yum &> /dev/null; then
                sudo yum install -y tesseract
                print_success "Tesseract installed via yum"
            else
                print_error "Package manager not supported. Please install Tesseract manually."
                exit 1
            fi
        else
            print_error "Unsupported OS. Please install Tesseract manually."
            exit 1
        fi
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf .venv
    fi
    
    $PYTHON_CMD -m venv .venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source .venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements_simple.txt
    print_success "Dependencies installed"
}

# Create data directory
create_data_dir() {
    print_status "Creating data directory..."
    mkdir -p data
    print_success "Data directory created"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    python -c "import flask, sklearn, sentence_transformers, fitz, docx, pandas, sqlite3; print('✅ All imports successful')"
    
    # Test Tesseract
    tesseract --version > /dev/null 2>&1 && echo "✅ Tesseract working" || echo "❌ Tesseract not working"
    
    print_success "Installation test completed"
}

# Show next steps
show_next_steps() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source .venv/bin/activate"
    echo ""
    echo "2. Start the TF-IDF RAG system:"
    echo "   python app_document_rag.py"
    echo ""
    echo "3. In another terminal, start the SentenceTransformer RAG system:"
    echo "   python app_sentence_transformers_rag.py"
    echo ""
    echo "4. Test the systems:"
    echo "   curl http://localhost:5004/health"
    echo "   curl http://localhost:5005/health"
    echo ""
    echo "5. Upload a document:"
    echo "   curl -X POST http://localhost:5004/upload-document \\"
    echo "     -F \"file=@your_document.pdf\" \\"
    echo "     -F \"topic=your_topic\""
    echo ""
    echo "6. Search documents:"
    echo "   curl -X POST http://localhost:5004/search \\"
    echo "     -H \"Content-Type: application/json\" \\"
    echo "     -d '{\"query\": \"your search query\", \"top_k\": 3}'"
    echo ""
    echo "📚 For more information, see the README.md file"
    echo "🐛 For issues, check the GitHub repository"
    echo ""
}

# Main setup function
main() {
    echo "Starting setup process..."
    echo ""
    
    check_python
    check_tesseract
    create_venv
    activate_venv
    install_dependencies
    create_data_dir
    test_installation
    show_next_steps
}

# Run main function
main "$@" 