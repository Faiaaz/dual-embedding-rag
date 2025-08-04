import requests
import json
import os

# API base URL (PDF-enabled version runs on port 5003)
BASE_URL = "http://localhost:5003"

def test_health():
    """Test the health endpoint"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_upload_pdf(pdf_path):
    """Test uploading a PDF file"""
    print(f"=== Testing PDF Upload: {pdf_path} ===")
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        print("Please provide a valid PDF file path")
        return
    
    # Prepare the file upload
    with open(pdf_path, 'rb') as f:
        files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
        data = {
            'topic': 'test_document',
            'source': 'upload_test'
        }
        
        response = requests.post(f"{BASE_URL}/upload-pdf", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_search_after_upload():
    """Test searching after PDF upload"""
    print("=== Testing Search After PDF Upload ===")
    
    # Try some generic queries that might match PDF content
    queries = [
        "What is this document about?",
        "Summarize the main points",
        "What are the key concepts?",
        "Tell me about the content"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        response = requests.post(f"{BASE_URL}/search", json={"query": query, "top_k": 2})
        print(f"Status: {response.status_code}")
        
        results = response.json()['results']
        for i, result in enumerate(results):
            doc = result['document']
            similarity = result['similarity']
            print(f"  {i+1}. Similarity: {similarity:.3f}")
            print(f"     Text: {doc['text'][:100]}...")
            if 'filename' in doc['metadata']:
                print(f"     Source: {doc['metadata']['filename']}")
        print()

def test_generate_after_upload():
    """Test generating responses after PDF upload"""
    print("=== Testing Generate After PDF Upload ===")
    
    queries = [
        "What is this document about?",
        "Summarize the main content",
        "What are the key points discussed?"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        response = requests.post(f"{BASE_URL}/generate", json={"query": query, "top_k": 2})
        print(f"Status: {response.status_code}")
        
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Context used: {len(result['context'])} documents")
        print()

def create_sample_pdf():
    """Create a simple sample PDF for testing"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        filename = "sample_document.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        
        # Add some sample content
        c.drawString(100, 750, "Sample Document for RAG Testing")
        c.drawString(100, 720, "This is a test document created to demonstrate")
        c.drawString(100, 700, "PDF processing capabilities in the RAG system.")
        c.drawString(100, 680, "")
        c.drawString(100, 660, "Key Concepts:")
        c.drawString(100, 640, "1. PDF text extraction using PyMuPDF")
        c.drawString(100, 620, "2. Text chunking for better processing")
        c.drawString(100, 600, "3. Vector embeddings and similarity search")
        c.drawString(100, 580, "4. RAG pipeline with document retrieval")
        c.drawString(100, 560, "")
        c.drawString(100, 540, "This document will be processed and added to")
        c.drawString(100, 520, "the knowledge base for question answering.")
        
        c.save()
        print(f"✅ Created sample PDF: {filename}")
        return filename
        
    except ImportError:
        print("❌ reportlab not available. Please install it with: pip install reportlab")
        return None

def main():
    """Run all tests"""
    print("🚀 Testing PDF-Enabled RAG API")
    print("=" * 50)
    
    try:
        # Test health
        test_health()
        
        # Check if we have a PDF to test with
        pdf_path = "sample_document.pdf"
        
        # Create sample PDF if it doesn't exist
        if not os.path.exists(pdf_path):
            print("Creating sample PDF for testing...")
            pdf_path = create_sample_pdf()
            if not pdf_path:
                print("❌ No PDF file available for testing")
                print("Please provide a PDF file path or install reportlab")
                return
        
        # Test PDF upload
        test_upload_pdf(pdf_path)
        
        # Test search after upload
        test_search_after_upload()
        
        # Test generate after upload
        test_generate_after_upload()
        
        print("✅ All tests completed!")
        print("\n🎉 Your PDF-enabled RAG API is working!")
        print("📚 You can now:")
        print("   - Upload PDF files for processing")
        print("   - Ask questions about PDF content")
        print("   - Get answers based on extracted text")
        print("\n🌐 API is running at: http://localhost:5003")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print("   Make sure the server is running: python app_pdf_rag.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 