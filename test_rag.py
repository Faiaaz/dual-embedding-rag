import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_get_documents():
    """Test getting all documents"""
    print("=== Testing Get Documents ===")
    response = requests.get(f"{BASE_URL}/documents")
    print(f"Status: {response.status_code}")
    print(f"Total documents: {response.json()['count']}")
    print("Documents:")
    for doc in response.json()['documents']:
        print(f"  - ID {doc['id']}: {doc['text'][:100]}...")
    print()

def test_add_document():
    """Test adding a new document"""
    print("=== Testing Add Document ===")
    new_doc = {
        "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.",
        "metadata": {
            "topic": "machine_learning",
            "source": "educational",
            "difficulty": "beginner"
        }
    }
    
    response = requests.post(f"{BASE_URL}/documents", json=new_doc)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_search():
    """Test searching for documents"""
    print("=== Testing Search ===")
    queries = [
        "What is RAG?",
        "How do embeddings work?",
        "What is machine learning?",
        "Explain cosine similarity"
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
            print(f"     Metadata: {doc['metadata']}")
        print()

def test_generate():
    """Test generating responses"""
    print("=== Testing Generate Response ===")
    queries = [
        "What is RAG and how does it work?",
        "Explain vector embeddings",
        "What is the RAG pipeline?",
        "How does machine learning work?"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        response = requests.post(f"{BASE_URL}/generate", json={"query": query, "top_k": 2})
        print(f"Status: {response.status_code}")
        
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Context used: {len(result['context'])} documents")
        print()

def main():
    """Run all tests"""
    print("🚀 Testing Simple RAG API")
    print("=" * 50)
    
    try:
        test_health()
        test_get_documents()
        test_add_document()
        test_search()
        test_generate()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Make sure the server is running on http://localhost:5000")
        print("   Run: python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 