#!/usr/bin/env python3
"""
Test script for OCR-enabled PDF upload functionality
"""

import requests
import json

# API base URL
API_BASE_URL = "http://localhost:5004"

def test_health():
    """Test the health endpoint to check OCR availability"""
    print("🔍 Checking API health and OCR availability...")
    
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API Status: {data['status']}")
        print(f"📊 Documents: {data['documents_count']}")
        print(f"📁 Supported Formats: {', '.join(data['supported_formats'])}")
        print(f"🔍 OCR Available: {data['ocr_available']}")
        print(f"📝 OCR Note: {data['ocr_note']}")
        return data['ocr_available']
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def upload_pdf_with_ocr(pdf_file, topic="test", source="test_upload", use_ocr=True):
    """Upload a PDF file with optional OCR"""
    print(f"\n📤 Uploading {pdf_file} with OCR={use_ocr}...")
    
    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': f}
            data = {
                'topic': topic,
                'source': source,
                'use_ocr': str(use_ocr).lower()
            }
            
            response = requests.post(f"{API_BASE_URL}/upload-document", files=files, data=data)
            
            if response.status_code == 201:
                result = response.json()
                print(f"✅ Upload successful!")
                print(f"📄 File: {result['filename']}")
                print(f"📊 Type: {result['file_type']}")
                print(f"🔢 Chunks: {result['chunks_created']}")
                print(f"📈 Total Documents: {result['total_documents']}")
                return result
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"Error: {response.json()}")
                return None
                
    except FileNotFoundError:
        print(f"❌ File not found: {pdf_file}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def search_content(query, top_k=3):
    """Search for content in the uploaded documents"""
    print(f"\n🔍 Searching for: '{query}'")
    
    try:
        response = requests.post(f"{API_BASE_URL}/search", 
                               json={"query": query, "top_k": top_k})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['count']} results")
            
            for i, doc_result in enumerate(result['results'], 1):
                doc = doc_result['document']
                metadata = doc['metadata']
                similarity = doc_result['similarity']
                
                print(f"\n📄 Result {i} (Similarity: {similarity:.3f})")
                print(f"📁 File: {metadata.get('filename', 'Unknown')}")
                print(f"🔍 OCR Used: {metadata.get('ocr_used', False)}")
                print(f"📝 Text: {doc['text'][:200]}...")
            
            return result
        else:
            print(f"❌ Search failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main test function"""
    print("🚀 Testing OCR-enabled PDF upload functionality")
    print("=" * 50)
    
    # Check health and OCR availability
    ocr_available = test_health()
    
    if not ocr_available:
        print("\n⚠️  OCR is not available. To enable OCR:")
        print("1. Install Tesseract: brew install tesseract")
        print("2. Restart the server")
        print("\nContinuing with regular PDF upload test...")
    
    # Test with a sample PDF (if available)
    pdf_files = [
        "matome n3 kanji pdf-free.pdf",
        "Coffee_Machine_Program_Requirements.pdf", 
        "chapter 7 data encoding.pdf"
    ]
    
    for pdf_file in pdf_files:
        if upload_pdf_with_ocr(pdf_file, topic="ocr_test", use_ocr=ocr_available):
            # Search for content
            search_content("program requirements", 2)
            search_content("kanji", 2)
            break
    
    print("\n" + "=" * 50)
    print("✅ OCR test completed!")

if __name__ == "__main__":
    main() 