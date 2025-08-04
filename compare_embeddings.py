#!/usr/bin/env python3
"""
Script to compare TF-IDF vs SentenceTransformer performance
"""

import requests
import json
import time

def test_query(api_url, query, top_k=3):
    """Test a query on the given API endpoint"""
    try:
        response = requests.post(
            f"{api_url}/search",
            json={"query": query, "top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main comparison function"""
    
    print("🚀 Starting Embedding Comparison Test")
    print("Make sure both servers are running:")
    print("  - TF-IDF server on port 5004")
    print("  - SentenceTransformer server on port 5005")
    print()
    
    # Test queries
    test_queries = [
        "Dragon Ball Z characters",
        "main protagonists and heroes", 
        "villains and antagonists"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: '{query}'")
        print("-" * 40)
        
        # Test TF-IDF (port 5004)
        print("🔤 TF-IDF Results:")
        tfidf_result = test_query("http://localhost:5004", query)
        if "error" not in tfidf_result:
            if tfidf_result["results"]:
                best_match = tfidf_result["results"][0]
                print(f"  Top similarity: {best_match['similarity']:.4f}")
                print(f"  Source: {best_match['document']['metadata'].get('filename', 'sample')}")
            else:
                print("  No results found")
        else:
            print(f"  Error: {tfidf_result['error']}")
        
        # Test SentenceTransformer (port 5005)
        print("🧠 SentenceTransformer Results:")
        st_result = test_query("http://localhost:5005", query)
        if "error" not in st_result:
            if st_result["results"]:
                best_match = st_result["results"][0]
                print(f"  Top similarity: {best_match['similarity']:.4f}")
                print(f"  Source: {best_match['document']['metadata'].get('filename', 'sample')}")
            else:
                print("  No results found")
        else:
            print(f"  Error: {st_result['error']}")
        
        time.sleep(1)  # Brief pause between queries

if __name__ == "__main__":
    main() 