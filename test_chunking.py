#!/usr/bin/env python3
"""
Test script to demonstrate enhanced chunking capabilities
"""

from app_document_rag import clean_text, chunk_text

def test_chunking():
    """Test the enhanced chunking system with sample text"""
    
    # Sample text with various content types
    sample_text = """
    This is a sample document that demonstrates the enhanced chunking capabilities.
    
    Chapter 1: Introduction
    
    The first chapter introduces the main concepts. This paragraph contains important information about the topic. We will explore various aspects in detail.
    
    Chapter 2: Methodology
    
    Our methodology involves several steps:
    • Step 1: Data collection
    • Step 2: Analysis
    • Step 3: Validation
    
    Chapter 3: Results
    
    The results show significant improvements. The data indicates a 25% increase in performance. This is a remarkable achievement that demonstrates the effectiveness of our approach.
    
    Chapter 4: Conclusion
    
    In conclusion, we have successfully demonstrated the enhanced chunking system. The results are promising and suggest further research is warranted.
    
    References:
    1. Smith, J. (2024). "Advanced Text Processing"
    2. Johnson, A. (2024). "Chunking Algorithms"
    3. Brown, M. (2024). "Natural Language Processing"
    """
    
    print("🧪 Testing Enhanced Chunking System")
    print("=" * 50)
    
    # Test text cleaning
    print("\n1️⃣ Text Cleaning:")
    cleaned = clean_text(sample_text)
    print(f"Original length: {len(sample_text)} characters")
    print(f"Cleaned length: {len(cleaned)} characters")
    print("\nCleaned text preview:")
    print("-" * 30)
    print(cleaned[:300] + "..." if len(cleaned) > 300 else cleaned)
    
    # Test chunking with different parameters
    print("\n2️⃣ Chunking Results:")
    
    # Test 1: Default chunking
    chunks_default = chunk_text(cleaned)
    print(f"\n📝 Default chunks (800 chars, 150 overlap): {len(chunks_default)}")
    for i, chunk in enumerate(chunks_default, 1):
        print(f"  Chunk {i}: {len(chunk)} chars")
        print(f"    Preview: {chunk[:100]}...")
    
    # Test 2: Smaller chunks
    chunks_small = chunk_text(cleaned, chunk_size=400, overlap=100)
    print(f"\n📝 Small chunks (400 chars, 100 overlap): {len(chunks_small)}")
    for i, chunk in enumerate(chunks_small, 1):
        print(f"  Chunk {i}: {len(chunk)} chars")
        print(f"    Preview: {chunk[:80]}...")
    
    # Test 3: Larger chunks
    chunks_large = chunk_text(cleaned, chunk_size=1200, overlap=200)
    print(f"\n📝 Large chunks (1200 chars, 200 overlap): {len(chunks_large)}")
    for i, chunk in enumerate(chunks_large, 1):
        print(f"  Chunk {i}: {len(chunk)} chars")
        print(f"    Preview: {chunk[:120]}...")
    
    # Test semantic boundary detection
    print("\n3️⃣ Semantic Boundary Detection:")
    print("Chunks break at natural boundaries like:")
    print("• Sentence endings (. ! ?)")
    print("• Paragraph breaks (double newlines)")
    print("• Section headers")
    print("• List items")
    
    # Show overlap between chunks
    if len(chunks_default) > 1:
        print(f"\n4️⃣ Overlap Analysis:")
        chunk1_end = chunks_default[0][-100:]
        chunk2_start = chunks_default[1][:100]
        print(f"End of chunk 1: ...{chunk1_end}")
        print(f"Start of chunk 2: {chunk2_start}...")
        
        # Find overlap
        overlap_text = ""
        for i in range(min(100, len(chunk1_end))):
            if chunk1_end[-i:] == chunk2_start[:i]:
                overlap_text = chunk1_end[-i:]
        
        if overlap_text:
            print(f"Overlap found: '{overlap_text}' ({len(overlap_text)} chars)")
        else:
            print("No direct overlap found (semantic boundaries used)")

if __name__ == "__main__":
    test_chunking() 