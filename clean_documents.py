#!/usr/bin/env python3
"""
Script to remove documents with specific filename from documents.json
"""

import json

def clean_documents(filename_to_remove):
    """Remove all documents with the specified filename"""
    
    # Read the current documents
    try:
        with open('documents.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
    except FileNotFoundError:
        print("❌ documents.json not found")
        return
    except json.JSONDecodeError:
        print("❌ Invalid JSON in documents.json")
        return
    
    print(f"📊 Original document count: {len(documents)}")
    
    # Find documents to remove
    documents_to_remove = []
    for i, doc in enumerate(documents):
        if doc.get('metadata', {}).get('filename') == filename_to_remove:
            documents_to_remove.append(i)
    
    if not documents_to_remove:
        print(f"✅ No documents found with filename: {filename_to_remove}")
        return
    
    print(f"🗑️  Found {len(documents_to_remove)} documents to remove")
    
    # Remove documents (in reverse order to maintain indices)
    for i in reversed(documents_to_remove):
        removed_doc = documents.pop(i)
        print(f"   Removed document {i}: {removed_doc.get('metadata', {}).get('filename', 'Unknown')}")
    
    # Update document IDs
    for i, doc in enumerate(documents):
        doc['id'] = i
    
    # Save the cleaned documents
    try:
        with open('documents.json', 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved {len(documents)} documents")
        print(f"📊 New document count: {len(documents)}")
        print(f"🗑️  Removed {len(documents_to_remove)} documents")
    except Exception as e:
        print(f"❌ Error saving documents: {e}")

if __name__ == "__main__":
    filename_to_remove = "matome_n3_kanji_pdf-free.pdf"
    clean_documents(filename_to_remove) 