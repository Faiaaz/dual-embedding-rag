#!/usr/bin/env python3
"""
Script to reprocess all existing documents with enhanced chunking
"""

import json
import os
from app_document_rag import clean_text, chunk_text

def reprocess_documents():
    """Reprocess all documents in documents.json with enhanced chunking"""
    
    # Load current documents
    if not os.path.exists('documents.json'):
        print("❌ documents.json not found")
        return
    
    try:
        with open('documents.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return
    
    print(f"📊 Found {len(documents)} existing documents")
    
    # Group documents by filename to reprocess them together
    documents_by_file = {}
    standalone_docs = []
    
    for doc in documents:
        filename = doc.get('metadata', {}).get('filename')
        if filename and doc.get('metadata', {}).get('source') == 'file_upload':
            if filename not in documents_by_file:
                documents_by_file[filename] = []
            documents_by_file[filename].append(doc)
        else:
            standalone_docs.append(doc)
    
    print(f"📁 Found {len(documents_by_file)} files to reprocess")
    print(f"📄 Found {len(standalone_docs)} standalone documents")
    
    # Create backup
    backup_file = 'documents_backup.json'
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print(f"💾 Created backup: {backup_file}")
    
    # Reprocess each file
    new_documents = []
    
    # Keep standalone documents as they are
    for doc in standalone_docs:
        new_documents.append(doc)
    
    # Reprocess file-based documents
    for filename, file_docs in documents_by_file.items():
        print(f"\n🔄 Reprocessing: {filename}")
        
        # Get the original text by combining all chunks
        original_text = ""
        metadata = file_docs[0]['metadata'].copy()
        
        # Sort by chunk_id to maintain order
        sorted_docs = sorted(file_docs, key=lambda x: x.get('metadata', {}).get('chunk_id', 0))
        
        for doc in sorted_docs:
            original_text += doc['text'] + "\n\n"
        
        # Clean and chunk the text
        cleaned_text = clean_text(original_text)
        chunks = chunk_text(cleaned_text, chunk_size=800, overlap=150, min_chunk_size=200)
        
        print(f"  📝 Original chunks: {len(file_docs)}")
        print(f"  📝 New chunks: {len(chunks)}")
        
        # Create new documents for each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'reprocessed': True
            })
            
            new_doc = {
                'text': chunk,
                'metadata': chunk_metadata,
                'id': len(new_documents)
            }
            new_documents.append(new_doc)
    
    # Update IDs
    for i, doc in enumerate(new_documents):
        doc['id'] = i
    
    # Save new documents
    with open('documents.json', 'w', encoding='utf-8') as f:
        json.dump(new_documents, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Reprocessing complete!")
    print(f"📊 Original documents: {len(documents)}")
    print(f"📊 New documents: {len(new_documents)}")
    print(f"💾 Backup saved as: {backup_file}")

if __name__ == "__main__":
    reprocess_documents() 