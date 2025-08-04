from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from typing import List, Dict, Any

app = Flask(__name__)

class SimpleRAG:
    def __init__(self):
        # Use TF-IDF instead of sentence transformers for simplicity
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.documents = []
        self.embeddings = None
        self.is_fitted = False
        
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Add a document to the knowledge base"""
        if metadata is None:
            metadata = {}
        
        # Store document with metadata
        doc = {
            'text': text,
            'metadata': metadata,
            'id': len(self.documents)
        }
        self.documents.append(doc)
        
        # Re-fit the vectorizer with all documents
        self._update_embeddings()
        
        return doc['id']
    
    def _update_embeddings(self):
        """Update embeddings when documents change"""
        if not self.documents:
            return
        
        # Extract all document texts
        texts = [doc['text'] for doc in self.documents]
        
        # Fit vectorizer and create embeddings
        self.embeddings = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
    
    def search(self, query: str, top_k: int = 3):
        """Search for relevant documents using TF-IDF similarity"""
        if not self.documents or not self.is_fitted:
            return []
        
        # Transform query to vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include documents with some similarity
                results.append({
                    'document': self.documents[idx],
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def generate_response(self, query: str, top_k: int = 3):
        """Generate a response using retrieved context"""
        # Search for relevant documents
        search_results = self.search(query, top_k)
        
        if not search_results:
            return {
                'response': 'I don\'t have enough information to answer that question.',
                'context': [],
                'query': query
            }
        
        # Build context from retrieved documents
        context = []
        for result in search_results:
            context.append({
                'text': result['document']['text'],
                'similarity': result['similarity'],
                'metadata': result['document']['metadata']
            })
        
        # Simple response generation (in a real system, you'd use an LLM here)
        # For this demo, we'll just return the most relevant document
        best_match = context[0]
        
        response = f"Based on the most relevant information I found:\n\n{best_match['text']}\n\n(Similarity score: {best_match['similarity']:.3f})"
        
        return {
            'response': response,
            'context': context,
            'query': query
        }

# Initialize RAG system
rag = SimpleRAG()

# Add some sample documents to get started
sample_docs = [
    {
        'text': 'RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation to provide more accurate and contextual responses.',
        'metadata': {'topic': 'definition', 'source': 'educational'}
    },
    {
        'text': 'Vector embeddings are numerical representations of text that capture semantic meaning. They allow us to find similar documents by comparing their embeddings.',
        'metadata': {'topic': 'embeddings', 'source': 'technical'}
    },
    {
        'text': 'The RAG pipeline typically involves: 1) Storing documents with embeddings, 2) Searching for relevant documents when given a query, 3) Using retrieved context to generate a response.',
        'metadata': {'topic': 'pipeline', 'source': 'process'}
    },
    {
        'text': 'Cosine similarity is a common method for comparing vector embeddings. It measures the cosine of the angle between two vectors, ranging from -1 to 1.',
        'metadata': {'topic': 'similarity', 'source': 'mathematical'}
    },
    {
        'text': 'TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a statistical measure used to evaluate how important a word is to a document in a collection.',
        'metadata': {'topic': 'tfidf', 'source': 'statistical'}
    }
]

# Add sample documents to the RAG system
for doc in sample_docs:
    rag.add_document(doc['text'], doc['metadata'])

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Simple RAG API is running',
        'documents_count': len(rag.documents),
        'method': 'TF-IDF (simplified for Python 3.12 compatibility)'
    })

@app.route('/documents', methods=['GET'])
def get_documents():
    """Get all documents in the knowledge base"""
    return jsonify({
        'documents': rag.documents,
        'count': len(rag.documents)
    })

@app.route('/documents', methods=['POST'])
def add_document():
    """Add a new document to the knowledge base"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Text field is required'}), 400
    
    text = data['text']
    metadata = data.get('metadata', {})
    
    doc_id = rag.add_document(text, metadata)
    
    return jsonify({
        'message': 'Document added successfully',
        'document_id': doc_id,
        'total_documents': len(rag.documents)
    }), 201

@app.route('/search', methods=['POST'])
def search_documents():
    """Search for relevant documents"""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Query field is required'}), 400
    
    query = data['query']
    top_k = data.get('top_k', 3)
    
    results = rag.search(query, top_k)
    
    return jsonify({
        'query': query,
        'results': results,
        'count': len(results)
    })

@app.route('/generate', methods=['POST'])
def generate_response():
    """Generate a response using RAG"""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Query field is required'}), 400
    
    query = data['query']
    top_k = data.get('top_k', 3)
    
    response = rag.generate_response(query, top_k)
    
    return jsonify(response)

@app.route('/')
def home():
    """Home page with API documentation"""
    return jsonify({
        'message': 'Welcome to the Simple RAG API! (Python 3.12 Compatible)',
        'note': 'This version uses TF-IDF instead of sentence transformers for better compatibility',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /documents': 'Get all documents',
            'POST /documents': 'Add a new document',
            'POST /search': 'Search for relevant documents',
            'POST /generate': 'Generate a response using RAG'
        },
        'example_usage': {
            'add_document': {
                'method': 'POST',
                'endpoint': '/documents',
                'body': {'text': 'Your document text', 'metadata': {'topic': 'example'}}
            },
            'search': {
                'method': 'POST',
                'endpoint': '/search',
                'body': {'query': 'What is RAG?', 'top_k': 3}
            },
            'generate': {
                'method': 'POST',
                'endpoint': '/generate',
                'body': {'query': 'How does RAG work?', 'top_k': 3}
            }
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 