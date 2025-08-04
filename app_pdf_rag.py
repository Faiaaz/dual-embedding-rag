from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        text = ""
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for better processing"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

class PDFRAG:
    def __init__(self, storage_file='documents.json'):
        self.storage_file = storage_file
        # Use TF-IDF instead of sentence transformers for simplicity
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.documents = []
        self.embeddings = None
        self.is_fitted = False
        
        # Load existing documents from file
        self._load_documents()
        
    def _load_documents(self):
        """Load documents from storage file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    self.documents = json.load(f)
                print(f"Loaded {len(self.documents)} documents from {self.storage_file}")
                if self.documents:
                    self._update_embeddings()
            except Exception as e:
                print(f"Error loading documents: {e}")
                self.documents = []
        else:
            print(f"No existing documents found. Starting fresh.")
            self.documents = []
    
    def _save_documents(self):
        """Save documents to storage file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.documents, f, indent=2)
            print(f"Saved {len(self.documents)} documents to {self.storage_file}")
        except Exception as e:
            print(f"Error saving documents: {e}")
        
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
        
        # Save to file
        self._save_documents()
        
        return doc['id']
    
    def add_pdf_document(self, pdf_path: str, metadata: Dict[str, Any] = None) -> List[int]:
        """Add a PDF document to the knowledge base by extracting and chunking text"""
        if metadata is None:
            metadata = {}
        
        # Extract text from PDF
        full_text = extract_text_from_pdf(pdf_path)
        
        # Chunk the text
        chunks = chunk_text(full_text)
        
        # Add each chunk as a separate document
        doc_ids = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'source': 'pdf',
                'chunk_id': i,
                'total_chunks': len(chunks),
                'filename': os.path.basename(pdf_path)
            })
            
            doc_id = self.add_document(chunk, chunk_metadata)
            doc_ids.append(doc_id)
        
        return doc_ids
    
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

# Initialize RAG system with PDF support
rag = PDFRAG()

# Add sample documents only if no documents exist
if len(rag.documents) == 0:
    print("Adding sample documents...")
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
        'message': 'PDF-Enabled RAG API is running',
        'documents_count': len(rag.documents),
        'method': 'TF-IDF with PDF support',
        'storage_file': rag.storage_file,
        'supported_formats': list(ALLOWED_EXTENSIONS)
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
        'message': 'Document added successfully and saved to file',
        'document_id': doc_id,
        'total_documents': len(rag.documents),
        'storage_file': rag.storage_file
    }), 201

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """Upload and process a PDF file"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Only PDF files are supported.'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get metadata from form data
        metadata = {}
        if request.form.get('topic'):
            metadata['topic'] = request.form.get('topic')
        if request.form.get('source'):
            metadata['source'] = request.form.get('source')
        
        # Process the PDF
        doc_ids = rag.add_pdf_document(filepath, metadata)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({
            'message': 'PDF processed successfully',
            'filename': filename,
            'document_ids': doc_ids,
            'chunks_created': len(doc_ids),
            'total_documents': len(rag.documents),
            'storage_file': rag.storage_file
        }), 201
        
    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500

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
        'message': 'Welcome to the PDF-Enabled RAG API!',
        'note': 'This version supports PDF uploads and text extraction',
        'storage_file': rag.storage_file,
        'endpoints': {
            'GET /health': 'Health check',
            'GET /documents': 'Get all documents',
            'POST /documents': 'Add a new document',
            'POST /upload-pdf': 'Upload and process a PDF file',
            'POST /search': 'Search for relevant documents',
            'POST /generate': 'Generate a response using RAG'
        },
        'example_usage': {
            'add_document': {
                'method': 'POST',
                'endpoint': '/documents',
                'body': {'text': 'Your document text', 'metadata': {'topic': 'example'}}
            },
            'upload_pdf': {
                'method': 'POST',
                'endpoint': '/upload-pdf',
                'form_data': {'file': 'your_file.pdf', 'topic': 'example', 'source': 'upload'}
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
    app.run(debug=True, host='0.0.0.0', port=5003) 