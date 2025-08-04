# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_simple.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_simple.txt

# Copy application code
COPY . .

# Create directories for document storage
RUN mkdir -p /app/data

# Expose ports for both RAG systems
EXPOSE 5004 5005

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV MAX_CONTENT_LENGTH=104857600

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5004/health || exit 1

# Default command (can be overridden)
CMD ["python", "app_document_rag.py"] 