import os

class Config:
    # Application settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    
    # RAG settings
    EMBEDDING_MODEL = 'nomic-embed-text:latest'
    LANGUAGE_MODEL = 'llama3:latest'
    KNOWLEDGE_PDF = 'galaincha_manual.pdf'
    OLLAMA_SERVER = 'http://localhost:11434'
    TOP_N_RESULTS = 3
    
    # Text processing
    MAX_CHUNK_SIZE = 1000  # characters
    MIN_CHUNK_SIZE = 50    # characters
    CHUNK_OVERLAP = 100    # characters
    
    # Paths
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'