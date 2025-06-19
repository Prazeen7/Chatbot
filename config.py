# config.py
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True
EMBEDDING_MODEL = 'nomic-embed-text:latest'
LANGUAGE_MODEL = 'llama3:latest'
KNOWLEDGE_PDF = 'galaincha_manual.pdf'
OLLAMA_SERVER = 'http://localhost:11434'
TOP_N_RESULTS = 3
MAX_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 50
CHUNK_OVERLAP = 100