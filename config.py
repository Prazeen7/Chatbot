import os

# Application settings
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# RAG settings
EMBEDDING_MODEL = 'nomic-embed-text:latest'
LANGUAGE_MODEL = 'llama3:latest'
DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
OLLAMA_SERVER = 'http://localhost:11434'
TOP_N_RESULTS = 3

# Retrieval settings
MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds
MIN_SIMILARITY_THRESHOLD = 0.15  # Minimum similarity score to consider
CONTEXT_WINDOW = 4096  # Tokens for LLM context

# Text processing
MAX_CHUNK_SIZE = 800  # Reduced for better embedding quality
MIN_CHUNK_SIZE = 50
CHUNK_OVERLAP = 80
MIN_CHARACTERS_FOR_EMBEDDING = 20  # Skip very short texts

# Paths
STATIC_FOLDER = 'static'
TEMPLATES_FOLDER = 'templates'