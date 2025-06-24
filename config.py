class Config:
    # Application settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True

    # RAG settings
    EMBEDDING_MODEL = 'nomic-embed-text:latest'
    LANGUAGE_MODEL = 'llama3:latest'
    DATA_FOLDER = 'data'  # Folder containing PDF and DOCX files
    OLLAMA_SERVER = 'http://localhost:11434'
    TOP_N_RESULTS = 3

    # Text processing
    MAX_CHUNK_SIZE = 1000  # characters
    MIN_CHUNK_SIZE = 50    # characters
    CHUNK_OVERLAP = 100    # characters

    # Performance settings
    MAX_CONTEXT_LENGTH = 2048
    MAX_RESPONSE_LENGTH = 500
    ENABLE_CACHING = True
    CONCURRENT_PROCESSING = True