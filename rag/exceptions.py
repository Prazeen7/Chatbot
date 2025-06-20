class RAGError(Exception):
    """Base exception for RAG system"""
    pass

class EmbeddingError(RAGError):
    """Exception for embedding-related errors"""
    pass

class RetrievalError(RAGError):
    """Exception for retrieval-related errors"""
    pass

class DocumentProcessingError(RAGError):
    """Exception for document processing errors"""
    pass