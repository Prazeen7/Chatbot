from rag.embeddings import get_embedding, cosine_similarity
from warnings import warn
import ollama
import time
from typing import List, Tuple, Optional
import numpy as np

# Constants for retry mechanism
MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds
MIN_SIMILARITY_THRESHOLD = 0.15  # Minimum similarity score to consider

def initialize_vector_db(embedder_client, model_name: str, documents: List[str]) -> List[Tuple[str, List[float]]]:
    """
    Initialize vector database with retry mechanism and progress tracking
    """
    vector_db = []
    total_docs = len(documents)
    
    for idx, doc in enumerate(documents, 1):
        for attempt in range(MAX_RETRIES):
            try:
                embedding = get_embedding(embedder_client, model_name, doc)
                vector_db.append((doc, embedding))
                print(f"Processed document {idx}/{total_docs}")  # Progress indicator
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    warn(f"Failed to embed document after {MAX_RETRIES} attempts: {str(e)[:100]}...")
                time.sleep(RETRY_DELAY * (attempt + 1))
    
    if not vector_db:
        raise ValueError("Vector database initialization failed - no valid embeddings generated")
    
    return vector_db

def retrieve(
    vector_db: List[Tuple[str, List[float]]],
    embedder_client,
    model_name: str,
    query: str,
    top_n: int = 3,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD
) -> List[Tuple[str, float]]:
    """
    Enhanced retrieval with similarity threshold and fallback mechanism
    """
    for attempt in range(MAX_RETRIES):
        try:
            query_embed = get_embedding(embedder_client, model_name, query)
            
            # Calculate similarities using numpy for better performance
            chunks, embeddings = zip(*vector_db)
            embeddings_array = np.array(embeddings)
            query_array = np.array(query_embed)
            
            # Batch cosine similarity calculation
            norms = np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_array)
            similarities = np.dot(embeddings_array, query_array) / norms
            
            # Filter and sort results
            scored = [(chunks[i], float(similarities[i])) 
                     for i in range(len(chunks)) 
                     if similarities[i] > similarity_threshold]
            
            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)
            
            return scored[:top_n]
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                warn(f"Retrieval failed after {MAX_RETRIES} attempts: {str(e)}")
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))

def generate_answer(
    vector_db: List[Tuple[str, List[float]]],
    embedder_client,
    embedding_model: str,
    language_model: str,
    query: str,
    top_n: int = 3,
    fallback_response: Optional[str] = None
) -> str:
    """
    Robust answer generation with context validation and fallback options
    """
    if fallback_response is None:
        fallback_response = (
            "I couldn't find enough relevant information in my knowledge base "
            "to answer your question accurately."
        )
    
    try:
        relevant = retrieve(vector_db, embedder_client, embedding_model, query, top_n)
        
        if not relevant:
            return fallback_response
        
        # Validate that we have meaningful results
        best_score = relevant[0][1]
        if best_score < MIN_SIMILARITY_THRESHOLD * 1.5:  # Slightly higher bar for generation
            return fallback_response
        
        # Build context with scores for debugging (removed in final output)
        context = "\n".join(
            f"[Relevance: {score:.2f}] {chunk}" 
            for chunk, score in relevant
        )
        
        # Enhanced prompt template
        prompt_template = f"""You are a senior technical expert. Use ONLY the following context to answer:

{context}

Response Requirements:
1. Be technically precise and concise
2. If context doesn't perfectly match, say "Based on similar information..."
3. NEVER invent technical details
4. For procedures:
   - Use clear numbered steps
   - Include safety warnings if present in context
5. For specifications:
   - Use exact values from context
   - Mark approximations with "~" if not exact
6. If conflicting info exists, say "Sources differ on this..."
7. Use simple, clear English"""

        for attempt in range(MAX_RETRIES):
            try:
                response = ollama.Client().chat(
                    model=language_model,
                    messages=[{
                        'role': 'system',
                        'content': prompt_template
                    }, {
                        'role': 'user', 
                        'content': query
                    }],
                    options={
                        'temperature': 0.3,  # Lower for more deterministic answers
                        'num_ctx': 4096  # Ensure we can handle larger contexts
                    }
                )
                return response['message']['content']
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY * (attempt + 1))
                
    except Exception as e:
        warn(f"Answer generation failed: {str(e)}")
        return f"System error: {str(e)[:200]}" if DEBUG else fallback_response