# retrieval.py
from typing import List, Tuple
import ollama
from rag.embeddings import get_embedding, cosine_similarity
from rag.exceptions import RetrievalError

def initialize_vector_db(embedder_client, model_name: str, documents: List[str]) -> List[Tuple[str, List[float]]]:
    vector_db = []
    for doc in documents:
        try:
            embedding = get_embedding(embedder_client, model_name, doc)
            vector_db.append((doc, embedding))
        except Exception as e:
            continue  # Skip failed embeddings
    return vector_db

def retrieve(vector_db, embedder_client, model_name: str, query: str, top_n: int = 3) -> List[Tuple[str, float]]:
    try:
        query_embed = get_embedding(embedder_client, model_name, query)
        scored = [
            (chunk, cosine_similarity(query_embed, embed))
            for chunk, embed in vector_db
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    except Exception as e:
        raise RetrievalError(f"Failed to retrieve documents: {str(e)}")

def generate_answer(vector_db, embedder_client, embedding_model: str, language_model: str, query: str, top_n: int = 3) -> str:
    try:
        relevant = retrieve(vector_db, embedder_client, embedding_model, query, top_n)
        if not relevant:
            return "No relevant information found in the knowledge base."

        context = "\n".join(f"- {chunk}" for chunk, score in relevant)
        
        response = ollama.Client().chat(
            model=language_model,
        messages=[{
                'role': 'system',
                'content': f"""Answer STRICTLY using only this context:
                {context}

                Response Rules:
                1. Answer directly with technical authority
                2. If context doesn't EXACTLY match the question, use uncertain response
                3. Never guess or invent information
                4. For procedures:
                   - Use numbered steps
                   - No extra line breaks
                5. For other info: concise paragraphs"""
            }, {
                'role': 'user', 
                'content': query
            }]
        )
        return response['message']['content']
    except Exception as e:
        raise RetrievalError(f"Failed to generate answer: {str(e)}")