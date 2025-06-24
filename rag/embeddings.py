# embeddings.py
import ollama
import numpy as np
from typing import List, Tuple
from rag.exceptions import EmbeddingError

def verify_model(client, model_name: str):
    try:
        available_models = [m['name'] for m in client.list().get('models', [])]
        if model_name not in available_models:
            raise EmbeddingError(f"Model {model_name} not found. Available models: {available_models}")
    except Exception as e:
        raise EmbeddingError(f"Failed to verify model: {str(e)}")

def get_embedding(client, model_name: str, text: str) -> List[float]:
    try:
        response = client.embeddings(model=model_name, prompt=text)
        return response['embedding']
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embedding: {str(e)}")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_np, b_np = np.array(a), np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
