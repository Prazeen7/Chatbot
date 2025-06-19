import numpy as np
from typing import List

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    a_np, b_np = np.array(a), np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

def validate_model_available(client, model_name: str) -> bool:
    """Check if a model is available in Ollama"""
    available_models = [m['name'] for m in client.list().get('models', [])]
    return model_name in available_models