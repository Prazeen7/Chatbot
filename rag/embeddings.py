import ollama
import numpy as np
from config import *

def verify_model(client, model_name):
    try:
        available_models = [m['name'] for m in client.list().get('models', [])]
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not found. Available models: {available_models}")
    except Exception as e:
        raise ValueError(f"Failed to verify model: {str(e)}")

def get_embedding(client, text, model_name):
    try:
        response = client.embeddings(model=model_name, prompt=text)
        return response['embedding']
    except Exception as e:
        raise ValueError(f"Failed to generate embedding: {str(e)}")

def cosine_similarity(a, b):
    a_np, b_np = np.array(a), np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))