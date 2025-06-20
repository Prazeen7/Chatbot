import numpy as np
import ollama
from warnings import warn

def verify_model(client, model_name):
    try:
        available_models = [m['name'] for m in client.list().get('models', [])]
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not found. Available models: {available_models}")
    except Exception as e:
        warn(f"Failed to verify model: {str(e)}")
        raise

def get_embedding(client, model_name, text):
    try:
        response = client.embeddings(model=model_name, prompt=text)
        return response['embedding']
    except Exception as e:
        warn(f"Failed to generate embedding: {str(e)}")
        raise

def cosine_similarity(a, b):
    a_np, b_np = np.array(a), np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))