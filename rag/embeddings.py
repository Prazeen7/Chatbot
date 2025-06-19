import ollama
import numpy as np
from typing import List, Tuple
from rag.exceptions import EmbeddingError

class EmbeddingManager:
    def __init__(self, model_name: str, ollama_host: str):
        self.model_name = model_name
        self.client = ollama.Client(host=ollama_host)
        self._verify_model()

    def _verify_model(self):
        try:
            available_models = [m['name'] for m in self.client.list().get('models', [])]
            if self.model_name not in available_models:
                raise EmbeddingError(f"Model {self.model_name} not found. Available models: {available_models}")
        except Exception as e:
            raise EmbeddingError(f"Failed to verify model: {str(e)}")

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            return response['embedding']
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        a_np, b_np = np.array(a), np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))