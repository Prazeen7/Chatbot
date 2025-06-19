from typing import List, Tuple
import ollama
from rag.embeddings import EmbeddingManager
from rag.exceptions import RetrievalError

class Retriever:
    def __init__(self, embedder: EmbeddingManager, documents: List[str]):
        self.embedder = embedder
        self.documents = documents
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        self.vector_db = []
        for doc in self.documents:
            try:
                embedding = self.embedder.get_embedding(doc)
                self.vector_db.append((doc, embedding))
            except Exception as e:
                continue  # Skip failed embeddings

    def retrieve(self, query: str, top_n: int = 3) -> List[Tuple[str, float]]:
        try:
            query_embed = self.embedder.get_embedding(query)
            scored = [
                (chunk, self.embedder.cosine_similarity(query_embed, embed))
                for chunk, embed in self.vector_db
            ]
            return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}")

    def generate_answer(self, query: str, language_model: str, top_n: int = 3) -> str:
        try:
            relevant = self.retrieve(query, top_n)
            if not relevant:
                return "No relevant information found in the knowledge base."

            context = "\n".join(f"- {chunk}" for chunk, score in relevant)
            
            response = ollama.Client().chat(
                model=language_model,
                messages=[{
                    'role': 'system',
                    'content': f"Answer the question using ONLY these facts:\n{context}\n\nIf unsure, say you don't know."
                }, {
                    'role': 'user', 
                    'content': query
                }]
            )
            return response['message']['content']
        except Exception as e:
            raise RetrievalError(f"Failed to generate answer: {str(e)}")