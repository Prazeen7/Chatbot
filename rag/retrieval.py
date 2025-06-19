from config import *
from rag.embeddings import get_embedding, cosine_similarity
import re

def initialize_vector_db(documents, client, model_name):
    vector_db = []
    for doc in documents:
        try:
            embedding = get_embedding(client, doc, model_name)
            vector_db.append((doc, embedding))
        except Exception:
            continue
    return vector_db

def retrieve(query, vector_db, client, model_name, top_n=3):
    try:
        query_embed = get_embedding(client, query, model_name)
        scored = [
            (chunk, cosine_similarity(query_embed, embed))
            for chunk, embed in vector_db
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    except Exception as e:
        raise ValueError(f"Failed to retrieve documents: {str(e)}")

def generate_answer(query, vector_db, client, top_n=3):
    try:
        relevant = retrieve(query, vector_db, client, EMBEDDING_MODEL, top_n)
        if not relevant:
            return format_uncertain_response()
        
        context = "\n".join(f"{chunk}" for chunk, score in relevant)
        
        response = client.chat(
            model=LANGUAGE_MODEL,
            messages=[{
                'role': 'system',
                'content': f"""You are a senior Galaincha technical expert. Answer STRICTLY using only this context:
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
        
        # Verify answer is fully supported by context
        if not is_answer_in_context(response['message']['content'], context):
            return format_uncertain_response()
            
        return format_expert_response(response['message']['content'])
    
    except Exception:
        return format_uncertain_response()

def format_uncertain_response():
    return ("I'm unsure about your specific concern. For precise technical support, please:\n\n"
            "• Email: info@galaincha.com.np (from your registered Galaincha email)\n"
            "• Phone: +977 9818502734\n\n"
            "Our support team will assist you with verified information.")

def is_answer_in_context(answer, context):
    """Verify the generated answer is fully supported by context"""
    answer_keywords = set(re.findall(r'\w{4,}', answer.lower()))
    context_keywords = set(re.findall(r'\w{4,}', context.lower()))
    return len(answer_keywords - context_keywords) < 3  # Allow minor connective words

def format_expert_response(text):
    """Format verified technical responses"""
    # Clean step formatting
    text = re.sub(r'(\d+\.)\s+', r'\1 ', text)
    # Remove any accidental reference phrases
    text = re.sub(r'\b(according to|per|based on)\b.+\n?', '', text, flags=re.IGNORECASE)
    return text.strip()