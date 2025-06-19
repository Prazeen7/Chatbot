from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
from config import *
from rag.document_processor import load_documents
from rag.embeddings import verify_model, get_embedding, cosine_similarity
from rag.retrieval import initialize_vector_db, generate_answer
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize RAG components
try:
    logger.info("Initializing RAG system...")
    documents = load_documents(KNOWLEDGE_PDF)
    client = ollama.Client(host=OLLAMA_SERVER)
    verify_model(client, EMBEDDING_MODEL)
    vector_db = initialize_vector_db(documents, client, EMBEDDING_MODEL)
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {str(e)}")
    raise RuntimeError(f"Failed to initialize RAG system: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/styles.css')
def serve_css():
    return send_from_directory('templates', 'styles.css')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get('question', '')
    
    if not query:
        return jsonify({'error': 'Empty question'}), 400
    
    try:
        answer = generate_answer(query, vector_db, client, TOP_N_RESULTS)
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)