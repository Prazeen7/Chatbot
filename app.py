from flask import Flask, render_template, request, jsonify, send_from_directory
from rag.document_processor import load_documents
from rag.embeddings import verify_model
from rag.retrieval import initialize_vector_db, generate_answer
import logging
import ollama
from config import (
    HOST, PORT, DEBUG,
    EMBEDDING_MODEL, LANGUAGE_MODEL,
    OLLAMA_SERVER, TOP_N_RESULTS,
    TEMPLATES_FOLDER
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize RAG components
try:
    logger.info("Initializing RAG system...")
    documents = load_documents()
    client = ollama.Client(host=OLLAMA_SERVER)
    verify_model(client, EMBEDDING_MODEL)
    vector_db = initialize_vector_db(client, EMBEDDING_MODEL, documents)
    logger.info(f"RAG system initialized with {len(documents)} chunks from data folder")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {str(e)}")
    raise RuntimeError(f"Failed to initialize RAG system: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/styles.css')
def serve_css():
    return send_from_directory(TEMPLATES_FOLDER, 'styles.css')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get('question', '')
    
    if not query:
        return jsonify({'error': 'Empty question'}), 400
    
    try:
        answer = generate_answer(
            vector_db,
            client,
            EMBEDDING_MODEL,
            LANGUAGE_MODEL,
            query,
            top_n=TOP_N_RESULTS
        )
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)