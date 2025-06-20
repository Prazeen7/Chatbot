# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from rag.document_processor import load_documents
from rag.embeddings import verify_model, get_embedding, cosine_similarity
from rag.retrieval import initialize_vector_db, retrieve, generate_answer
from config import Config
import logging
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize RAG components
try:
    logger.info("Initializing RAG system...")
    documents = load_documents(Config.KNOWLEDGE_PDF)
    client = ollama.Client(host=Config.OLLAMA_SERVER)
    verify_model(client, Config.EMBEDDING_MODEL)
    vector_db = initialize_vector_db(client, Config.EMBEDDING_MODEL, documents)
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
        answer = generate_answer(
            vector_db,
            client,
            Config.EMBEDDING_MODEL,
            Config.LANGUAGE_MODEL,
            query,
            top_n=Config.TOP_N_RESULTS
        )
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)