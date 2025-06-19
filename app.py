from flask import Flask, render_template, request, jsonify
import ollama
import numpy as np
from pypdf import PdfReader
import pytesseract
from PIL import Image
import io
import re
import os
import subprocess
from warnings import warn

app = Flask(__name__)

# Configuration
EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3'
KNOWLEDGE_PDF = 'galaincha_manual.pdf'
OLLAMA_SERVER = 'http://localhost:11434'  # Change to your Ollama server IP
VECTOR_DB = []

# Initialize Ollama client
ollama_client = ollama.Client(host=OLLAMA_SERVER)

def check_dependencies():
    """Verify all required dependencies are installed"""
    try:
        import pytesseract
        from PIL import Image
        from pypdf import PdfReader
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pypdf pytesseract pillow")
        return False

    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        warn("Tesseract OCR not found in PATH. Image OCR will be disabled.")
        return False

CAN_OCR = check_dependencies()

def initialize_rag_system():
    """Initialize the RAG system"""
    print("Initializing RAG system...")
    
    if not os.path.exists(KNOWLEDGE_PDF):
        print(f"Error: PDF file not found at {KNOWLEDGE_PDF}")
        return False
    
    # Load models
    try:
        available = [m['name'] for m in ollama_client.list().get('models', [])]
        for model in [EMBEDDING_MODEL, LANGUAGE_MODEL]:
            if model not in available:
                print(f"Downloading {model}...")
                ollama_client.pull(model)
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return False
    
    # Process PDF
    chunks = extract_text_from_pdf()
    print(f"Extracted {len(chunks)} text chunks")
    
    for i, chunk in enumerate(chunks, 1):
        add_chunk_to_database(chunk)
        if i % 20 == 0:
            print(f"Processed {i}/{len(chunks)} chunks")
    
    return True

def extract_text_from_pdf():
    """Extract and clean text from PDF"""
    text_chunks = []
    try:
        reader = PdfReader(KNOWLEDGE_PDF)
        for page in reader.pages:
            if text := page.extract_text():
                text_chunks.extend(split_into_chunks(text))
            if CAN_OCR:
                text_chunks.extend(process_images(page))
    except Exception as e:
        print(f"PDF processing error: {e}")
    return [clean_text(chunk) for chunk in text_chunks if chunk]

def split_into_chunks(text):
    return [chunk.strip() for chunk in re.split(r'(?<=[.!?])\s+', text) if len(chunk) > 10]

def process_images(page):
    img_texts = []
    for image_file in page.images:
        try:
            img = Image.open(io.BytesIO(image_file.data))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img_text = pytesseract.image_to_string(img)
            if img_text.strip():
                img_texts.append(img_text.strip())
        except Exception as e:
            print(f"Image processing skipped: {str(e)[:100]}...")
    return img_texts

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

def add_chunk_to_database(chunk):
    try:
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
        VECTOR_DB.append((chunk, response['embedding']))
    except Exception as e:
        print(f"Failed to embed chunk: {str(e)[:100]}...")

def cosine_similarity(a, b):
    a_np, b_np = np.array(a), np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

def retrieve(query, top_n=3):
    try:
        query_embed = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=query)['embedding']
        scored = [(chunk, cosine_similarity(query_embed, embed)) for chunk, embed in VECTOR_DB]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get('question', '')
    
    if not query:
        return jsonify({'error': 'Empty question'}), 400
    
    relevant = retrieve(query)
    if not relevant:
        return jsonify({'answer': 'No relevant information found.'})
    
    context = "\n".join(f"- {chunk}" for chunk, score in relevant)
    
    try:
        response = ollama_client.chat(
            model=LANGUAGE_MODEL,
            messages=[{
                'role': 'system',
                'content': f"Answer using ONLY these facts:\n{context}\n\nIf unsure, say you don't know."
            }, {
                'role': 'user', 
                'content': query
            }]
        )
        return jsonify({'answer': response['message']['content']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if initialize_rag_system():
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize RAG system")