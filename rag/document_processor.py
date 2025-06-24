import re
import os
import pytesseract
from PIL import Image
import io
from pypdf import PdfReader
from warnings import warn
from typing import List
from config import Config

def validate_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

def check_ocr_capability() -> bool:
    try:
        import pytesseract
        from PIL import Image
        return True
    except ImportError:
        warn("OCR dependencies not found. Image processing disabled.")
        return False

def split_into_chunks(text: str) -> List[str]:
    # First split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > Config.MAX_CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = para[-Config.CHUNK_OVERLAP:] + " " if Config.CHUNK_OVERLAP else ""
            else:
                chunks.append(para[:Config.MAX_CHUNK_SIZE])
                current_chunk = para[Config.MAX_CHUNK_SIZE - Config.CHUNK_OVERLAP:] + " "
        else:
            current_chunk += " " + para if current_chunk else para
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return [chunk for chunk in chunks if len(chunk) >= Config.MIN_CHUNK_SIZE]

def process_images(page, can_ocr: bool) -> List[str]:
    if not can_ocr:
        return []
    
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
            warn(f"Image processing skipped: {str(e)[:100]}...")
    return img_texts

def save_chunks_to_file(chunks: List[str], output_file: str = "chunks.txt") -> str:
    """Save text chunks to a file, creating directories if needed."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"=== Chunk {i} ===\n{chunk}\n\n")
        
        print(f"Saved {len(chunks)} chunks to: {os.path.abspath(output_file)}")
        return output_file
    except Exception as e:
        warn(f"Failed to save chunks: {str(e)}")
        return ""

def load_documents(pdf_path: str, save_chunks: bool = False, output_file: str = "chunks.txt") -> List[str]:
    """Extract text from PDF without cleaning, optionally save chunks to file.
    Returns list of chunks and optionally saves them to specified file."""
    validate_pdf(pdf_path)
    can_ocr = check_ocr_capability()
    
    text_chunks = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            if text := page.extract_text():
                text_chunks.extend(split_into_chunks(text))
            if can_ocr:
                text_chunks.extend(process_images(page, can_ocr))
    except Exception as e:
        raise RuntimeError(f"PDF processing error: {e}")
    
    filtered_chunks = [chunk for chunk in text_chunks if chunk]
    
    if save_chunks:
        saved_path = save_chunks_to_file(filtered_chunks, output_file)
        if not saved_path:
            warn("Chunks were processed but not saved to file")
    
    return filtered_chunks