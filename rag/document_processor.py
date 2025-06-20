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

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

def load_documents(pdf_path: str) -> List[str]:
    """Extract and clean text from PDF"""
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
    
    return [clean_text(chunk) for chunk in text_chunks if chunk]