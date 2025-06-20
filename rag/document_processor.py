import re
import os
import pytesseract
from PIL import Image
import io
from pypdf import PdfReader
from warnings import warn
from config import (
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_FOLDER
)

def validate_pdf_folder(data_folder):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found at {data_folder}")
    
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_folder}")
    return pdf_files

def check_ocr_capability():
    try:
        pytesseract.get_tesseract_version()
        return True
    except:
        warn("OCR dependencies not found. Image processing disabled.")
        return False

def split_into_chunks(text):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > MAX_CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = para[-CHUNK_OVERLAP:] + " " if CHUNK_OVERLAP else ""
            else:
                chunks.append(para[:MAX_CHUNK_SIZE])
                current_chunk = para[MAX_CHUNK_SIZE - CHUNK_OVERLAP:] + " "
        else:
            current_chunk += " " + para if current_chunk else para
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_SIZE]

def process_images(page, can_ocr):
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

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

def load_documents():
    pdf_files = validate_pdf_folder(DATA_FOLDER)
    can_ocr = check_ocr_capability()
    text_chunks = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_FOLDER, pdf_file)
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                if text := page.extract_text():
                    text_chunks.extend(split_into_chunks(text))
                if can_ocr:
                    text_chunks.extend(process_images(page, can_ocr))
        except Exception as e:
            warn(f"Failed to process {pdf_file}: {str(e)[:100]}...")
            continue
    
    return [clean_text(chunk) for chunk in text_chunks if chunk]