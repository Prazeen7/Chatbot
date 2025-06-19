import re
import os
import pytesseract
from PIL import Image
import io
from pypdf import PdfReader
from warnings import warn
from typing import List
from config import Config

class DocumentProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self._validate_pdf()
        self.can_ocr = self._check_ocr_capability()

    def _validate_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")

    def _check_ocr_capability(self) -> bool:
        try:
            import pytesseract
            from PIL import Image
            return True
        except ImportError:
            warn("OCR dependencies not found. Image processing disabled.")
            return False

    def load_documents(self) -> List[str]:
        """Extract and clean text from PDF"""
        text_chunks = []
        try:
            reader = PdfReader(self.pdf_path)
            for page in reader.pages:
                if text := page.extract_text():
                    text_chunks.extend(self._split_into_chunks(text))
                if self.can_ocr:
                    text_chunks.extend(self._process_images(page))
        except Exception as e:
            raise RuntimeError(f"PDF processing error: {e}")
        
        return [self._clean_text(chunk) for chunk in text_chunks if chunk]

    def _split_into_chunks(self, text: str) -> List[str]:
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

    def _process_images(self, page) -> List[str]:
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

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)