import os
from typing import Tuple, Optional, Dict, Any

class MultiProcessor:
    """Unified file processor that automatically handles PDF, Markdown, Excel, and Word files"""
    
    def __init__(self, tesseract_cmd_path: Optional[str] = None):
        """
        Initialize the multi-file processor
        
        Args:
            tesseract_cmd_path: Optional path to Tesseract OCR executable (for PDF/image text extraction)
        """
        self.tesseract_cmd_path = tesseract_cmd_path
        self._processors = {}  # Lazy initialization of processors

    def file2text(self, file_path: str, 
                 include_metadata: bool = False,
                 include_images: bool = False,
                 ocr_languages: str = 'eng+chi_sim') -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Extract text from a file (PDF, Markdown, Excel, or Word)
        
        Args:
            file_path: Path to the input file
            include_metadata: Whether to include file metadata
            include_images: Whether to extract text from images (PDF only)
            ocr_languages: Languages for OCR (e.g., 'eng+chi_sim')
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        # Initialize processor only when needed
        if ext == '.pdf':
            if 'pdf' not in self._processors:
                from PDFProcessor import PDFProcessor
                self._processors['pdf'] = PDFProcessor(self.tesseract_cmd_path)
            return self._processors['pdf'].extract_text_from_pdf(
                file_path, 
                include_metadata=include_metadata,
                include_images=include_images,
                ocr_languages=ocr_languages
            )
            
        elif ext == '.md':
            if 'md' not in self._processors:
                from MarkdownProcessor import MarkdownProcessor
                self._processors['md'] = MarkdownProcessor(self.tesseract_cmd_path)
            return self._processors['md'].extract_text_from_markdown(
                file_path,
                include_metadata=include_metadata,
                include_images=include_images,
                ocr_languages=ocr_languages
            )
            
        elif ext in ('.xlsx', '.xls'):
            if 'excel' not in self._processors:
                from ExcelProcessor import ExcelProcessor
                self._processors['excel'] = ExcelProcessor()
            return self._processors['excel'].extract_text_from_excel(
                file_path,
                include_metadata=include_metadata
            )
            
        elif ext in ('.docx', '.doc'):
            if 'word' not in self._processors:
                from WordProcessor import WordProcessor
                self._processors['word'] = WordProcessor()
            return self._processors['word'].extract_text_from_word(
                file_path,
                include_metadata=include_metadata,
                include_images=include_images,
                ocr_languages=ocr_languages
            )
        elif ext == '.txt':
            if 'txt' not in self._processors:
                from TxtProcessor import TxtProcessor
                self._processors['txt'] = TxtProcessor()
            return self._processors['txt'].extract_text_from_txt(
                file_path,
                include_metadata=include_metadata
            )
        elif ext == '.pptx':
            if 'pptx' not in self._processors:
                from PPTProcessor import PPTProcessor
                self._processors['pptx'] = PPTProcessor()
            return self._processors['pptx'].extract_text_from_ppt(
                file_path,
                include_metadata=include_metadata
            )
        elif ext in ('.bmp','.jpg','.png'):
            if 'image' not in self._processors:
                from ImageProcessor import ImageProcessor
                self._processors['image'] = ImageProcessor()
            return self._processors['image'].extract_text_from_image(
                file_path,
                include_metadata=include_metadata
            )
        else:
            raise ValueError(f"Unsupported file type: {ext}")