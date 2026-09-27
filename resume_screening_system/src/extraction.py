"""
Module Name: extraction.py
Aim: To extract raw text content from resumes and job descriptions across multiple file formats (PDF, DOCX, TXT).
Input: File paths or document streams (.pdf, .docx, .txt).
Output: Extracted text as raw strings.
"""

import os
import logging
import PyPDF2
import docx2txt

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text content from a PDF document.

    Args:
        file_path (str): Path to the PDF file on disk.

    Returns:
        str: Extracted plain text content from all pages, or empty string on failure.
    """
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.warning(f"PDF file is missing or empty: {file_path}")
            return ""

        extracted_text = []
        with open(file_path, "rb") as file_obj:
            reader = PyPDF2.PdfReader(file_obj)
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(page_text)

        return "\n".join(extracted_text).strip()
    except Exception as exc:
        logger.warning(f"Failed to extract text from PDF '{file_path}': {exc}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract raw text content from a Microsoft Word (.docx) document.

    Args:
        file_path (str): Path to the .docx file on disk.

    Returns:
        str: Extracted plain text content from the document, or empty string on failure.
    """
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.warning(f"DOCX file is missing or empty: {file_path}")
            return ""

        text = docx2txt.process(file_path)
        return text.strip() if text else ""
    except Exception as exc:
        logger.warning(f"Failed to extract text from DOCX '{file_path}': {exc}")
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract raw text content from a plain text (.txt) file.

    Args:
        file_path (str): Path to the .txt file on disk.

    Returns:
        str: Plain text content read from the file, or empty string on failure.
    """
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.warning(f"TXT file is missing or empty: {file_path}")
            return ""

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
            return file_obj.read().strip()
    except Exception as exc:
        logger.warning(f"Failed to extract text from TXT '{file_path}': {exc}")
        return ""


def extract_text(file_path: str) -> str:
    """
    Detect file format (.pdf, .docx, .txt) and extract raw text content safely.

    Args:
        file_path (str): Path to the resume or job description document.

    Returns:
        str: Extracted plain text from the document, or empty string if unreadable/corrupt.
    """
    try:
        if not isinstance(file_path, str) or not file_path.strip():
            logger.warning("Invalid or empty file path provided for text extraction.")
            return ""

        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return ""

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return extract_text_from_pdf(file_path)
        elif ext == ".docx":
            return extract_text_from_docx(file_path)
        elif ext == ".txt":
            return extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file format '{ext}' for file: {file_path}")
            return ""
    except Exception as exc:
        logger.warning(f"Unexpected error extracting text from '{file_path}': {exc}")
        return ""
