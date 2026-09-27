"""
Module Name: test_extraction.py
Aim: Verify supported document extraction and error handling.
Input: Temporary valid, empty, corrupt, and nonexistent document paths.
Output: Passing or failing unittest results.
"""

import os
import sys
import unittest
import tempfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from extraction import extract_text, extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt


class TestExtraction(unittest.TestCase):
    """Test suite for extraction functions."""

    def test_nonexistent_file(self):
        """Test that nonexistent file path returns empty string without crashing."""
        result = extract_text("nonexistent_file_path_12345.pdf")
        self.assertEqual(result, "")

    def test_empty_and_corrupt_files(self):
        """Test handling of empty and corrupt files."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name
            f.write(b"corrupt pdf header not real pdf")

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            docx_path = f.name
            f.write(b"not a valid zip docx file")

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            txt_path = f.name
            f.write(b"Sample Resume Content Python Developer")

        try:
            # Corrupt PDF should return empty string and not raise exception
            self.assertEqual(extract_text(pdf_path), "")
            # Corrupt DOCX should return empty string and not raise exception
            self.assertEqual(extract_text(docx_path), "")
            # Valid TXT should return content
            self.assertEqual(extract_text(txt_path), "Sample Resume Content Python Developer")
        finally:
            os.remove(pdf_path)
            os.remove(docx_path)
            os.remove(txt_path)


if __name__ == "__main__":
    unittest.main()
