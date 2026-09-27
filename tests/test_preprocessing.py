"""
Module Name: test_preprocessing.py
Aim: Verify text cleaning, stopword removal, punctuation handling, and lemmatization.
Input: Representative raw text strings.
Output: Passing or failing unittest results.
"""

import os
import sys
import unittest

# Ensure src is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import clean_text


class TestPreprocessing(unittest.TestCase):
    """Test suite for clean_text function in preprocessing module."""

    def test_empty_string(self):
        """Test that empty strings and whitespace-only strings return empty results."""
        cleaned_str, tokens = clean_text("")
        self.assertEqual(cleaned_str, "")
        self.assertEqual(tokens, [])

        cleaned_str_ws, tokens_ws = clean_text("   \n\t  ")
        self.assertEqual(cleaned_str_ws, "")
        self.assertEqual(tokens_ws, [])

    def test_all_stopwords(self):
        """Test that strings containing only stopwords return empty results."""
        stopword_input = "the and in on at which for with about against between into through during"
        cleaned_str, tokens = clean_text(stopword_input)
        self.assertEqual(cleaned_str, "")
        self.assertEqual(tokens, [])

    def test_mixed_case_and_punctuation(self):
        """Test lowercasing, punctuation removal, and digit removal."""
        raw_text = "Software Engineer (Python, C++ & Java)! 2024. Contact: dev@example.com, https://github.com"
        cleaned_str, tokens = clean_text(raw_text)

        # Expected tokens must be lowercase, no digits, no punctuation, no urls, no stopwords
        self.assertIsInstance(cleaned_str, str)
        self.assertIsInstance(tokens, list)
        self.assertIn("software", tokens)
        self.assertIn("engineer", tokens)
        self.assertIn("python", tokens)
        self.assertIn("java", tokens)

        # Assert punctuation and numbers are not in tokens
        for token in tokens:
            self.assertTrue(token.isalpha())
            self.assertTrue(token.islower())

        self.assertNotIn("2024", tokens)
        self.assertNotIn("https", tokens)
        self.assertNotIn("@", cleaned_str)
        self.assertNotIn("!", cleaned_str)

    def test_lemmatization(self):
        """Test that word forms are lemmatized to their root form."""
        raw_text = "Experienced developers building scalable applications and services"
        cleaned_str, tokens = clean_text(raw_text)
        self.assertIn("developer", tokens)
        self.assertIn("application", tokens)
        self.assertIn("service", tokens)


if __name__ == "__main__":
    unittest.main()
