"""
Module Name: preprocessing.py
Aim: To clean, normalize, and preprocess raw text data using standard Natural Language Processing (NLP) techniques.
Input: Raw text string of resumes or job descriptions.
Output: Tuple of cleaned text string and list of lemmatized word tokens.
"""

import re
import socket
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_resources() -> None:
    """
    Check and automatically download required NLTK datasets ('punkt', 'stopwords', 'wordnet') if missing.
    """
    packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    old_timeout = socket.getdefaulttimeout()
    for resource_path, package_name in packages:
        try:
            nltk.data.find(resource_path)
        except (LookupError, AttributeError):
            try:
                socket.setdefaulttimeout(5.0)
                nltk.download(package_name, quiet=True)
            except Exception:
                pass
            finally:
                socket.setdefaulttimeout(old_timeout)


_ensure_nltk_resources()
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> tuple:
    """
    Clean, lowercase, strip punctuation/digits, tokenize, remove stopwords, and lemmatize text.

    Args:
        text (str): Raw input text string.

    Returns:
        tuple: (cleaned_string, list_of_tokens)
    """
    if not isinstance(text, str) or not text.strip():
        return "", []

    _ensure_nltk_resources()

    # Convert to lowercase
    text_lower = text.lower()

    # Remove URLs and email addresses
    text_no_urls = re.sub(r"https?://\S+|www\.\S+", " ", text_lower)
    text_no_emails = re.sub(r"\S+@\S+", " ", text_no_urls)

    # Strip digits and punctuation
    text_no_digits = re.sub(r"\d+", " ", text_no_emails)
    text_no_punct = text_no_digits.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Normalize whitespace
    normalized_text = re.sub(r"\s+", " ", text_no_punct).strip()
    if not normalized_text:
        return "", []

    # Tokenize with nltk.word_tokenize
    try:
        tokens = nltk.word_tokenize(normalized_text)
    except Exception:
        tokens = normalized_text.split()

    # Retrieve English stopwords
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set()

    # Filter stopwords and lemmatize with WordNetLemmatizer
    cleaned_tokens = []
    for token in tokens:
        token_clean = token.strip()
        if token_clean and token_clean.isalpha() and token_clean not in stop_words:
            lemmatized_token = _lemmatizer.lemmatize(token_clean)
            if lemmatized_token:
                cleaned_tokens.append(lemmatized_token)

    cleaned_string = " ".join(cleaned_tokens)
    return cleaned_string, cleaned_tokens


def remove_stopwords(tokens: list, custom_stopwords: list = None) -> list:
    """
    Remove English and optional custom stopwords from token list.

    Args:
        tokens (list): List of word tokens.
        custom_stopwords (list, optional): Additional custom stop words. Defaults to None.

    Returns:
        list: Filtered list of tokens without stopwords.
    """
    _ensure_nltk_resources()
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set()

    if custom_stopwords:
        stop_words.update(set(custom_stopwords))

    return [tok for tok in tokens if tok.lower() not in stop_words]


def lemmatize_text(tokens: list) -> list:
    """
    Lemmatize tokens to root forms using WordNetLemmatizer.

    Args:
        tokens (list): List of word tokens.

    Returns:
        list: List of lemmatized tokens.
    """
    _ensure_nltk_resources()
    return [_lemmatizer.lemmatize(tok) for tok in tokens]


def tokenize_text(text: str) -> list:
    """
    Tokenize input string into word tokens.

    Args:
        text (str): Input text string.

    Returns:
        list: List of word tokens.
    """
    _ensure_nltk_resources()
    if not text:
        return []
    try:
        return nltk.word_tokenize(text)
    except Exception:
        return text.split()


def preprocess_pipeline(text: str) -> tuple:
    """
    Execute full text preprocessing pipeline on raw text.

    Args:
        text (str): Raw input text string.

    Returns:
        tuple: (cleaned_string, list_of_tokens)
    """
    return clean_text(text)
