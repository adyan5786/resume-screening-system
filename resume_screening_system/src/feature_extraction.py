"""
Module Name: feature_extraction.py
Aim: To transform preprocessed text documents into numerical feature representations using vectorization techniques.
Input: Collection or Series of preprocessed text documents (resumes and job descriptions).
Output: Numerical feature matrices (TF-IDF vectors) and fitted vectorizer artifacts.

Design Note:
    The primary vectorizer is fit on a COMBINED corpus (all cleaned resumes + the job description)
    so that IDF weights reflect the shared vocabulary of both document types. Each document is
    then individually transformed with that single fitted vectorizer, ensuring the resume vectors
    and JD vector live in the same feature space — a prerequisite for meaningful cosine similarity.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf_vectorizer(corpus: list, max_features: int = 5000, ngram_range: tuple = (1, 2)) -> object:
    """
    Initialize and fit a TF-IDF vectorizer on a text corpus.

    Intended usage: pass the combined corpus (cleaned resumes + job description)
    so that IDF weights are computed over the full shared vocabulary.

    Args:
        corpus (list): List of preprocessed text documents.
        max_features (int, optional): Maximum vocabulary size. Defaults to 5000.
        ngram_range (tuple, optional): Lower and upper boundary of n-values for n-grams. Defaults to (1, 2).

    Returns:
        object: Fitted sklearn TfidfVectorizer instance.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,        # Apply log(1 + tf) to dampen high raw frequencies
        strip_accents="unicode",
        analyzer="word",
        min_df=1,
    )
    vectorizer.fit(corpus)
    return vectorizer


def transform_tfidf(vectorizer: object, texts: list) -> object:
    """
    Transform input text documents into TF-IDF sparse matrix using a fitted vectorizer.

    Args:
        vectorizer (object): Fitted TfidfVectorizer instance.
        texts (list): List of preprocessed text documents to vectorize.

    Returns:
        object: Sparse matrix of TF-IDF feature representations.
    """
    return vectorizer.transform(texts)


def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Extract top salient keywords from text based on TF-IDF scoring.

    Args:
        text (str): Input preprocessed text document.
        top_n (int, optional): Number of top keywords to extract. Defaults to 10.

    Returns:
        list: List of top n keywords sorted by importance score.
    """
    # Fit on the single document (IDF becomes 1 for all terms; ranking is purely by TF)
    single_doc_vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer="word")
    tfidf_matrix = single_doc_vectorizer.fit_transform([text])

    feature_names = single_doc_vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    # Pair terms with scores and sort descending
    term_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [term for term, _ in term_scores[:top_n]]


def extract_features(corpus: list, max_features: int = 5000) -> tuple:
    """
    Fit vectorizer and transform text corpus into TF-IDF feature matrix in one step.

    Args:
        corpus (list): List of preprocessed text documents.
        max_features (int, optional): Maximum vocabulary size. Defaults to 5000.

    Returns:
        tuple: (feature_matrix, fitted_vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=1,
    )
    feature_matrix = vectorizer.fit_transform(corpus)
    return feature_matrix, vectorizer


def build_combined_vectors(
    cleaned_resumes: list,
    cleaned_jd: str,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
) -> tuple:
    """
    Fit a single TfidfVectorizer on the combined corpus (resumes + JD) and return
    individual vectors for each resume and the JD.

    This is the **primary entry point** for the similarity-ranking pipeline:
    by fitting on the union of all documents the IDF values correctly penalise
    terms that are common across both resumes and the job description.

    Args:
        cleaned_resumes (list): List of preprocessed resume strings.
        cleaned_jd (str): Preprocessed job description string.
        max_features (int, optional): Vocabulary cap. Defaults to 5000.
        ngram_range (tuple, optional): N-gram range. Defaults to (1, 2).

    Returns:
        tuple: (resume_vectors, jd_vector, fitted_vectorizer)
            - resume_vectors : sparse matrix, shape (n_resumes, n_features)
            - jd_vector      : sparse matrix, shape (1, n_features)
            - fitted_vectorizer: the fitted TfidfVectorizer for later inspection
    """
    combined_corpus = cleaned_resumes + [cleaned_jd]
    vectorizer = fit_tfidf_vectorizer(combined_corpus, max_features=max_features, ngram_range=ngram_range)

    resume_vectors = transform_tfidf(vectorizer, cleaned_resumes)
    jd_vector = transform_tfidf(vectorizer, [cleaned_jd])

    return resume_vectors, jd_vector, vectorizer
