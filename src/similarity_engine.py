"""
Module Name: similarity_engine.py
Aim: To compute similarity scores between candidate resume vectors and job description vectors for candidate ranking.
Input: Resume feature vectors and target job description feature vector.
Output: Similarity scores and ranked list of matched candidate resumes.

Design Note:
    This module implements three scoring methods with a shared interface so they can be
    benchmarked side-by-side:

    1. rank_resumes()           — PRIMARY method: cosine similarity on TF-IDF vectors.
    2. bow_cosine_score()       — BASELINE 1: cosine similarity on raw CountVectorizer (BoW) vectors.
    3. keyword_overlap_score()  — BASELINE 2: Jaccard-style overlap (shared unique tokens / JD unique tokens).

    All three return a pandas DataFrame with columns [resume_id, similarity_score], sorted
    descending by score, so results can be compared directly.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity


# ---------------------------------------------------------------------------
# Low-level helper
# ---------------------------------------------------------------------------

def compute_cosine_similarity(resume_vector, jd_vector) -> float:
    """
    Compute cosine similarity score between a single resume vector and job description vector.

    Args:
        resume_vector (array-like): Vectorized feature representation of a single resume.
            Accepts a dense array, 1-D numpy array, or sparse row.
        jd_vector (array-like): Vectorized feature representation of the job description.

    Returns:
        float: Cosine similarity score in [0.0, 1.0].
    """
    score = sk_cosine_similarity(resume_vector, jd_vector)
    # cosine_similarity returns a 2-D array; extract the scalar
    return float(score.ravel()[0])


# ---------------------------------------------------------------------------
# Primary ranking method — TF-IDF cosine similarity
# ---------------------------------------------------------------------------

def rank_resumes(resume_vectors, jd_vector, resume_ids: list = None) -> pd.DataFrame:
    """
    Rank candidate resumes against a job description using cosine similarity on TF-IDF vectors.

    This is the **primary similarity-ranking method** of the pipeline.

    Args:
        resume_vectors (array-like): Matrix of TF-IDF vectors, shape (n_resumes, n_features).
            Accepts sparse or dense matrices produced by TfidfVectorizer.transform().
        jd_vector (array-like): TF-IDF vector for the job description, shape (1, n_features).
        resume_ids (list, optional): Unique identifiers for each resume (e.g. filenames).
            Defaults to ["resume_0", "resume_1", ...].

    Returns:
        pd.DataFrame: Rows sorted descending by 'similarity_score', columns:
            - resume_id      (str)
            - similarity_score (float, range 0–1)
    """
    n = resume_vectors.shape[0]

    if resume_ids is None:
        resume_ids = [f"resume_{i}" for i in range(n)]

    if len(resume_ids) != n:
        raise ValueError(
            f"Length mismatch: resume_vectors has {n} rows but resume_ids has {len(resume_ids)} entries."
        )

    # Compute cosine similarity between every resume vector and the JD vector in one call
    scores = sk_cosine_similarity(resume_vectors, jd_vector).ravel()

    df = pd.DataFrame({"resume_id": resume_ids, "similarity_score": scores})
    df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Baseline 1 — Bag-of-Words cosine similarity (CountVectorizer)
# ---------------------------------------------------------------------------

def bow_cosine_score(cleaned_resumes: list, cleaned_jd: str, resume_ids: list = None) -> pd.DataFrame:
    """
    Baseline: rank resumes using cosine similarity on raw Bag-of-Words (CountVectorizer) vectors.

    Accepts the same logical inputs as rank_resumes() but works directly from cleaned text
    strings, fitting a CountVectorizer internally on the combined corpus.

    Args:
        cleaned_resumes (list): List of preprocessed resume strings.
        cleaned_jd (str): Preprocessed job description string.
        resume_ids (list, optional): Unique identifiers for each resume. Defaults to indexed names.

    Returns:
        pd.DataFrame: Rows sorted descending by 'similarity_score', columns:
            - resume_id      (str)
            - similarity_score (float, range 0–1)
    """
    n = len(cleaned_resumes)

    if resume_ids is None:
        resume_ids = [f"resume_{i}" for i in range(n)]

    combined_corpus = cleaned_resumes + [cleaned_jd]
    vectorizer = CountVectorizer(analyzer="word", min_df=1)
    vectorizer.fit(combined_corpus)

    resume_vectors = vectorizer.transform(cleaned_resumes)
    jd_vector = vectorizer.transform([cleaned_jd])

    scores = sk_cosine_similarity(resume_vectors, jd_vector).ravel()

    df = pd.DataFrame({"resume_id": resume_ids, "similarity_score": scores})
    df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Baseline 2 — Keyword overlap score
# ---------------------------------------------------------------------------

def keyword_overlap_score(cleaned_resumes: list, cleaned_jd: str, resume_ids: list = None) -> pd.DataFrame:
    """
    Baseline: rank resumes by keyword overlap ratio (shared unique tokens / JD unique tokens).

    Score = |tokens(resume) ∩ tokens(jd)| / |tokens(jd)|

    This is a simple, interpretable baseline that requires no vectorization.
    Accepts the same logical inputs as rank_resumes() for direct comparison.

    Args:
        cleaned_resumes (list): List of preprocessed resume strings.
        cleaned_jd (str): Preprocessed job description string.
        resume_ids (list, optional): Unique identifiers for each resume. Defaults to indexed names.

    Returns:
        pd.DataFrame: Rows sorted descending by 'similarity_score', columns:
            - resume_id      (str)
            - similarity_score (float, range 0–1)
    """
    n = len(cleaned_resumes)

    if resume_ids is None:
        resume_ids = [f"resume_{i}" for i in range(n)]

    jd_tokens = set(cleaned_jd.split())

    if not jd_tokens:
        # Degenerate case: empty JD — return zero scores
        return pd.DataFrame({"resume_id": resume_ids, "similarity_score": [0.0] * n})

    scores = []
    for resume_text in cleaned_resumes:
        resume_tokens = set(resume_text.split())
        overlap = resume_tokens & jd_tokens
        scores.append(len(overlap) / len(jd_tokens))

    df = pd.DataFrame({"resume_id": resume_ids, "similarity_score": scores})
    df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Utility — retained from original design for skill-level analysis
# ---------------------------------------------------------------------------

def match_skills(resume_text: str, required_skills: list) -> dict:
    """
    Compare skills present in resume against required job description skills.

    Args:
        resume_text (str): Preprocessed resume text.
        required_skills (list): List of required skill keywords from the job description.

    Returns:
        dict: Summary dictionary containing matched skills, missing skills, and match percentage.
    """
    resume_tokens = set(resume_text.lower().split())
    matched = [skill for skill in required_skills if skill.lower() in resume_tokens]
    missing = [skill for skill in required_skills if skill.lower() not in resume_tokens]
    match_pct = (len(matched) / len(required_skills) * 100) if required_skills else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": round(match_pct, 2),
    }
